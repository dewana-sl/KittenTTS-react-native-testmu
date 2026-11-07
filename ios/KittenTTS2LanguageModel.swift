// Kitten TTS 2's speech language model on the GPU, through MLX: a port of the
// Python package's MLX backend (kittentts2/mlx/loader.py and model.py).
//
// The checkpoint's linear weights are ternary: in every 128-wide group each value
// is -scale, 0 or +scale. MLX multiplies by a 4-bit weight as scale * code + bias,
// so with bias = -scale the codes 0, 1 and 2 are those three values exactly, and
// the trits on disk already use that numbering. Loading is bit shuffling.

import Foundation
import MLX

struct KittenLanguageModelConfig {
    var hiddenSize: Int
    var layers: Int
    var heads: Int
    var kvHeads: Int
    var headDim: Int
    var ropeTheta: Float
    var rmsNormEps: Float
    /// First token id the output head covers: the control tokens and the codec tokens follow it.
    var headStart: Int
}

enum KittenLanguageModelError: Error, LocalizedError {
    case badCheckpoint(String)
    case cancelled

    var errorDescription: String? {
        switch self {
        case .badCheckpoint(let detail): return "Not a packed Kitten TTS 2 checkpoint: \(detail)"
        case .cancelled: return "The request was cancelled."
        }
    }
}

private struct TernaryLinear {
    let codes: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let groupSize: Int

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        quantizedMM(x, codes, scales: scales, biases: biases, transpose: true, groupSize: groupSize, bits: 4)
    }
}

private struct Layer {
    let inputNorm: MLXArray
    let postAttentionNorm: MLXArray
    let qNorm: MLXArray
    let kNorm: MLXArray
    let q, k, v, o, gate, up, down: TernaryLinear
}

/// Keys and values for one layer, grown in blocks: reallocating per token would copy it every step.
private final class KVCache {
    private static let step = 256
    var keys: MLXArray?
    var values: MLXArray?
    var offset = 0

    func updateAndFetch(_ newKeys: MLXArray, _ newValues: MLXArray) -> (MLXArray, MLXArray) {
        let length = newKeys.dim(2)
        if keys == nil || offset + length > keys!.dim(2) {
            let capacity = ((offset + length + Self.step - 1) / Self.step) * Self.step
            let shape = [newKeys.dim(0), newKeys.dim(1), capacity, newKeys.dim(3)]
            let grownKeys = MLXArray.zeros(shape, dtype: newKeys.dtype)
            let grownValues = MLXArray.zeros(shape, dtype: newValues.dtype)
            if let keys, let values, offset > 0 {
                grownKeys[0..., 0..., ..<offset, 0...] = keys[0..., 0..., ..<offset, 0...]
                grownValues[0..., 0..., ..<offset, 0...] = values[0..., 0..., ..<offset, 0...]
            }
            keys = grownKeys
            values = grownValues
        }
        keys![0..., 0..., offset ..< offset + length, 0...] = newKeys
        values![0..., 0..., offset ..< offset + length, 0...] = newValues
        offset += length
        return (keys![0..., 0..., ..<offset, 0...], values![0..., 0..., ..<offset, 0...])
    }
}

final class KittenLanguageModel {
    let config: KittenLanguageModelConfig
    private let embed: MLXArray
    private let head: MLXArray
    private let norm: MLXArray
    private let speakerWeight: MLXArray
    private let speakerBias: MLXArray
    private let speakerNormWeight: MLXArray
    private let speakerNormBias: MLXArray
    private let layers: [Layer]

    /// Reads the packed checkpoint (`lm/model-ternary.safetensors`) and unpacks it for MLX.
    init(url: URL, config: KittenLanguageModelConfig) throws {
        self.config = config
        var (tensors, metadata) = try loadArraysAndMetadata(url: url)
        guard metadata["format"] == "kittenml-ternary-1.58" else {
            throw KittenLanguageModelError.badCheckpoint("format \(metadata["format"] ?? "none")")
        }
        guard let groupSize = Int(metadata["group_size"] ?? ""),
              let packedJSON = metadata["packed"]?.data(using: .utf8),
              let packed = try JSONSerialization.jsonObject(with: packedJSON) as? [String: [String: Any]]
        else { throw KittenLanguageModelError.badCheckpoint("no packing metadata") }

        func dense(_ name: String) throws -> MLXArray {
            guard let tensor = tensors.removeValue(forKey: name) else {
                throw KittenLanguageModelError.badCheckpoint("\(name) is missing")
            }
            return tensor.asType(.bfloat16)
        }
        // Base-3 bytes, five trits each and lowest first, become eight nibbles per uint32.
        let powers = MLXArray([1, 3, 9, 27, 81].map { UInt32($0) })
        let shifts = MLXArray((0 ..< 8).map { UInt32($0 * 4) })
        func linear(_ name: String) throws -> TernaryLinear {
            guard let spec = packed[name], let shape = spec["shape"] as? [Int], shape.count == 2,
                  let bytes = tensors.removeValue(forKey: "\(name)::codes"),
                  let scale = tensors.removeValue(forKey: "\(name)::scale")
            else { throw KittenLanguageModelError.badCheckpoint("\(name) is not packed") }
            let (rows, columns) = (shape[0], shape[1])
            let trits = bytes.asType(.uint32).reshaped([-1, 1]).floorDivide(powers) % 3
            let nibbles = trits.reshaped([-1])[..<(rows * columns)].reshaped([rows, columns / 8, 8])
            let codes = (nibbles << shifts).sum(axis: -1).asType(.uint32)
            let scales = scale.asType(.bfloat16)
            let layer = TernaryLinear(codes: codes, scales: scales, biases: -scales, groupSize: groupSize)
            // One tensor at a time: the unpacked trits are eight times the size of the nibbles.
            eval(layer.codes, layer.scales, layer.biases)
            return layer
        }

        embed = try dense("model.embed_tokens.weight")
        head = embed[config.headStart...]
        norm = try dense("model.norm.weight")
        speakerWeight = try dense("spk_proj.0.weight")
        speakerBias = try dense("spk_proj.0.bias")
        speakerNormWeight = try dense("spk_proj.1.weight")
        speakerNormBias = try dense("spk_proj.1.bias")
        var built: [Layer] = []
        for i in 0 ..< config.layers {
            let p = "model.layers.\(i)"
            built.append(Layer(
                inputNorm: try dense("\(p).input_layernorm.weight"),
                postAttentionNorm: try dense("\(p).post_attention_layernorm.weight"),
                qNorm: try dense("\(p).self_attn.q_norm.weight"),
                kNorm: try dense("\(p).self_attn.k_norm.weight"),
                q: try linear("\(p).self_attn.q_proj.weight"),
                k: try linear("\(p).self_attn.k_proj.weight"),
                v: try linear("\(p).self_attn.v_proj.weight"),
                o: try linear("\(p).self_attn.o_proj.weight"),
                gate: try linear("\(p).mlp.gate_proj.weight"),
                up: try linear("\(p).mlp.up_proj.weight"),
                down: try linear("\(p).mlp.down_proj.weight")))
        }
        layers = built
        eval(embed, head, norm, speakerWeight, speakerBias, speakerNormWeight, speakerNormBias)
        for layer in layers { eval(layer.inputNorm, layer.postAttentionNorm, layer.qNorm, layer.kNorm) }
    }

    /// The speaker head: a 512-dim embedding becomes the hidden state of position 0.
    func speakerHidden(_ embedding: [Float]) -> MLXArray {
        let x = MLXArray(embedding).reshaped([1, embedding.count]).asType(.bfloat16)
        let projected = matmul(x, speakerWeight.T) + speakerBias
        return MLXFast.layerNorm(projected, weight: speakerNormWeight, bias: speakerNormBias, eps: 1e-5)
            .reshaped([1, 1, -1])
    }

    /// Run `prompt` from an empty cache, with position 0 replaced by the speaker. Returns head logits.
    func prefill(_ prompt: [Int], speaker: MLXArray, cache: Cache) throws -> [Float] {
        let ids = MLXArray(prompt.map { Int32($0) })
        var embeddings = take(embed, ids, axis: 0)[.newAxis]
        embeddings = concatenated([speaker.asType(embeddings.dtype), embeddings[0..., 1...]], axis: 1)
        return try forward(embeddings, cache: cache)
    }

    /// One more token. Returns head logits.
    func step(_ token: Int, cache: Cache) throws -> [Float] {
        try forward(take(embed, MLXArray([Int32(token)]), axis: 0)[.newAxis], cache: cache)
    }

    final class Cache {
        fileprivate let layers: [KVCache]
        init(layers: Int) { self.layers = (0 ..< layers).map { _ in KVCache() } }
    }

    func newCache() -> Cache { Cache(layers: config.layers) }

    private func forward(_ embeddings: MLXArray, cache: Cache) throws -> [Float] {
        let length = embeddings.dim(1)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = length > 1 ? .causal : .none
        let scale = 1 / Float(config.headDim).squareRoot()
        var x = embeddings
        for (layer, kv) in zip(layers, cache.layers) {
            let h = MLXFast.rmsNorm(x, weight: layer.inputNorm, eps: config.rmsNormEps)
            var q = MLXFast.rmsNorm(layer.q(h).reshaped([1, length, config.heads, -1]),
                                    weight: layer.qNorm, eps: config.rmsNormEps).transposed(0, 2, 1, 3)
            var k = MLXFast.rmsNorm(layer.k(h).reshaped([1, length, config.kvHeads, -1]),
                                    weight: layer.kNorm, eps: config.rmsNormEps).transposed(0, 2, 1, 3)
            let v = layer.v(h).reshaped([1, length, config.kvHeads, -1]).transposed(0, 2, 1, 3)
            q = MLXFast.RoPE(q, dimensions: config.headDim, traditional: false, base: config.ropeTheta,
                             scale: 1, offset: kv.offset)
            k = MLXFast.RoPE(k, dimensions: config.headDim, traditional: false, base: config.ropeTheta,
                             scale: 1, offset: kv.offset)
            let (keys, values) = kv.updateAndFetch(k, v)
            let attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: keys, values: values, scale: scale, mask: mask)
            x = x + layer.o(attended.transposed(0, 2, 1, 3).reshaped([1, length, -1]))
            let m = MLXFast.rmsNorm(x, weight: layer.postAttentionNorm, eps: config.rmsNormEps)
            let gate = layer.gate(m)
            x = x + layer.down(gate * sigmoid(gate) * layer.up(m))
        }
        let last = MLXFast.rmsNorm(x[0..., (length - 1)...], weight: norm, eps: config.rmsNormEps)
        // The tied output head, cut to the rows the sampler draws from.
        let logits = matmul(last.reshaped([1, -1]), head.T).asType(.float32).reshaped([-1])
        // A failed evaluation goes to MLX's error handler and leaves nothing to read: throw it first.
        try withError { eval(logits) }
        return logits.asArray(Float.self)
    }
}
