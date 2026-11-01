// Kitten TTS 2's audio decoder on the GPU, through MLX: chatterbox-turbo's s3gen,
// ported from the Python package's vendored s3gen.py. Codec tokens become a mel
// spectrogram (the flow: a conformer encoder, then two meanflow steps of a UNet
// estimator), and the mel becomes 24 kHz audio (the HiFT vocoder).
//
// Weights are read by their torch names from the file
// tools/kitten-tts-2-onnx/decoder_mlx.py writes, convolutions already folded and
// laid out for MLX. They stay float16 in memory and are widened to float32 where
// they are used; activations are float32 and channels-last, [1, time, channels].
// The graph is evaluated layer by layer, so a long chunk's intermediates are
// freed as it goes instead of all living until the end: on a phone, memory is
// what limits the chunk length.

import Foundation
import MLX

enum KittenAudioDecoderError: Error, LocalizedError {
    case missingWeight(String)

    var errorDescription: String? {
        switch self {
        case .missingWeight(let name): return "The decoder weights have no \(name)."
        }
    }
}

final class KittenAudioDecoder {
    let weights: [String: MLXArray]
    private let stftBasis: MLXArray
    private let istftBasis: MLXArray
    private let overlapKernel: MLXArray
    private let windowSquared: MLXArray

    private static let mels = 80
    private static let fft = 16
    private static let hop = 4
    private static let samplesPerFrame = 480
    private static let sampleRate: Float = 24_000

    init(url: URL) throws {
        weights = try loadArrays(url: url).mapValues { $0.asType(.float16) }
        for name in ["flow.input_embedding.weight", "mel2wav.conv_post.weight"] where weights[name] == nil {
            throw KittenAudioDecoderError.missingWeight(name)
        }
        eval(Array(weights.values))
        Memory.clearCache()

        // A 16-point STFT and its inverse, with the periodic Hann window torch uses.
        let n = Self.fft
        let bins = n / 2 + 1
        let window = (0 ..< n).map { 0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(n)) }
        var forward = [Float](repeating: 0, count: 2 * bins * n)        // [2 * bins, n]
        var inverse = [Float](repeating: 0, count: n * 2 * bins)        // [n, 2 * bins]
        for k in 0 ..< bins {
            let scale = (k == 0 || k == bins - 1) ? 1.0 : 2.0
            for t in 0 ..< n {
                let angle = 2 * Double.pi * Double(k * t) / Double(n)
                forward[k * n + t] = Float(cos(angle) * window[t])
                forward[(bins + k) * n + t] = Float(-sin(angle) * window[t])
                inverse[t * 2 * bins + k] = Float(cos(angle) * scale / Double(n) * window[t])
                let sinTerm = (k == 0 || k == bins - 1) ? 0 : -sin(angle) * scale / Double(n)
                inverse[t * 2 * bins + bins + k] = Float(sinTerm * window[t])
            }
        }
        stftBasis = MLXArray(forward, [2 * bins, n, 1])                  // conv weight [out, k, in]
        istftBasis = MLXArray(inverse, [n, 2 * bins])
        var identity = [Float](repeating: 0, count: n * n)
        for i in 0 ..< n { identity[i * n + i] = 1 }
        overlapKernel = MLXArray(identity, [1, n, n])                    // [out 1, k n, in n]
        windowSquared = MLXArray(window.map { Float($0 * $0) }, [1, n, 1])
    }

    /// Codec tokens (lookahead padding included) to audio samples. The noise
    /// arguments exist for parity checks; normally they are drawn from `seed`.
    func decode(
        tokens: [Int], promptToken: [Int], promptFeat: MLXArray, embedding: [Float], seed: UInt64,
        flowNoise: MLXArray? = nil, phase: MLXArray? = nil, sourceNoise: MLXArray? = nil
    ) throws -> MLXArray {
        let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 3)
        let mel = try flow(tokens: tokens, promptToken: promptToken, promptFeat: promptFeat,
                           embedding: embedding, noise: flowNoise, key: keys[0])
        return try vocoder(mel, phase: phase, noise: sourceNoise, phaseKey: keys[1], noiseKey: keys[2])
    }

    // MARK: - Layers

    func w(_ name: String) throws -> MLXArray {
        guard let value = weights[name] else { throw KittenAudioDecoderError.missingWeight(name) }
        return value.asType(.float32)
    }

    func linear(_ x: MLXArray, _ name: String, bias: Bool = true) throws -> MLXArray {
        let y = matmul(x, try w("\(name).weight").T)
        return bias ? y + (try w("\(name).bias")) : y
    }

    func layerNorm(_ x: MLXArray, _ name: String, eps: Float) throws -> MLXArray {
        MLXFast.layerNorm(x, weight: try w("\(name).weight"), bias: try w("\(name).bias"), eps: eps)
    }

    func conv(
        _ x: MLXArray, _ name: String, stride: Int = 1, padding: Int = 0, dilation: Int = 1
    ) throws -> MLXArray {
        conv1d(x, try w("\(name).weight"), stride: stride, padding: padding, dilation: dilation)
            + (try w("\(name).bias"))
    }

    /// Zero padding on the time axis.
    func padTime(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        padded(x, widths: [IntOrPair(0), IntOrPair((left, right)), IntOrPair(0)])
    }

    /// Nearest-neighbour upsampling on the time axis.
    func repeatTime(_ x: MLXArray, _ factor: Int) -> MLXArray {
        let shape = x.shape
        return broadcast(x.expandedDimensions(axis: 2), to: [shape[0], shape[1], factor, shape[2]])
            .reshaped([shape[0], shape[1] * factor, shape[2]])
    }

    func mish(_ x: MLXArray) -> MLXArray { x * tanh(logAddExp(x, MLXArray(Float(0)))) }
    func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }
    func gelu(_ x: MLXArray) -> MLXArray { 0.5 * x * (1 + erf(x / Float(2).squareRoot())) }
    func elu(_ x: MLXArray) -> MLXArray { which(x .> 0, x, exp(x) - 1) }
    func leakyRelu(_ x: MLXArray, _ slope: Float) -> MLXArray { which(x .> 0, x, x * slope) }

    // MARK: - Flow

    /// Codec tokens to the target's mel frames, [1, 2T, 80].
    func flow(
        tokens: [Int], promptToken: [Int], promptFeat: MLXArray, embedding: [Float], noise: MLXArray?,
        key: MLXArray
    ) throws -> MLXArray {
        let embeddingArray = MLXArray(embedding).reshaped([1, embedding.count])
        let normalized = embeddingArray / maximum(sqrt((embeddingArray * embeddingArray).sum(axis: -1, keepDims: true)), 1e-12)
        let spks = try linear(normalized, "flow.spk_embed_affine_layer")                 // [1, 80]

        let ids = MLXArray((promptToken + tokens).map { Int32($0) })
        var x = take(try w("flow.input_embedding.weight"), ids, axis: 0)[.newAxis]          // [1, N, 512]
        let mu = try linear(try encoder(x), "flow.encoder_proj")                           // [1, 2N, 80]
        let frames = mu.dim(1)
        let promptFrames = promptFeat.dim(0)
        let cond = concatenated([promptFeat[.newAxis],
                                 MLXArray.zeros([1, frames - promptFrames, Self.mels])], axis: 1)
        let spksTime = broadcast(spks[0..., .newAxis, 0...], to: [1, frames, Self.mels])

        x = noise.map { $0[.newAxis] } ?? MLXRandom.normal([1, frames, Self.mels], key: key)
        for (t, r) in [(Float(0), Float(0.5)), (Float(0.5), Float(1))] {
            let velocity = try estimator(x, mu: mu, spks: spksTime, cond: cond, t: t, r: r)
            x = x + (r - t) * velocity
        }
        return x[0..., promptFrames...]                                                   // [1, 2T, 80]
    }

    func positionalEncoding(_ length: Int, dims: Int = 512) -> MLXArray {
        // Relative positions length - 1 down to -(length - 1), sines and cosines interleaved.
        let positions = MLXArray((0 ..< (2 * length - 1)).map { Float(length - 1 - $0) })
        let div = MLXArray((0 ..< dims / 2).map { Float(exp(Double(2 * $0) * -(log(10_000.0) / Double(dims)))) })
        let angles = positions[0..., .newAxis] * div[.newAxis]
        return stacked([sin(angles), cos(angles)], axis: -1).reshaped([1, 2 * length - 1, dims])
    }

    func encoder(_ input: MLXArray) throws -> MLXArray {
        let scale = Float(512).squareRoot()
        var x = try layerNorm(try linear(input, "flow.encoder.embed.out.0"), "flow.encoder.embed.out.1", eps: 1e-5) * scale
        var pos = positionalEncoding(x.dim(1))
        // Look ahead three tokens, then a causal convolution, as a residual.
        var h = leakyRelu(try conv(padTime(x, left: 0, right: 3), "flow.encoder.pre_lookahead_layer.conv1"), 0.01)
        h = try conv(padTime(h, left: 2, right: 0), "flow.encoder.pre_lookahead_layer.conv2")
        x = x + h
        for i in 0 ..< 6 {
            x = try conformerLayer(x, pos: pos, "flow.encoder.encoders.\(i)")
            eval(x)
        }
        x = try conv(padTime(repeatTime(x, 2), left: 4, right: 0), "flow.encoder.up_layer.conv")
        x = try layerNorm(try linear(x, "flow.encoder.up_embed.out.0"), "flow.encoder.up_embed.out.1", eps: 1e-5) * scale
        pos = positionalEncoding(x.dim(1))
        for i in 0 ..< 4 {
            x = try conformerLayer(x, pos: pos, "flow.encoder.up_encoders.\(i)")
            eval(x)
        }
        return try layerNorm(x, "flow.encoder.after_norm", eps: 1e-5)
    }

    func conformerLayer(_ input: MLXArray, pos: MLXArray, _ p: String) throws -> MLXArray {
        let heads = 8, dk = 64
        let length = input.dim(1)
        var x = input
        let h = try layerNorm(x, "\(p).norm_mha", eps: 1e-12)
        let q = try linear(h, "\(p).self_attn.linear_q").reshaped([1, length, heads, dk])
        let k = try linear(h, "\(p).self_attn.linear_k").reshaped([1, length, heads, dk]).transposed(0, 2, 1, 3)
        let v = try linear(h, "\(p).self_attn.linear_v").reshaped([1, length, heads, dk]).transposed(0, 2, 1, 3)
        let pe = try linear(pos, "\(p).self_attn.linear_pos", bias: false)
            .reshaped([1, -1, heads, dk]).transposed(0, 2, 1, 3)                            // [1, H, 2L-1, D]
        let qu = (q + (try w("\(p).self_attn.pos_bias_u"))).transposed(0, 2, 1, 3)
        let qv = (q + (try w("\(p).self_attn.pos_bias_v"))).transposed(0, 2, 1, 3)
        // One head at a time: the position scores are [L, 2L-1] per head, which
        // for every head at once would be hundreds of megabytes on a long chunk.
        var outputs: [MLXArray] = []
        for head in 0 ..< heads {
            let ac = matmul(qu[0..., head], k[0..., head].transposed(0, 2, 1))             // [1, L, L]
            var bd = matmul(qv[0..., head], pe[0..., head].transposed(0, 2, 1))            // [1, L, 2L-1]
            // Transformer-XL's relative shift.
            bd = concatenated([MLXArray.zeros([1, length, 1]), bd], axis: -1)
                .reshaped([1, 2 * length, length])[0..., 1...]
                .reshaped([1, length, 2 * length - 1])[0..., 0..., ..<length]
            let attention = softmax((ac + bd) / Float(dk).squareRoot(), axis: -1)
            let out = matmul(attention, v[0..., head])                                     // [1, L, D]
            eval(out)
            outputs.append(out)
        }
        let attended = stacked(outputs, axis: 2).reshaped([1, length, heads * dk])
        x = x + (try linear(attended, "\(p).self_attn.linear_out"))
        let f = try layerNorm(x, "\(p).norm_ff", eps: 1e-12)
        return x + (try linear(silu(try linear(f, "\(p).feed_forward.w_1")), "\(p).feed_forward.w_2"))
    }

    // MARK: - Estimator

    func timeEmbedding(_ t: Float) throws -> MLXArray {
        let half = 160
        let factor = log(10_000.0) / Double(half - 1)
        let freqs = MLXArray((0 ..< half).map { Float(exp(Double($0) * -factor)) })
        let angles = 1000 * t * freqs
        let embedding = concatenated([sin(angles), cos(angles)], axis: -1)[.newAxis]       // [1, 320]
        let e = "flow.decoder.estimator.time_mlp"
        return try linear(silu(try linear(embedding, "\(e).linear_1")), "\(e).linear_2")   // [1, 1024]
    }

    func causalBlock(_ x: MLXArray, _ p: String) throws -> MLXArray {
        let h = try conv(padTime(x, left: 2, right: 0), "\(p).block.0")
        return mish(try layerNorm(h, "\(p).block.2", eps: 1e-5))
    }

    func resnet(_ x: MLXArray, _ p: String, time: MLXArray) throws -> MLXArray {
        var h = try causalBlock(x, "\(p).block1")
        h = h + (try linear(mish(time), "\(p).mlp.1"))[0..., .newAxis, 0...]
        h = try causalBlock(h, "\(p).block2")
        return h + (try conv(x, "\(p).res_conv"))
    }

    func transformer(_ input: MLXArray, _ p: String) throws -> MLXArray {
        let heads = 8, dk = 64
        let length = input.dim(1)
        var x = input
        let h = try layerNorm(x, "\(p).norm1", eps: 1e-5)
        let q = try linear(h, "\(p).attn1.to_q", bias: false).reshaped([1, length, heads, dk]).transposed(0, 2, 1, 3)
        let k = try linear(h, "\(p).attn1.to_k", bias: false).reshaped([1, length, heads, dk]).transposed(0, 2, 1, 3)
        let v = try linear(h, "\(p).attn1.to_v", bias: false).reshaped([1, length, heads, dk]).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(dk).squareRoot(), mask: .none)
        x = x + (try linear(attended.transposed(0, 2, 1, 3).reshaped([1, length, heads * dk]), "\(p).attn1.to_out.0"))
        let f = try layerNorm(x, "\(p).norm3", eps: 1e-5)
        return x + (try linear(gelu(try linear(f, "\(p).ff.net.0.proj")), "\(p).ff.net.2"))
    }

    func estimator(
        _ input: MLXArray, mu: MLXArray, spks: MLXArray, cond: MLXArray, t: Float, r: Float
    ) throws -> MLXArray {
        let e = "flow.decoder.estimator"
        let time = try linear(concatenated([try timeEmbedding(t), try timeEmbedding(r)], axis: -1),
                              "\(e).time_embed_mixer", bias: false)
        var x = concatenated([input, mu, spks, cond], axis: -1)                             // [1, M, 320]

        x = try resnet(x, "\(e).down_blocks.0.0", time: time)
        for j in 0 ..< 4 { x = try transformer(x, "\(e).down_blocks.0.1.\(j)") }
        eval(x)
        let skip = x
        x = try conv(padTime(x, left: 2, right: 0), "\(e).down_blocks.0.2")

        for i in 0 ..< 12 {
            x = try resnet(x, "\(e).mid_blocks.\(i).0", time: time)
            for j in 0 ..< 4 { x = try transformer(x, "\(e).mid_blocks.\(i).1.\(j)") }
            eval(x)
        }

        x = concatenated([x[0..., ..<skip.dim(1)], skip], axis: -1)
        x = try resnet(x, "\(e).up_blocks.0.0", time: time)
        for j in 0 ..< 4 { x = try transformer(x, "\(e).up_blocks.0.1.\(j)") }
        x = try conv(padTime(x, left: 2, right: 0), "\(e).up_blocks.0.2")

        x = try causalBlock(x, "\(e).final_block")
        return try conv(x, "\(e).final_proj")
    }

    // MARK: - Vocoder

    func snake(_ x: MLXArray, _ name: String) throws -> MLXArray {
        let alpha = try w(name)
        let s = sin(x * alpha)
        return x + (1 / (alpha + 1e-9)) * s * s
    }

    func resBlock(_ input: MLXArray, _ p: String, kernel: Int) throws -> MLXArray {
        var x = input
        for (index, dilation) in [1, 3, 5].enumerated() {
            var t = try snake(x, "\(p).activations1.\(index).alpha")
            t = try conv(t, "\(p).convs1.\(index)", padding: (kernel * dilation - dilation) / 2, dilation: dilation)
            t = try snake(t, "\(p).activations2.\(index).alpha")
            t = try conv(t, "\(p).convs2.\(index)", padding: (kernel - 1) / 2)
            x = x + t
        }
        return x
    }

    /// Mel frames, [1, F, 80], to F * 480 samples.
    func vocoder(
        _ mel: MLXArray, phase: MLXArray?, noise: MLXArray?, phaseKey: MLXArray, noiseKey: MLXArray
    ) throws -> MLXArray {
        let v = "mel2wav"
        // F0 from the mel, held for each output sample.
        var f = mel
        for i in [0, 2, 4, 6, 8] { f = elu(try conv(f, "\(v).f0_predictor.condnet.\(i)", padding: 1)) }
        let f0 = repeatTime(abs(try linear(f, "\(v).f0_predictor.classifier")), Self.samplesPerFrame)   // [1, L, 1]
        let length = f0.dim(1)

        // The harmonic source: nine sines at f0's multiples, noise where unvoiced.
        let harmonics = MLXArray((1 ... 9).map { Float($0) })
        let increments = f0 * harmonics / Self.sampleRate                                  // [1, L, 9]
        let theta = 2 * Float.pi * remainder(cumsum(increments, axis: 1), 1)
        var phases = phase ?? MLXRandom.uniform(Float(-Float.pi) ..< Float(Float.pi), [9], key: phaseKey)
        phases = which(MLXArray(0 ..< 9) .== 0, MLXArray(Float(0)), phases)
        let voiced = (f0 .> 10).asType(.float32)
        let amplitude = voiced * 0.003 + (1 - voiced) * (0.1 / 3)
        let gaussian = noise.map { $0[.newAxis] } ?? MLXRandom.normal([1, length, 9], key: noiseKey)
        let sines = 0.1 * sin(theta + phases) * voiced + amplitude * gaussian
        let source = tanh(try linear(sines, "\(v).m_source.l_linear"))                    // [1, L, 1]

        // The source's STFT: reflect-pad by half a window, then a strided convolution.
        let half = Self.fft / 2
        let left = source[0..., 1 ... half][0..., .stride(by: -1)]
        let right = source[0..., (length - half - 1) ..< (length - 1)][0..., .stride(by: -1)]
        let sourceSpectrum = conv1d(concatenated([left, source, right], axis: 1), stftBasis, stride: Self.hop)
        eval(sourceSpectrum)

        var x = try conv(mel, "\(v).conv_pre", padding: 3)
        let rates = [8, 5, 3], kernels = [16, 11, 7]
        let sourceStrides = [15, 3, 1], sourcePadding = [7, 1, 0]
        let sourceResKernels = [7, 7, 11], resKernels = [3, 7, 11]
        for i in 0 ..< 3 {
            x = leakyRelu(x, 0.1)
            x = convTransposed1d(x, try w("\(v).ups.\(i).weight"), stride: rates[i],
                                 padding: (kernels[i] - rates[i]) / 2) + (try w("\(v).ups.\(i).bias"))
            if i == 2 { x = concatenated([x[0..., 1 ..< 2], x], axis: 1) }               // reflection pad (1, 0)
            var s = try conv(sourceSpectrum, "\(v).source_downs.\(i)", stride: sourceStrides[i], padding: sourcePadding[i])
            s = try resBlock(s, "\(v).source_resblocks.\(i)", kernel: sourceResKernels[i])
            x = x + s
            var sum: MLXArray?
            for j in 0 ..< 3 {
                let y = try resBlock(x, "\(v).resblocks.\(i * 3 + j)", kernel: resKernels[j])
                sum = sum.map { $0 + y } ?? y
            }
            x = sum! / 3
            eval(x)
        }
        x = try conv(leakyRelu(x, 0.01), "\(v).conv_post", padding: 3)                     // [1, F, 18]

        // Inverse STFT: one frame per row through the inverse basis, then overlap-add.
        let bins = Self.fft / 2 + 1
        let magnitude = minimum(exp(x[0..., 0..., ..<bins]), 100)
        let angle = sin(x[0..., 0..., bins...])
        let spectrum = concatenated([magnitude * cos(angle), magnitude * sin(angle)], axis: -1)   // [1, F, 18]
        let frames = matmul(spectrum, istftBasis.T)                                        // [1, F, 16]
        let signal = convTransposed1d(frames, overlapKernel, stride: Self.hop)              // [1, (F-1)*4+16, 1]
        let envelope = convTransposed1d(MLXArray.ones([1, frames.dim(1), 1]), windowSquared, stride: Self.hop)
        let audio = (signal / maximum(envelope, 1e-11))[0, half ..< (signal.dim(1) - half), 0]
        return clip(audio, min: -0.99, max: 0.99)
    }
}
