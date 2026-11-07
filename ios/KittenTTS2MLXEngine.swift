// Kitten TTS 2 on the GPU behind the React Native module (KittenTTS2MLXModule.m):
// the language model (KittenTTS2LanguageModel.swift) and the audio decoder
// (KittenTTS2AudioDecoder.swift). Work runs one call at a time on its own queue,
// token loop and sampling included, so nothing crosses to JavaScript per token.
// MLX errors are thrown (withError) and reach JavaScript; otherwise MLX ends the app.

import Foundation
import Metal
import MLX

@objc(KittenTTS2MLXEngine)
public final class KittenTTS2MLXEngine: NSObject {
    @objc public static let shared = KittenTTS2MLXEngine()

    private let queue = DispatchQueue(label: "dev.kittenml.kittentts2.mlx", qos: .userInitiated)
    private var model: KittenLanguageModel?
    private var decoder: KittenAudioDecoder?
    private let lock = NSLock()
    private var cancelled = false

    /// MLX needs an Apple GPU family the simulator does not have.
    @objc public static var isAvailable: Bool {
        #if targetEnvironment(simulator)
        return false
        #else
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        return device.supportsFamily(.apple7)
        #endif
    }

    @objc public func load(
        path: String, config: [String: Any], completion: @escaping (NSError?) -> Void
    ) {
        queue.async {
            do {
                // MLX keeps freed buffers for reuse; on a phone, keep that cache small.
                Memory.cacheLimit = 64 * 1024 * 1024
                self.model = nil
                let config = try Self.config(config)
                self.model = try withError {
                    try KittenLanguageModel(url: URL(fileURLWithPath: path), config: config)
                }
                Memory.clearCache()
                completion(nil)
            } catch {
                completion(Self.error(error))
            }
        }
    }

    @objc public func loadDecoder(path: String, completion: @escaping (NSError?) -> Void) {
        queue.async {
            do {
                self.decoder = nil
                self.decoder = try withError { try KittenAudioDecoder(url: URL(fileURLWithPath: path)) }
                completion(nil)
            } catch {
                completion(Self.error(error))
            }
        }
    }

    /// Codec tokens to 24 kHz float32 samples, returned as base64 bytes: a few
    /// megabytes cross the bridge as one string much faster than as numbers.
    @objc public func decode(
        tokens: [NSNumber], promptToken: [NSNumber], promptFeat: String, embedding: [NSNumber],
        seed: UInt32, completion: @escaping (String?, NSError?) -> Void
    ) {
        queue.async {
            do {
                guard let decoder = self.decoder else {
                    throw NSError(domain: "KittenTTS2MLX", code: 1,
                                  userInfo: [NSLocalizedDescriptionKey: "The MLX decoder is not loaded."])
                }
                guard let featData = Data(base64Encoded: promptFeat), featData.count % (4 * 80) == 0 else {
                    throw NSError(domain: "KittenTTS2MLX", code: 3,
                                  userInfo: [NSLocalizedDescriptionKey: "promptFeat is not float32 mel frames."])
                }
                let feat = featData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
                let samples = try withError {
                    try decoder.decode(
                        tokens: tokens.map(\.intValue), promptToken: promptToken.map(\.intValue),
                        promptFeat: MLXArray(feat, [feat.count / 80, 80]),
                        embedding: embedding.map(\.floatValue), seed: UInt64(seed)
                    ).asArray(Float.self)
                }
                Memory.clearCache()
                let bytes = samples.withUnsafeBufferPointer { Data(buffer: $0) }
                completion(bytes.base64EncodedString(), nil)
            } catch {
                completion(nil, Self.error(error))
            }
        }
    }

    /// Returns the token ids written after the prompt, up to `<speech_end>` or `[STOP]`.
    @objc public func generate(
        prompt: [NSNumber], speaker: [NSNumber], settings: [String: Any], maxNewTokens: Int,
        seed: UInt32, completion: @escaping ([NSNumber]?, NSError?) -> Void
    ) {
        setCancelled(false)
        queue.async {
            do {
                guard let model = self.model else {
                    throw NSError(domain: "KittenTTS2MLX", code: 1,
                                  userInfo: [NSLocalizedDescriptionKey: "The MLX model is not loaded."])
                }
                let tokens = try withError {
                    try self.run(
                        model, prompt: prompt.map(\.intValue), speaker: speaker.map(\.floatValue),
                        settings: settings, maxNewTokens: maxNewTokens, seed: seed)
                }
                Memory.clearCache()
                completion(tokens.map { NSNumber(value: $0) }, nil)
            } catch {
                completion(nil, Self.error(error))
            }
        }
    }

    /// Stops the generation in flight between tokens.
    @objc public func cancel() {
        setCancelled(true)
    }

    @objc public func unload(completion: @escaping () -> Void) {
        setCancelled(true)
        queue.async {
            self.model = nil
            self.decoder = nil
            Memory.clearCache()
            completion()
        }
    }

    private func run(
        _ model: KittenLanguageModel, prompt: [Int], speaker: [Float], settings: [String: Any],
        maxNewTokens: Int, seed: UInt32
    ) throws -> [Int] {
        let number = { (key: String) -> Double? in (settings[key] as? NSNumber)?.doubleValue }
        let sampling = KittenSamplingSettings(
            temperature: Float(number("temperature") ?? 1), topK: Int(number("topK") ?? 0),
            topP: Float(number("topP") ?? 1), minP: Float(number("minP") ?? 0),
            repetitionPenalty: Float(number("repetitionPenalty") ?? 1),
            repetitionWindow: number("repetitionWindow").map { Int($0) },
            tokenRunPenalty: Float(number("tokenRunPenalty") ?? 0),
            tokenRunGrace: Int(number("tokenRunGrace") ?? 10))
        let speechEnd = Int(number("speechEnd") ?? 0)
        let stop = Int(number("stop") ?? 0)
        let ids = KittenSamplerIds(
            headStart: model.config.headStart, audioStart: Int(number("audioStart") ?? 0),
            audioEnd: Int(number("audioEnd") ?? 0), exempt: [speechEnd, stop])
        var random = KittenRandom(seed: seed)

        let cache = model.newCache()
        var logits = model.prefill(prompt, speaker: model.speakerHidden(speaker), cache: cache)
        var history = prompt
        var generated: [Int] = []
        while generated.count < maxNewTokens {
            let token = kittenSampleToken(
                logits, history: history, promptLength: prompt.count, settings: sampling, ids: ids,
                random: &random)
            if token == speechEnd || token == stop { break }
            history.append(token)
            generated.append(token)
            if generated.count == maxNewTokens { break }
            if isCancelled() { throw KittenLanguageModelError.cancelled }
            logits = model.step(token, cache: cache)
        }
        return generated
    }

    private static func config(_ values: [String: Any]) throws -> KittenLanguageModelConfig {
        func int(_ key: String) throws -> Int {
            guard let value = values[key] as? NSNumber else {
                throw KittenLanguageModelError.badCheckpoint("config has no \(key)")
            }
            return value.intValue
        }
        return KittenLanguageModelConfig(
            hiddenSize: try int("hidden_size"), layers: try int("num_hidden_layers"),
            heads: try int("num_attention_heads"), kvHeads: try int("num_key_value_heads"),
            headDim: try int("head_dim"),
            ropeTheta: (values["rope_theta"] as? NSNumber)?.floatValue ?? 1_000_000,
            rmsNormEps: (values["rms_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6,
            headStart: try int("head_start"))
    }

    private static func error(_ error: Error) -> NSError {
        let code = (error as? KittenLanguageModelError).map { if case .cancelled = $0 { return 2 } else { return 3 } } ?? 3
        return NSError(domain: "KittenTTS2MLX", code: code,
                       userInfo: [NSLocalizedDescriptionKey: error.localizedDescription])
    }

    private func setCancelled(_ value: Bool) {
        lock.lock()
        cancelled = value
        lock.unlock()
    }

    private func isCancelled() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }
}
