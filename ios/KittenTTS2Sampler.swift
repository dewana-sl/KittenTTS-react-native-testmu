// Kitten TTS 2's decode-time sampling: the same steps, in the same order, as
// src/kittentts2/sampling.ts, which ports the Python package's logits.py:
//
//     token-run penalty -> repetition penalty -> temperature
//                       -> top_k -> top_p -> min_p -> sample
//
// The scores cover token ids from `headStart` up.

import Foundation

struct KittenSamplingSettings {
    var temperature: Float
    var topK: Int
    var topP: Float
    var minP: Float
    var repetitionPenalty: Float
    /// Penalise only the last N generated tokens; nil scans the whole sequence.
    var repetitionWindow: Int?
    var tokenRunPenalty: Float
    var tokenRunGrace: Int
}

struct KittenSamplerIds {
    var headStart: Int
    var audioStart: Int
    var audioEnd: Int
    var exempt: [Int]
}

/// mulberry32, the generator the JavaScript sampler uses.
struct KittenRandom {
    private var state: UInt32

    init(seed: UInt32) { state = seed }

    mutating func next() -> Double {
        state = state &+ 0x6D2B79F5
        var t = state
        t = (t ^ (t >> 15)) &* (t | 1)
        t ^= t &+ ((t ^ (t >> 7)) &* (t | 61))
        return Double(t ^ (t >> 14)) / 4294967296
    }
}

func kittenSampleToken(
    _ input: [Float], history: [Int], promptLength: Int,
    settings: KittenSamplingSettings, ids: KittenSamplerIds, random: inout KittenRandom
) -> Int {
    var scores = input
    let headStart = ids.headStart
    let last = history[history.count - 1]

    if settings.tokenRunPenalty != 0, last >= ids.audioStart, last < ids.audioEnd {
        var run = 0
        var i = history.count - 1
        while i >= 0, history[i] == last { run += 1; i -= 1 }
        if run >= settings.tokenRunGrace {
            scores[last - headStart] -= Float(run - settings.tokenRunGrace + 1) * settings.tokenRunPenalty
        }
    }

    if settings.repetitionPenalty != 1 {
        let from = settings.repetitionWindow.map { max(promptLength, history.count - $0) } ?? 0
        var seen = Set<Int>()
        var original: [(Int, Float)] = []
        if from < history.count {
            for i in from ..< history.count {
                let index = history[i] - headStart
                if index < 0 || index >= scores.count || seen.contains(index) { continue }
                seen.insert(index)
                original.append((index, scores[index]))
            }
        }
        for (index, value) in original where !ids.exempt.contains(index + headStart) {
            scores[index] = value < 0 ? value * settings.repetitionPenalty : value / settings.repetitionPenalty
        }
    }

    if settings.temperature != 1 {
        for i in scores.indices { scores[i] /= settings.temperature }
    }

    // Candidates in ascending score order, as the warpers see them after sorting.
    var candidates: [Int]
    if settings.topK == 0 || settings.topK >= scores.count {
        candidates = scores.indices.sorted { scores[$0] < scores[$1] }
    } else {
        let best = scores.indices.sorted { scores[$0] > scores[$1] }.prefix(settings.topK)
        let kth = scores[best.last!]
        // transformers keeps every score equal to the k-th one.
        let ties = scores.indices.filter { scores[$0] == kth && !best.contains($0) }
        candidates = ties + best.reversed()
    }

    if settings.topP < 1 {
        let probabilities = softmax(candidates.map { scores[$0] })
        var cumulative: Float = 0
        var firstKept = candidates.count - 1
        for i in candidates.indices {
            cumulative += probabilities[i]
            if cumulative > 1 - settings.topP { firstKept = i; break }
        }
        candidates = Array(candidates[min(firstKept, candidates.count - 1)...])
    }

    if settings.minP > 0 {
        let probabilities = softmax(candidates.map { scores[$0] })
        let threshold = settings.minP * probabilities[probabilities.count - 1]
        candidates = candidates.indices.filter { $0 == candidates.count - 1 || probabilities[$0] >= threshold }
            .map { candidates[$0] }
    }

    let probabilities = softmax(candidates.map { scores[$0] })
    var draw = Float(random.next())
    for i in stride(from: candidates.count - 1, through: 0, by: -1) {
        draw -= probabilities[i]
        if draw < 0 { return candidates[i] + headStart }
    }
    return candidates[candidates.count - 1] + headStart
}

private func softmax(_ values: [Float]) -> [Float] {
    let maximum = values.max() ?? 0
    let exps = values.map { Foundation.exp($0 - maximum) }
    let sum = exps.reduce(0, +)
    return exps.map { $0 / sum }
}
