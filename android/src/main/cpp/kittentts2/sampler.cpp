#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

namespace kt2 {

double Random::next() {
    state_ += 0x6D2B79F5u;
    uint32_t t = state_;
    t = (t ^ (t >> 15)) * (t | 1u);
    t ^= t + ((t ^ (t >> 7)) * (t | 61u));
    return static_cast<double>(t ^ (t >> 14)) / 4294967296.0;
}

namespace {

std::vector<float> softmax(const std::vector<float> & scores, const std::vector<int> & indices) {
    float maximum = -INFINITY;
    for (int i : indices) maximum = std::max(maximum, scores[i]);
    std::vector<float> out(indices.size());
    float sum = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        out[i] = std::exp(scores[indices[i]] - maximum);
        sum += out[i];
    }
    for (float & value : out) value /= sum;
    return out;
}

} // namespace

int sampleToken(const std::vector<float> & input, const std::vector<int> & history, int promptLength,
                const SamplingSettings & settings, const SamplerIds & ids, Random & random) {
    std::vector<float> scores = input;
    const int headStart = ids.headStart;
    const int count = static_cast<int>(scores.size());
    const int last = history.back();

    if (settings.tokenRunPenalty != 0 && last >= ids.audioStart && last < ids.audioEnd) {
        int run = 0;
        for (int i = static_cast<int>(history.size()) - 1; i >= 0 && history[i] == last; i--) run++;
        if (run >= settings.tokenRunGrace) {
            scores[last - headStart] -= static_cast<float>(run - settings.tokenRunGrace + 1) * settings.tokenRunPenalty;
        }
    }

    if (settings.repetitionPenalty != 1) {
        const int size = static_cast<int>(history.size());
        const int from = settings.repetitionWindow ? std::max(promptLength, size - *settings.repetitionWindow) : 0;
        std::unordered_set<int> seen;
        std::vector<std::pair<int, float>> original;
        for (int i = from; i < size; i++) {
            const int index = history[i] - headStart;
            if (index < 0 || index >= count || !seen.insert(index).second) continue;
            original.emplace_back(index, scores[index]);
        }
        for (const auto & [index, value] : original) {
            if (std::find(ids.exempt.begin(), ids.exempt.end(), index + headStart) != ids.exempt.end()) continue;
            scores[index] = value < 0 ? value * settings.repetitionPenalty : value / settings.repetitionPenalty;
        }
    }

    if (settings.temperature != 1) {
        for (float & score : scores) score /= settings.temperature;
    }

    // Candidates in ascending score order, as the warpers see them after sorting.
    std::vector<int> candidates(count);
    std::iota(candidates.begin(), candidates.end(), 0);
    if (settings.topK == 0 || settings.topK >= count) {
        std::stable_sort(candidates.begin(), candidates.end(), [&](int a, int b) { return scores[a] < scores[b]; });
    } else {
        std::stable_sort(candidates.begin(), candidates.end(), [&](int a, int b) { return scores[a] > scores[b]; });
        std::vector<int> best(candidates.begin(), candidates.begin() + settings.topK);
        const float kth = scores[best.back()];
        // transformers keeps every score equal to the k-th one.
        std::unordered_set<int> kept(best.begin(), best.end());
        std::vector<int> ties;
        for (int i = 0; i < count; i++) {
            if (scores[i] == kth && !kept.count(i)) ties.push_back(i);
        }
        candidates = ties;
        candidates.insert(candidates.end(), best.rbegin(), best.rend());
    }

    if (settings.topP < 1) {
        const std::vector<float> probabilities = softmax(scores, candidates);
        float cumulative = 0;
        size_t firstKept = candidates.size() - 1;
        for (size_t i = 0; i < candidates.size(); i++) {
            cumulative += probabilities[i];
            if (cumulative > 1 - settings.topP) {
                firstKept = i;
                break;
            }
        }
        candidates.erase(candidates.begin(), candidates.begin() + static_cast<long>(std::min(firstKept, candidates.size() - 1)));
    }

    if (settings.minP > 0) {
        const std::vector<float> probabilities = softmax(scores, candidates);
        const float threshold = settings.minP * probabilities.back();
        std::vector<int> kept;
        for (size_t i = 0; i < candidates.size(); i++) {
            if (i == candidates.size() - 1 || probabilities[i] >= threshold) kept.push_back(candidates[i]);
        }
        candidates = kept;
    }

    const std::vector<float> probabilities = softmax(scores, candidates);
    auto draw = static_cast<float>(random.next());
    for (int i = static_cast<int>(candidates.size()) - 1; i >= 0; i--) {
        draw -= probabilities[i];
        if (draw < 0) return candidates[i] + headStart;
    }
    return candidates.back() + headStart;
}

} // namespace kt2
