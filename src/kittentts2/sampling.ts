/**
 * Kitten TTS 2's decode-time sampling, a port of the Python package's
 * `logits.py` and the transformers warpers it composes, in the same order:
 *
 *     token-run penalty -> repetition penalty -> temperature
 *                       -> top_k -> top_p -> min_p -> sample
 *
 * The scores cover token ids from `headStart` up: the control tokens and the
 * 6561 codec tokens. Text tokens are never drawn after `<speech_start>`.
 */

export interface SamplingSettings {
  temperature: number;
  topK: number;
  topP: number;
  minP: number;
  repetitionPenalty: number;
  /** Penalise only the last N generated tokens; `null` scans the whole sequence. */
  repetitionWindow: number | null;
  tokenRunPenalty: number;
  tokenRunGrace: number;
}

export interface SamplerIds {
  headStart: number;
  audioStart: number;
  audioEnd: number;
  /** Ids the repetition penalty leaves alone: `<speech_end>` and `[STOP]`. */
  exempt: readonly number[];
}

/** A seedable uniform generator (mulberry32). */
export class Random {
  private state: number;

  constructor(seed?: number) {
    this.state = (seed ?? Math.floor(Math.random() * 0x100000000)) >>> 0;
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Standard normal values (Box-Muller). */
  fillNormal(out: Float32Array): Float32Array {
    for (let i = 0; i < out.length; i += 2) {
      const u = 1 - this.next();
      const radius = Math.sqrt(-2 * Math.log(u));
      const angle = 2 * Math.PI * this.next();
      out[i] = radius * Math.cos(angle);
      if (i + 1 < out.length) out[i + 1] = radius * Math.sin(angle);
    }
    return out;
  }
}

/**
 * Draw the next token id. `scores` is modified in place; `history` is the whole
 * sequence so far, prompt included, and `promptLength` where generation began.
 */
export function sampleToken(
  scores: Float32Array,
  history: readonly number[],
  promptLength: number,
  settings: SamplingSettings,
  ids: SamplerIds,
  random: Random,
): number {
  const { headStart } = ids;
  const last = history[history.length - 1];

  // Escalating penalty on a codec token repeated past the grace length.
  if (settings.tokenRunPenalty && last >= ids.audioStart && last < ids.audioEnd) {
    let run = 0;
    for (let i = history.length - 1; i >= 0 && history[i] === last; i -= 1) run += 1;
    if (run >= settings.tokenRunGrace) {
      scores[last - headStart] -= (run - settings.tokenRunGrace + 1) * settings.tokenRunPenalty;
    }
  }

  // Each token seen in the window is penalised once, from its original score.
  if (settings.repetitionPenalty !== 1) {
    const from = settings.repetitionWindow
      ? Math.max(promptLength, history.length - settings.repetitionWindow)
      : 0;
    const original = new Map<number, number>();
    for (let i = from; i < history.length; i += 1) {
      const index = history[i] - headStart;
      if (index < 0 || index >= scores.length || original.has(index)) continue;
      original.set(index, scores[index]);
    }
    for (const [index, value] of original) {
      if (ids.exempt.includes(index + headStart)) continue;
      scores[index] = value < 0 ? value * settings.repetitionPenalty : value / settings.repetitionPenalty;
    }
  }

  if (settings.temperature !== 1) {
    for (let i = 0; i < scores.length; i += 1) scores[i] /= settings.temperature;
  }

  // Candidates in ascending score order, as the warpers see them after sorting.
  let candidates = topCandidates(scores, settings.topK);

  if (settings.topP < 1) {
    const probabilities = softmax(candidates.map(index => scores[index]));
    let cumulative = 0;
    let firstKept = candidates.length - 1;
    for (let i = 0; i < candidates.length; i += 1) {
      cumulative += probabilities[i];
      if (cumulative > 1 - settings.topP) {
        firstKept = i;
        break;
      }
    }
    candidates = candidates.slice(Math.min(firstKept, candidates.length - 1));
  }

  if (settings.minP > 0) {
    const probabilities = softmax(candidates.map(index => scores[index]));
    const threshold = settings.minP * probabilities[probabilities.length - 1];
    const kept = candidates.filter((_, i) => i === candidates.length - 1 || probabilities[i] >= threshold);
    candidates = kept;
  }

  const probabilities = softmax(candidates.map(index => scores[index]));
  let draw = random.next();
  for (let i = candidates.length - 1; i >= 0; i -= 1) {
    draw -= probabilities[i];
    if (draw < 0) return candidates[i] + headStart;
  }
  return candidates[candidates.length - 1] + headStart;
}

/** Indices of the top-k scores (all of them when k is 0), ascending, ties at the cut kept. */
function topCandidates(scores: Float32Array, topK: number): number[] {
  const all = scores.length;
  if (!topK || topK >= all) {
    return Array.from({ length: all }, (_, i) => i).sort((a, b) => scores[a] - scores[b]);
  }
  // Keep a small ascending list of the best k seen; k is 50 for both presets.
  const best: number[] = [];
  for (let i = 0; i < all; i += 1) {
    const value = scores[i];
    if (best.length === topK && value <= scores[best[0]]) continue;
    let at = best.length;
    while (at > 0 && scores[best[at - 1]] > value) at -= 1;
    best.splice(at, 0, i);
    if (best.length > topK) best.shift();
  }
  // transformers keeps every score equal to the k-th one.
  const kth = scores[best[0]];
  const ties: number[] = [];
  for (let i = 0; i < all; i += 1) {
    if (scores[i] === kth && !best.includes(i)) ties.push(i);
  }
  return ties.length ? [...ties, ...best] : best;
}

function softmax(values: number[]): number[] {
  let max = -Infinity;
  for (const value of values) if (value > max) max = value;
  const exps = values.map(value => Math.exp(value - max));
  let sum = 0;
  for (const value of exps) sum += value;
  return exps.map(value => value / sum);
}
