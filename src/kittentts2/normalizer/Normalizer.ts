// A port of kitten-text-processing's English normalizer (normalizer.py,
// _ordering.py, _punctuation.py and _detokenize.py), which adapt NeMo 1.2.0
// (Apache-2.0, Copyright (c) 2021 NVIDIA CORPORATION) and sacremoses' detokenizer
// (MIT). Kitten TTS 2 normalizes every text through the English grammars.

import { Graph, NoPathError } from './Graph';
import { PRESERVE_ORDER_KEY, type TokenGroup, TokenParser, TokenValueError } from './TokenParser';
import { ALPHA_RANGES, CURRENCY_SYMBOLS } from './unicodeData';

const MAX_PERMUTATIONS_PER_SPLIT = 729;
const PUNCTUATION = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~';

export interface NormalizerGrammars {
  tagger: Uint8Array;
  verbalizer: Uint8Array;
  post?: Uint8Array;
}

export class Normalizer {
  private readonly tagger: Graph;
  private readonly verbalizer: Graph;
  private readonly post: Graph | null;

  /** Grammar files after gunzip. */
  constructor(grammars: NormalizerGrammars) {
    this.tagger = new Graph(grammars.tagger);
    this.verbalizer = new Graph(grammars.verbalizer);
    this.post = grammars.post ? new Graph(grammars.post, true) : null;
  }

  /** `kitten_text_processing.normalize_text(text)`: written to spoken English. */
  normalizeText(text: string): string {
    if (!(text && text.trim())) return text;
    return this.normalize(text).trim();
  }

  private normalize(input: string): string {
    const text = input.trim();
    if (!text) return text;
    const tokens = new TokenParser(this.tagger.rewrite(text)).parse();
    const groups = splitTokens(tokens);
    const chunks: string[] = [];
    try {
      for (const group of groups) {
        let found = false;
        for (const candidate of permutations(group)) {
          try {
            chunks.push(this.verbalizer.rewrite(candidate));
            found = true;
            break;
          } catch (error) {
            if (!(error instanceof NoPathError)) throw error;
          }
        }
        if (!found) return escape(text);
      }
    } catch (error) {
      // NeMo returns its escaped input on malformed verbalizer tokens.
      if (error instanceof NoPathError || error instanceof TokenValueError) return escape(text);
      throw error;
    }
    let output = chunks.join(' ').replace(/ {2,}/g, ' ');
    if (this.post && output.trim()) output = this.post.rewrite(output.trim());
    return postProcessPunctuation(input, detokenize(output));
  }
}

function escape(text: string): string {
  return text.replace(/\\/g, '\\\\').replace(/\[/g, '\\[').replace(/\]/g, '\\]');
}

// ── token ordering ────────────────────────────────────────────────────────────

function estimatePermutations(group: TokenGroup): number {
  let count = 1;
  for (const value of group.values()) {
    if (value instanceof Map) count *= estimatePermutations(value);
  }
  return count * factorial(group.size);
}

function factorial(n: number): number {
  let result = 1;
  for (let i = 2; i <= n; i += 1) result *= i;
  return result;
}

function splitTokens(tokens: TokenGroup[]): TokenGroup[][] {
  const splits: TokenGroup[][] = [];
  let start = 0;
  let current = 1;
  tokens.forEach((group, i) => {
    const n = estimatePermutations(group);
    if (n * current > MAX_PERMUTATIONS_PER_SPLIT) {
      splits.push(tokens.slice(start, i));
      start = i;
      current = 1;
    }
    if (n > MAX_PERMUTATIONS_PER_SPLIT) {
      throw new TokenValueError('A token group has too many orderings to try.');
    }
    current *= n;
  });
  splits.push(tokens.slice(start));
  return splits;
}

/** Serializations of a group in every key order (itertools order), recursively. */
function permute(group: TokenGroup): string[] {
  const entries = [...group.entries()];
  const orders = group.has(PRESERVE_ORDER_KEY) ? [entries] : orderings(entries);
  const out: string[] = [];
  for (const order of orders) {
    let partial = [''];
    for (const [key, value] of order) {
      if (typeof value === 'string') {
        partial = partial.map(prefix => `${prefix}${key}: "${value}" `);
      } else if (value instanceof Map) {
        const inner = permute(value);
        partial = partial.flatMap(prefix => inner.map(rest => `${prefix} ${key} { ${rest} } `));
      } else if (value === true) {
        partial = partial.map(prefix => `${prefix}${key}: true `);
      } else {
        throw new TokenValueError(`Key: ${key} Value: None`);
      }
    }
    out.push(...partial);
  }
  return out;
}

function orderings<T>(items: T[]): T[][] {
  if (items.length <= 1) return [items.slice()];
  const out: T[][] = [];
  items.forEach((item, i) => {
    for (const rest of orderings([...items.slice(0, i), ...items.slice(i + 1)])) out.push([item, ...rest]);
  });
  return out;
}

function* permutations(tokens: TokenGroup[]): Generator<string> {
  const options = tokens.map(permute);
  if (options.some(list => list.length === 0)) return;
  const index = new Array<number>(options.length).fill(0);
  for (;;) {
    yield options.map((list, i) => list[index[i]]).join('');
    let position = options.length - 1;
    while (position >= 0) {
      index[position] += 1;
      if (index[position] < options[position].length) break;
      index[position] = 0;
      position -= 1;
    }
    if (position < 0) return;
  }
}

// ── punctuation ───────────────────────────────────────────────────────────────

/** NeMo's post_process_punct: match spaces around punctuation to the input. */
function postProcessPunctuation(originalInput: string, normalizedText: string): string {
  let inputText = originalInput;
  if (inputText.includes('``') && !normalizedText.includes('``')) inputText = inputText.split('``').join('"');
  const input = Array.from(inputText);
  const output = Array.from(normalizedText);
  const marks = Array.from(PUNCTUATION).filter(mark => input.includes(mark));
  const count = (list: string[], mark: string) => list.filter(item => item === mark).length;

  for (const mark of marks) {
    try {
      const equal = count(input, mark) === count(output, mark);
      let indexIn = 0;
      let indexOut = 0;
      while (input.indexOf(mark, indexIn) >= 0) {
        indexOut = output.indexOf(mark, indexOut);
        if (indexOut < 0) throw new Error('missing mark');
        indexIn = input.indexOf(mark, indexIn);
        const valid = (indexOut > 0 && indexIn > 0 && output[indexOut - 1] === input[indexIn - 1])
          || (indexOut < output.length - 1 && indexIn < input.length - 1 && output[indexOut + 1] === input[indexIn + 1]);
        if (!equal && !valid) {
          indexIn += 1;
          continue;
        }
        if (indexIn > 0 && indexOut > 0) {
          if (output[indexOut - 1] === ' ' && input[indexIn - 1] !== ' ') output[indexOut - 1] = '';
          else if (output[indexOut - 1] !== ' ' && input[indexIn - 1] === ' ') output[indexOut - 1] += ' ';
        }
        if (indexIn < input.length - 1 && indexOut < output.length - 1) {
          if (output[indexOut + 1] === ' ' && input[indexIn + 1] !== ' ') output[indexOut + 1] = '';
          else if (output[indexOut + 1] !== ' ' && input[indexIn + 1] === ' ') output[indexOut] += ' ';
        }
        indexOut += 1;
        indexIn += 1;
      }
    } catch {
      // NeMo logs and skips this mark.
    }
  }
  return output.join('').replace(/ +/g, ' ');
}

// ── detokenization (English) ──────────────────────────────────────────────────

const CJK: ReadonlyArray<readonly [number, number]> = [
  [4352, 4607], [11904, 42191], [43072, 43135], [44032, 55215], [63744, 64255], [65072, 65103],
  [65381, 65500], [94208, 101119], [110592, 110895], [110960, 111359], [131072, 196607],
];

function isCJK(char: string): boolean {
  const code = char.codePointAt(0) ?? 0;
  return CJK.some(([low, high]) => code >= low && code <= high);
}

function isAlpha(char: string): boolean {
  const code = char.codePointAt(0) ?? 0;
  let low = 0;
  let high = ALPHA_RANGES.length / 2 - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (code < ALPHA_RANGES[2 * mid]) high = mid - 1;
    else if (code > ALPHA_RANGES[2 * mid + 1]) low = mid + 1;
    else return true;
  }
  return false;
}

function detokenize(text: string): string {
  const tokens = ` ${text} `.split(' @-@ ').join('-').split(/\s+/).filter(Boolean);
  const quotes = new Map<string, number>();
  let space = ' ';
  let result = '';
  tokens.forEach((token, i) => {
    const chars = Array.from(token);
    if (isCJK(chars[0])) {
      const previous = i ? Array.from(tokens[i - 1]) : [];
      result += (i && isCJK(previous[previous.length - 1]) ? '' : space) + token;
      space = ' ';
    } else if (chars.every(char => CURRENCY_SYMBOLS.includes(char) || '([{¿¡'.includes(char))) {
      result += space + token;
      space = '';
    } else if (/^[,.?!:;\\%}\])]+$/.test(token)) {
      result += token;
      space = ' ';
    } else if (i && chars.length > 1 && chars[0] === "'" && isAlpha(chars[1])) {
      result += token;
      space = ' ';
    } else if (/^['"„“`]+$/.test(token)) {
      const key = chars.every(char => '„“”'.includes(char)) ? '"' : token;
      const seen = quotes.get(key) ?? 0;
      if (seen % 2 || (token === "'" && i && tokens[i - 1].endsWith('s'))) {
        result += token;
        space = ' ';
        if (seen % 2) quotes.set(key, seen + 1);
      } else {
        result += space + token;
        space = '';
        quotes.set(key, seen + 1);
      }
    } else {
      result += space + token;
      space = ' ';
    }
  });
  return result.replace(/ {2,}/g, ' ').trim();
}
