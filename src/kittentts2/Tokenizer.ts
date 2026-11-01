import { encodeUTF8 } from './utf8';

// The Qwen2 pre-tokenizer, with its `(?i:...)` contractions spelled out: JavaScript
// has no inline case-insensitive groups.
const PRETOKENIZE =
  /'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;
// Candidates for added tokens such as <|speech_start|>, <think> or [START]; each
// match is looked up before it is treated as one.
const ADDED_TOKEN = /<\|[^|<>\s]{1,40}\|>|<\/?[a-z_]{1,20}>|\[[A-Z_]{1,20}\]/g;

/**
 * Kitten TTS 2's text tokenizer: Qwen2's byte-level BPE, as `transformers`
 * runs it with `add_special_tokens=False`. NFC first, then the added tokens are
 * split out, the rest is pre-tokenized and each piece is merged by rank.
 */
export class QwenTokenizer {
  private readonly cache = new Map<string, number[]>();

  private constructor(
    private readonly vocab: Map<string, number>,
    private readonly ranks: Map<string, number>,
    private readonly added: Map<string, number>,
  ) {}

  /** From the model repository's vocab.json, merges.txt and added_tokens.json. */
  static fromFiles(vocabJson: string, mergesText: string, addedTokensJson: string): QwenTokenizer {
    const vocab = new Map<string, number>(Object.entries(JSON.parse(vocabJson) as Record<string, number>));
    const ranks = new Map<string, number>();
    let rank = 0;
    for (const line of mergesText.split('\n')) {
      if (!line || line.startsWith('#version')) continue;
      ranks.set(line, rank);
      rank += 1;
    }
    const added = new Map<string, number>(Object.entries(JSON.parse(addedTokensJson) as Record<string, number>));
    return new QwenTokenizer(vocab, ranks, added);
  }

  encode(text: string): number[] {
    const normalized = typeof text.normalize === 'function' ? text.normalize('NFC') : text;
    const ids: number[] = [];
    let start = 0;
    for (const match of normalized.matchAll(ADDED_TOKEN)) {
      const id = this.added.get(match[0]);
      if (id === undefined) continue;
      this.encodeOrdinary(normalized.slice(start, match.index), ids);
      ids.push(id);
      start = (match.index ?? 0) + match[0].length;
    }
    this.encodeOrdinary(normalized.slice(start), ids);
    return ids;
  }

  private encodeOrdinary(text: string, ids: number[]): void {
    if (!text) return;
    for (const match of text.matchAll(PRETOKENIZE)) {
      for (const id of this.encodeWord(match[0])) ids.push(id);
    }
  }

  private encodeWord(word: string): number[] {
    const cached = this.cache.get(word);
    if (cached) return cached;
    let parts = Array.from(encodeUTF8(word), byte => BYTE_TO_UNICODE[byte]);
    while (parts.length > 1) {
      let best = Infinity;
      let bestIndex = -1;
      for (let i = 0; i < parts.length - 1; i += 1) {
        const rank = this.ranks.get(`${parts[i]} ${parts[i + 1]}`);
        if (rank !== undefined && rank < best) {
          best = rank;
          bestIndex = i;
        }
      }
      if (bestIndex < 0) break;
      const first = parts[bestIndex];
      const second = parts[bestIndex + 1];
      const merged: string[] = [];
      for (let i = 0; i < parts.length; i += 1) {
        if (i < parts.length - 1 && parts[i] === first && parts[i + 1] === second) {
          merged.push(first + second);
          i += 1;
        } else {
          merged.push(parts[i]);
        }
      }
      parts = merged;
    }
    const ids = parts.map(part => {
      const id = this.vocab.get(part);
      if (id === undefined) throw new Error(`The tokenizer has no entry for ${JSON.stringify(part)}.`);
      return id;
    });
    if (this.cache.size > 20_000) this.cache.clear();
    this.cache.set(word, ids);
    return ids;
  }
}

// GPT-2's byte-to-character table: printable bytes stand for themselves, the rest
// are moved up past 255 so every byte has a visible character.
const BYTE_TO_UNICODE: string[] = (() => {
  const table: string[] = new Array(256);
  let next = 0;
  for (let byte = 0; byte < 256; byte += 1) {
    const printable = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || (byte >= 174 && byte <= 255);
    table[byte] = String.fromCharCode(printable ? byte : 256 + next++);
  }
  return table;
})();
