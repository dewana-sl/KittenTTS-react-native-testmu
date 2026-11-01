// A port of the Python package's kittentts2/text.py: expression markup,
// written-to-spoken normalization, chunking and joining. The regexes that steer
// the normalizer away from known misreadings are kept as they are, so the text
// reaching the model matches what the Python package sends it.

export const CHUNK_CHARS = 380;
export const CHUNK_MIN_CHARS = 130;
export const CHUNK_GAP_S = 0.16;

const EXPRESSION_TAGS = ['mundane', 'nervous', 'tender', 'angry', 'excited', 'stern', 'sad', 'contemplative', 'surprised', 'joyful'];
const VOCAL_EVENT_TAGS = ['pause', 'sigh', 'gasp', 'laugh', 'giggle', 'sob', 'scoff', 'growl', 'um', 'gulp'];

const TAG_SPAN = new RegExp(`\\[(?:${EXPRESSION_TAGS.join('|')})\\]|<(?:${VOCAL_EVENT_TAGS.join('|')})>`, 'gi');
const EMPHASIS_SPAN = /\(\(\(([^()\n]{1,80})\)\)\)/g;
const placeholder = (index: number) => `${index}`;

const UI_GLYPHS = /[•‣⁃◦▸↳→←↑↓]+/g;

// Raw math glyphs destabilise the model, so they are read the way papers are read aloud.
const MATH_GLYPH_WORDS: Record<string, string> = {
  'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon', 'ζ': 'zeta',
  'η': 'eta', 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu',
  'ν': 'nu', 'ξ': 'xi', 'π': 'pi', 'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau',
  'υ': 'upsilon', 'φ': 'phi', 'ϕ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
  'Γ': 'gamma', 'Δ': 'delta', 'Θ': 'theta', 'Λ': 'lambda', 'Ξ': 'xi', 'Π': 'pi',
  'Σ': 'sigma', 'Φ': 'phi', 'Ψ': 'psi', 'Ω': 'omega',
  '=': 'equals', '≠': 'is not equal to', '≈': 'approximately equals',
  '≤': 'is less than or equal to', '≥': 'is greater than or equal to',
  '±': 'plus or minus', '×': 'times', '÷': 'divided by', '·': 'dot',
  '−': 'minus', '∗': 'star', '∘': 'composed with', '⊤': 'transpose',
  '∼': 'distributed as', '∝': 'proportional to', '≡': 'is equivalent to',
  '∈': 'in', '∉': 'not in', '⊂': 'subset of', '⊆': 'subset of', '∪': 'union',
  '∩': 'intersect', '⊕': 'plus', '⊗': 'times', '^': 'to the',
  '∑': 'sum of', '∏': 'product of', '∫': 'integral of', '∂': 'partial',
  '∇': 'gradient of', '√': 'square root of', '∞': 'infinity', '∆': 'delta',
  '∀': 'for all', '∃': 'there exists', '⇒': 'implies', '⇔': 'if and only if',
  '∥': 'norm', '‖': 'norm', '⌊': 'floor of', '⌋': '', '⌈': 'ceiling of', '⌉': '',
  'ℝ': 'R', 'ℕ': 'N', 'ℤ': 'Z', 'ℚ': 'Q', 'ℂ': 'C', '\u{1d53c}': 'the expectation of',
};
// Super- and subscript digits become plain digits, so "x²" reads as "x 2".
Array.from('⁰¹²³⁴⁵⁶⁷⁸⁹').forEach((glyph, digit) => { MATH_GLYPH_WORDS[glyph] = String(digit); });
Array.from('₀₁₂₃₄₅₆₇₈₉').forEach((glyph, digit) => { MATH_GLYPH_WORDS[glyph] = String(digit); });

const ELLIPSIS = /(?:\.\s+){2,}\.|…|\.\.\./g;
// Paired straight quotes become curly, so a closing quote after a digit is not read as inches.
const STRAIGHT_QUOTE_PAIR = /"([^"\n]*)"/g;
const NONTAG_BRACKET_SPAN = /\[([^[\]\n]{1,400})\]|<([^<>\n]{1,60})>|\(([^()\n]{1,400})\)/g;
const COMMA_COLLISION = /,\s*([,.!?;:])/g;
const TRAILING_NONTERMINAL = /[,;:\-–—]+\s*$/;
const ALREADY_TERMINATED = '.!?"’”\')]';
// Dashes become commas: dash-heavy text was the surest way to make a chunk run away.
const DASH = /\s*(?:[—–]|--)\s*/g;
const DASH_FIX: Array<[RegExp, string]> = [
  [/,\s*,+/g, ', '],
  [/\s+([,.!?;:])/g, '$1'],
  [/,\s*([.!?])/g, '$1'],
  [/\s{2,}/g, ' '],
  [/^\s*,\s*/, ''],
];
const LETTER = /\p{L}/u;

/** True when the text uses an expression tag, a vocal event or an emphasis span. */
export function hasExpressionTag(text: string): boolean {
  return new RegExp(TAG_SPAN.source, 'i').test(text) || new RegExp(EMPHASIS_SPAN.source).test(text);
}

function protectTags(text: string): [string, string[]] {
  const tags: string[] = [];
  const swap = (match: string) => {
    tags.push(match);
    return placeholder(tags.length - 1);
  };
  return [text.replace(TAG_SPAN, swap).replace(EMPHASIS_SPAN, swap), tags];
}

function restoreTags(text: string, tags: string[]): string {
  let out = text;
  tags.forEach((tag, i) => { out = out.split(placeholder(i)).join(tag); });
  return out;
}

function speakMathGlyphs(text: string): string {
  let out = '';
  for (const char of text.replace(ELLIPSIS, ', ')) {
    const word = MATH_GLYPH_WORDS[char];
    out += word === undefined ? char : ` ${word} `;
  }
  return out.replace(/[ \t]{2,}/g, ' ');
}

function reformatNontagBrackets(text: string): string {
  let out = text.replace(NONTAG_BRACKET_SPAN, (_match, square?: string, angle?: string, round?: string) =>
    `, ${(square ?? angle ?? round ?? '').trim()},`);
  for (;;) {
    const collapsed = out.replace(COMMA_COLLISION, '$1');
    if (collapsed === out) return collapsed;
    out = collapsed;
  }
}

function dashesToCommas(text: string): string {
  if (!text) return text;
  let out = text.replace(DASH, ', ');
  for (const [pattern, replacement] of DASH_FIX) out = out.replace(pattern, replacement);
  return out.trim();
}

function capitalizeAndTerminate(text: string): string {
  if (!text) return text;
  let out = text.replace(/\s+([?!])/g, '$1');
  let offset = 0;
  for (const char of out) {
    if (LETTER.test(char)) {
      out = out.slice(0, offset) + char.toUpperCase() + out.slice(offset + char.length);
      break;
    }
    offset += char.length;
  }
  let stripped = out.trimEnd();
  if (!stripped) return out;
  if (ALREADY_TERMINATED.includes(stripped[stripped.length - 1])) return stripped;
  stripped = stripped.replace(TRAILING_NONTERMINAL, '').trimEnd();
  return stripped ? `${stripped}.` : stripped;
}

/**
 * The spoken form the model hears for `text`. Never throws: a normalizer
 * failure falls back to the lightly cleaned text. Tags survive untouched.
 */
export function normalizeText(
  text: string,
  backend: ((text: string) => string) | null,
  normalize = true,
): string {
  let [protectedText, tags] = protectTags(String(text ?? ''));
  // A hard line wrap would reach the sentence splitter as a missing space.
  protectedText = protectedText.replace(/\s*\n+\s*/g, ' ');
  protectedText = protectedText.replace(/(?<=[a-z])([.!?;:])(?=[A-Z])/g, '$1 ');
  protectedText = speakMathGlyphs(protectedText);
  // "V0.8" -> "V 0 point 8": fused version numbers otherwise read as "dot".
  protectedText = protectedText.replace(/\b([vV])(\d+(?:\.\d+)+)/g,
    (_match, letter: string, version: string) => `${letter} ${version.split('.').join(' point ')}`);
  // A ?/! right after a digit would be read aloud by the number grammar.
  protectedText = protectedText.replace(/(?<=\d)\s*([?!])(?=\s|$)/g, ' $1');

  const raw = dashesToCommas(reformatNontagBrackets(protectedText).replace(UI_GLYPHS, ' '));
  if (!raw.trim()) return restoreTags(raw, tags);
  if (!normalize || !backend) return restoreTags(capitalizeAndTerminate(raw), tags);
  let spoken = '';
  try {
    spoken = (backend(raw.replace(STRAIGHT_QUOTE_PAIR, '“$1”')) || '').trim();
  } catch {
    spoken = ''; // a normalizer problem must not cost a generation
  }
  if (!spoken) return restoreTags(capitalizeAndTerminate(raw), tags);
  return restoreTags(capitalizeAndTerminate(dashesToCommas(spoken)), tags);
}

// ── chunking ──────────────────────────────────────────────────────────────────

const SENTENCE_END = /(?<=[.!?])["’”')\]]*\s+/;
const TERMINAL = '.!?,;:';

/** Sentence-aware greedy packing; text that fits comes back as one chunk. */
export function splitForSynthesis(text: string, maxChars = CHUNK_CHARS, minChars = CHUNK_MIN_CHARS): string[] {
  const stripped = (text ?? '').trim();
  if (!maxChars || stripped.length <= maxChars) return stripped ? [stripped] : [];
  const out: string[] = [];
  let current = '';
  for (const piece of stripped.split(SENTENCE_END)) {
    if (!piece || !piece.trim()) continue;
    let sentence = piece.trim();
    while (sentence.length > maxChars) {
      // Break a long sentence at its last clause boundary before the limit.
      const cut = Math.max(rfind(sentence, ', ', maxChars), rfind(sentence, '; ', maxChars), rfind(sentence, ' — ', maxChars));
      const at = cut > Math.floor(maxChars / 3) ? cut + 1 : maxChars;
      const head = sentence.slice(0, at).trim();
      sentence = sentence.slice(at).trim();
      if (current) {
        out.push(current);
        current = '';
      }
      out.push(head);
    }
    if (!current) current = sentence;
    else if (current.length + 1 + sentence.length <= maxChars) current += ` ${sentence}`;
    else {
      out.push(current);
      current = sentence;
    }
  }
  if (current) out.push(current);
  return mergeShort(out, maxChars, minChars).map(ensureTerminalPunctuation);
}

/** Python's str.rfind(sub, 0, end). */
function rfind(text: string, sub: string, end: number): number {
  const from = Math.min(end, text.length) - sub.length;
  return from < 0 ? -1 : text.lastIndexOf(sub, from);
}

/** Fold undersized chunks into a neighbour; short chunks are the unstable ones. */
function mergeShort(parts: string[], maxChars: number, minChars: number): string[] {
  if (!parts.length || minChars <= 0) return parts;
  const out = [...parts];
  const limit = Math.floor(maxChars * 1.4);
  let changed = true;
  while (changed && out.length > 1) {
    changed = false;
    for (let i = 0; i < out.length; i += 1) {
      const chunk = out[i];
      if (chunk.length >= minChars) continue;
      const previousOk = i > 0 && out[i - 1].length + 1 + chunk.length <= limit;
      const nextOk = i < out.length - 1 && chunk.length + 1 + out[i + 1].length <= limit;
      if (previousOk && (!nextOk || out[i - 1].length <= out[i + 1].length)) {
        out[i - 1] = `${out[i - 1].trimEnd()} ${chunk.trimStart()}`;
        out.splice(i, 1);
      } else if (nextOk) {
        out[i] = `${chunk.trimEnd()} ${out[i + 1].trimStart()}`;
        out.splice(i + 1, 1);
      } else {
        continue;
      }
      changed = true;
      break;
    }
  }
  return out;
}

function ensureTerminalPunctuation(chunk: string): string {
  const c = (chunk ?? '').trimEnd();
  if (!c) return c;
  const last = c[c.length - 1];
  return TERMINAL.includes(last) || '"’”\')]'.includes(last) ? c : `${c},`;
}

// ── joining ───────────────────────────────────────────────────────────────────

/** Strip the lead-in and trail-out silence of one generated chunk. */
export function trimSilence(wav: Float32Array, sampleRate: number, thresholdDb = -45, keepSeconds = 0.02): Float32Array {
  if (wav.length < Math.floor(sampleRate * 0.05)) return wav;
  const hop = Math.max(1, Math.floor(sampleRate * 0.01));
  const threshold = 10 ** (thresholdDb / 20);
  const frames = Math.floor(wav.length / hop);
  let first = -1;
  let last = -1;
  for (let f = 0; f < frames; f += 1) {
    let sum = 0;
    for (let i = f * hop; i < (f + 1) * hop; i += 1) sum += wav[i] * wav[i];
    if (Math.sqrt(sum / hop + 1e-12) > threshold) {
      if (first < 0) first = f;
      last = f;
    }
  }
  if (first < 0) return wav;
  const keep = Math.floor(sampleRate * keepSeconds);
  return wav.slice(Math.max(0, first * hop - keep), Math.min(wav.length, (last + 1) * hop + keep));
}

/** Concatenate chunks with one consistent pause and short fades, so seams are inaudible. */
export function joinChunks(wavs: Float32Array[], sampleRate: number, gapSeconds = CHUNK_GAP_S): Float32Array | null {
  const parts = wavs.filter(wav => wav && wav.length);
  if (!parts.length) return null;
  if (parts.length === 1) return parts[0];
  const gap = Math.floor(sampleRate * gapSeconds);
  const fade = Math.max(1, Math.floor(sampleRate * 0.008));
  const trimmed = parts.map(part => {
    const wav = Float32Array.from(trimSilence(part, sampleRate));
    if (wav.length > 2 * fade) {
      for (let i = 0; i < fade; i += 1) {
        const ramp = fade === 1 ? 0 : i / (fade - 1);
        wav[i] *= ramp;
        wav[wav.length - 1 - i] *= ramp;
      }
    }
    return wav;
  });
  const out = new Float32Array(trimmed.reduce((sum, wav) => sum + wav.length, 0) + gap * (trimmed.length - 1));
  let offset = 0;
  trimmed.forEach((wav, i) => {
    out.set(wav, offset);
    offset += wav.length + (i < trimmed.length - 1 ? gap : 0);
  });
  return out;
}
