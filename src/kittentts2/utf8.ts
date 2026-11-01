// Hermes has no TextDecoder, and the tokenizer and the normalizer grammars both
// work on UTF-8 bytes, so the two conversions are written out here.

export function encodeUTF8(text: string): Uint8Array {
  const bytes: number[] = [];
  for (let i = 0; i < text.length; i += 1) {
    let code = text.charCodeAt(i);
    if (code >= 0xd800 && code < 0xdc00 && i + 1 < text.length) {
      const low = text.charCodeAt(i + 1);
      if (low >= 0xdc00 && low < 0xe000) {
        code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
        i += 1;
      }
    }
    if (code >= 0xd800 && code < 0xe000) code = 0xfffd; // a lone surrogate
    if (code < 0x80) {
      bytes.push(code);
    } else if (code < 0x800) {
      bytes.push(0xc0 | (code >> 6), 0x80 | (code & 63));
    } else if (code < 0x10000) {
      bytes.push(0xe0 | (code >> 12), 0x80 | ((code >> 6) & 63), 0x80 | (code & 63));
    } else {
      bytes.push(0xf0 | (code >> 18), 0x80 | ((code >> 12) & 63), 0x80 | ((code >> 6) & 63), 0x80 | (code & 63));
    }
  }
  return Uint8Array.from(bytes);
}

export function decodeUTF8(bytes: ArrayLike<number>): string {
  let out = '';
  let i = 0;
  while (i < bytes.length) {
    const first = bytes[i];
    let code = 0xfffd;
    let size = 1;
    if (first < 0x80) {
      code = first;
    } else if (first >= 0xc2 && first < 0xe0 && isContinuation(bytes, i + 1, 1)) {
      code = ((first & 31) << 6) | (bytes[i + 1] & 63);
      size = 2;
    } else if (first >= 0xe0 && first < 0xf0 && isContinuation(bytes, i + 1, 2)) {
      code = ((first & 15) << 12) | ((bytes[i + 1] & 63) << 6) | (bytes[i + 2] & 63);
      size = 3;
    } else if (first >= 0xf0 && first < 0xf5 && isContinuation(bytes, i + 1, 3)) {
      code = ((first & 7) << 18) | ((bytes[i + 1] & 63) << 12) | ((bytes[i + 2] & 63) << 6) | (bytes[i + 3] & 63);
      size = 4;
    }
    out += String.fromCodePoint(code);
    i += size;
  }
  return out;
}

function isContinuation(bytes: ArrayLike<number>, start: number, count: number): boolean {
  if (start + count > bytes.length) return false;
  for (let i = start; i < start + count; i += 1) {
    if ((bytes[i] & 0xc0) !== 0x80) return false;
  }
  return true;
}
