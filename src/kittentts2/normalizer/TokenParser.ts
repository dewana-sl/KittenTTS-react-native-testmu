// A port of NeMo 1.2.0's TokenParser (Apache-2.0, Copyright (c) 2021 NVIDIA
// CORPORATION), as kitten-text-processing ships it. It parses the tagger's
// output, e.g. 'tokens { money { integer: "20" currency: "$" } } tokens { name: "left" }'.

/** A token value: text, `true` for preserve_order, a nested group, or null for "". */
export type TokenValue = string | true | null | TokenGroup;
export type TokenGroup = Map<string, TokenValue>;

export const PRESERVE_ORDER_KEY = 'preserve_order';

/** NeMo raises ValueError here; the normalizer answers those with the escaped input. */
export class TokenValueError extends Error {}

const WHITESPACE = ' \t\n\r\u000b\u000c';

export class TokenParser {
  private index = 0;
  private char: string | null;

  constructor(private readonly text: string) {
    if (!text) throw new Error('Cannot parse an empty tagger output.');
    this.char = text[0];
  }

  parse(): TokenGroup[] {
    const tokens: TokenGroup[] = [];
    while (this.parseWhitespace()) {
      const token = this.parseToken();
      if (!token) break;
      tokens.push(token);
    }
    return tokens;
  }

  private parseToken(): TokenGroup | null {
    const key = this.parseStringKey();
    if (key === null) return null;
    this.parseWhitespace();
    let value: TokenValue;
    if (key === PRESERVE_ORDER_KEY) {
      this.parseChar(':');
      this.parseWhitespace();
      for (const char of 'true') this.parseChar(char);
      value = true;
    } else {
      value = this.parseTokenValue();
    }
    return new Map([[key, value]]);
  }

  private parseTokenValue(): TokenValue {
    if (this.char === ':') {
      this.parseChar(':');
      this.parseWhitespace();
      this.parseChar('"');
      const value = this.parseStringValue();
      this.parseChar('"');
      return value;
    }
    if (this.char === '{') {
      const group: TokenGroup = new Map();
      this.parseChar('{');
      for (const token of this.parse()) {
        for (const [key, value] of token) group.set(key, value);
      }
      this.parseChar('}');
      return group;
    }
    throw new TokenValueError();
  }

  private parseChar(expected: string): void {
    if (this.char !== expected) throw new Error(`Expected ${expected} in the tagger output.`);
    this.read();
  }

  private parseStringKey(): string | null {
    if (this.char === null || WHITESPACE.includes(this.char)) {
      throw new Error('Expected a key in the tagger output.');
    }
    let key = '';
    while (this.char !== null && /^[A-Za-z_]$/.test(this.char)) {
      key += this.char;
      if (!this.read()) throw new TokenValueError();
    }
    return key || null;
  }

  private parseStringValue(): string | null {
    if (this.char === null) throw new Error('Expected a value in the tagger output.');
    let value = '';
    for (;;) {
      if (this.char === '"') {
        if (this.index + 1 >= this.text.length) throw new Error('Unterminated value in the tagger output.');
        if (this.text[this.index + 1] === ' ') break;
      }
      value += this.char;
      if (!this.read()) throw new TokenValueError();
    }
    return value || null;
  }

  private parseWhitespace(): boolean {
    let notEnd = this.char !== null;
    while (notEnd && this.char === ' ') notEnd = this.read();
    return notEnd;
  }

  private read(): boolean {
    if (this.index < this.text.length - 1) {
      this.index += 1;
      this.char = this.text[this.index];
      return true;
    }
    this.char = null;
    return false;
  }
}
