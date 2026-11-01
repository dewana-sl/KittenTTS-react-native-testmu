// A port of kitten-text-processing's _fst.py: byte-level tropical weighted
// transducers. The decoding order, tie handling and float32 rounding follow it
// step by step, because equal-weight paths are resolved by that order and the
// text the model hears has to match what the Python package produces.

import { decodeUTF8, encodeUTF8 } from '../utf8';

/** The grammar has no accepting path for this input (a ValueError in Python). */
export class NoPathError extends Error {}

interface Arc {
  target: number;
  weight: number;
  /** Output bytes as one character per byte. */
  suffix: string;
}

type Edge = [target: number, weight: number, output: string];

const MAGIC = 'KITTEN1\0';
const fround = Math.fround;

export class Graph {
  private readonly start: number;
  private readonly offsets: Uint32Array;
  private readonly finals: Float32Array;
  private readonly labels: Uint32Array;
  private readonly targets: Uint32Array;
  private readonly weights: Float32Array;
  private readonly states: number;
  private index = new Map<number, Map<number, Arc[]>>();
  private indegree: Uint8Array | undefined;

  /** `data` is the grammar file after gunzip. */
  constructor(data: Uint8Array, private readonly projectOutput = false) {
    const magic = String.fromCharCode(...data.subarray(0, 8));
    if (magic !== MAGIC) throw new Error('Invalid grammar file.');
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    this.start = view.getUint32(8, true);
    const states = view.getUint32(12, true);
    const arcs = view.getUint32(16, true);
    this.states = states;
    let offset = 20;
    const take = <T>(make: (buffer: ArrayBuffer) => T, count: number): T => {
      const end = offset + count * 4;
      if (end > data.byteLength) throw new Error('Truncated grammar file.');
      const copy = data.slice(offset, end).buffer as ArrayBuffer;
      offset = end;
      return make(copy);
    };
    this.offsets = take(buffer => new Uint32Array(buffer), states + 1);
    this.finals = take(buffer => new Float32Array(buffer), states);
    this.labels = take(buffer => new Uint32Array(buffer), arcs);
    this.targets = take(buffer => new Uint32Array(buffer), arcs);
    this.weights = take(buffer => new Float32Array(buffer), arcs);
  }

  rewrite(input: string): string {
    // Pynini compiles NUL bytes as epsilon labels, so they consume no input.
    const text = input.split('\0').join('');
    if (this.projectOutput) return this.rewriteOrdered(text);
    const bytes = encodeUTF8(text);
    let active = new Map<number, [number, string, boolean]>([[this.start, [0, '', false]]]);
    for (let position = 0; position <= bytes.length; position += 1) {
      const queue = [...active.keys()];
      const queued = new Set(queue);
      for (let head = 0; head < queue.length; head += 1) {
        const state = queue[head];
        queued.delete(state);
        const [cost, output, ambiguous] = active.get(state)!;
        for (const arc of this.arcs(state).get(0) ?? []) {
          const candidate = arc.weight ? fround(cost + arc.weight) : cost;
          const old = active.get(arc.target);
          const value = output + arc.suffix;
          if (old === undefined || candidate < old[0]) {
            active.set(arc.target, [candidate, value, ambiguous]);
          } else if (candidate === old[0] && !old[2] && (ambiguous || value !== old[1])) {
            active.set(arc.target, [old[0], old[1], true]);
          } else {
            continue;
          }
          if (!queued.has(arc.target)) {
            queue.push(arc.target);
            queued.add(arc.target);
          }
        }
      }
      if (position === bytes.length) {
        let best: [number, string, boolean] | undefined;
        for (const [state, [cost, output, ambiguous]] of active) {
          const final = this.finals[state];
          if (!Number.isFinite(final)) continue;
          const candidate = fround(cost + final);
          if (best === undefined || candidate < best[0]) {
            best = [candidate, output, ambiguous];
          } else if (candidate === best[0] && (ambiguous || output !== best[1])) {
            best = [best[0], best[1], true];
          }
        }
        if (best === undefined) throw new NoPathError(`No accepting path for ${JSON.stringify(text.slice(0, 80))}`);
        if (best[2]) return this.rewriteOrdered(text);
        return decodeBytes(best[1]);
      }
      const byte = bytes[position];
      const following = new Map<number, [number, string, boolean]>();
      for (const [state, [cost, output, ambiguous]] of active) {
        for (const arc of this.arcs(state).get(byte) ?? []) {
          const candidate = arc.weight ? fround(cost + arc.weight) : cost;
          const old = following.get(arc.target);
          const value = output + arc.suffix;
          if (old === undefined || candidate < old[0]) {
            following.set(arc.target, [candidate, value, ambiguous]);
          } else if (candidate === old[0] && !old[2] && (ambiguous || value !== old[1])) {
            following.set(arc.target, [old[0], old[1], true]);
          }
        }
      }
      if (following.size === 0) throw new NoPathError(`No accepting path for ${JSON.stringify(text.slice(0, 80))}`);
      active = following;
    }
    throw new NoPathError('unreachable');
  }

  private arcs(state: number): Map<number, Arc[]> {
    const cached = this.index.get(state);
    if (cached) return cached;
    if (!this.indegree) {
      const incoming = new Uint8Array(this.states);
      for (let i = 0; i < this.targets.length; i += 1) {
        const target = this.targets[i];
        if (incoming[target] < 2) incoming[target] += 1;
      }
      this.indegree = incoming;
    }
    const incoming = this.indegree;
    const index = new Map<number, Arc[]>();
    for (let i = this.offsets[state]; i < this.offsets[state + 1]; i += 1) {
      const label = this.labels[i];
      let target = this.targets[i];
      let suffix = label >> 8 ? String.fromCharCode(label >> 8) : '';
      // Contract deterministic zero-cost output-only chains, but never across a
      // join or a nonzero weight: float32 addition is not associative, and a
      // join's order resolves equal-weight paths.
      const visited = new Set<number>();
      while (!this.projectOutput && incoming[target] === 1 && !Number.isFinite(this.finals[target]) && !visited.has(target)) {
        const start = this.offsets[target];
        const end = this.offsets[target + 1];
        if (end !== start + 1 || (this.labels[start] & 255) || this.weights[start]) break;
        if (incoming[this.targets[start]] !== 1) break;
        visited.add(target);
        const output = this.labels[start] >> 8;
        if (output) suffix += String.fromCharCode(output);
        target = this.targets[start];
      }
      const key = label & 255;
      const list = index.get(key);
      const arc = { target, weight: this.weights[i], suffix };
      if (list) list.push(arc);
      else index.set(key, [arc]);
    }
    if (this.index.size >= 50_000) this.index.clear();
    this.index.set(state, index);
    return index;
  }

  private rewriteOrdered(text: string): string {
    const data = encodeUTF8(text);
    const length = data.length;
    const states = this.states;
    const node = (position: number, state: number) => position * states + state;
    const positionOf = (key: number) => Math.floor(key / states);
    const stateOf = (key: number) => key % states;
    const start = node(0, this.start);
    let adjacency = new Map<number, Edge[]>();

    const edges = (key: number): Edge[] => {
      const position = positionOf(key);
      const arcs = this.arcs(stateOf(key));
      const out: Edge[] = (arcs.get(0) ?? []).map(arc => [node(position, arc.target), arc.weight, arc.suffix]);
      if (position < length) {
        for (const arc of arcs.get(data[position]) ?? []) out.push([node(position + 1, arc.target), arc.weight, arc.suffix]);
      }
      adjacency.set(key, out);
      return out;
    };

    // Reverse DFS finishing order matches OpenFst's topological queue, which
    // decides the order of equal-weight readings.
    const seen = new Set([start]);
    const visiting = new Set([start]);
    const stack: Array<[number, Edge[], number]> = [[start, edges(start), 0]];
    let ordered: number[] = [];
    let live = new Set<number>();
    let cyclic = false;
    while (stack.length) {
      const top = stack[stack.length - 1];
      const [key, list] = top;
      if (top[2] >= list.length) {
        ordered.push(key);
        if ((positionOf(key) === length && Number.isFinite(this.finals[stateOf(key)]))
          || adjacency.get(key)!.some(([target]) => live.has(target))) {
          live.add(key);
        }
        visiting.delete(key);
        stack.pop();
        continue;
      }
      const target = list[top[2]][0];
      top[2] += 1;
      if (visiting.has(target)) cyclic = true;
      if (!seen.has(target)) {
        seen.add(target);
        visiting.add(target);
        stack.push([target, edges(target), 0]);
      }
    }
    // Some grammars contain dead epsilon cycles; back edges can hide productive
    // exits, so liveness is finished by reverse reachability for those.
    if (cyclic) {
      const reverse = new Map<number, number[]>();
      for (const [key, outgoing] of adjacency) {
        for (const [target] of outgoing) {
          const parents = reverse.get(target);
          if (parents) parents.push(key);
          else reverse.set(target, [key]);
        }
      }
      const pending = [...live];
      while (pending.length) {
        for (const parent of reverse.get(pending.pop()!) ?? []) {
          if (!live.has(parent)) {
            live.add(parent);
            pending.push(parent);
          }
        }
      }
    }
    if (!live.has(start)) throw new NoPathError(`No accepting path for ${JSON.stringify(text.slice(0, 80))}`);
    let finalWeights = new Map<number, number>();
    for (const key of live) {
      if (positionOf(key) === length) finalWeights.set(key, this.finals[stateOf(key)]);
    }
    if (this.projectOutput) {
      [adjacency, finalWeights, ordered] = projectLattice(adjacency, finalWeights, ordered, live, start);
      live = new Set(ordered);
    }

    const costs = new Map<number, number>([[start, 0]]);
    const parents = new Map<number, [number, string]>();
    let best = Infinity;
    let final: number | undefined;
    const pending = [...ordered].reverse().filter(key => live.has(key));
    const queued = new Set(pending);
    for (let head = 0; head < pending.length; head += 1) {
      const key = pending[head];
      queued.delete(key);
      const cost = costs.get(key) ?? Infinity;
      if (!Number.isFinite(cost)) continue;
      const finalWeight = finalWeights.get(key) ?? Infinity;
      if (Number.isFinite(finalWeight)) {
        const candidate = fround(cost + finalWeight);
        if (candidate < best) {
          best = candidate;
          final = key;
        }
      }
      for (const [target, weight, suffix] of adjacency.get(key)!) {
        if (!live.has(target)) continue;
        const candidate = weight ? fround(cost + weight) : cost;
        if (candidate < (costs.get(target) ?? Infinity)) {
          costs.set(target, candidate);
          parents.set(target, [key, suffix]);
          // Usually a DAG; a live epsilon cycle that improves a visited state requeues it.
          if (!queued.has(target)) {
            pending.push(target);
            queued.add(target);
          }
        }
      }
    }
    if (final === undefined) throw new NoPathError(`No accepting path for ${JSON.stringify(text.slice(0, 80))}`);
    const pieces: string[] = [];
    let at: number = final;
    while (at !== start) {
      const [parent, suffix]: [number, string] = parents.get(at)!;
      pieces.push(suffix);
      at = parent;
    }
    return decodeBytes(pieces.reverse().join(''));
  }
}

/**
 * Output projection and epsilon removal, as Pynini's top_rewrite does it. The
 * insertion order and reversal of expanded arcs follow OpenFst's RmEpsilon.
 */
function projectLattice(
  adjacency: Map<number, Edge[]>,
  finals: Map<number, number>,
  ordered: number[],
  live: Set<number>,
  start: number,
): [Map<number, Edge[]>, Map<number, number>, number[]] {
  const add = (a: number, b: number) => (b ? fround(a + b) : a);
  const incoming = new Set([start]);
  for (const key of live) {
    for (const [target, , output] of adjacency.get(key)!) {
      if (output && live.has(target)) incoming.add(target);
    }
  }
  for (const source of ordered) {
    if (!live.has(source) || !incoming.has(source)) continue;
    const distances = new Map<number, number>([[source, 0]]);
    const pending = [source];
    const queued = new Set([source]);
    for (let head = 0; head < pending.length; head += 1) {
      const key = pending[head];
      queued.delete(key);
      for (const [target, weight, output] of adjacency.get(key)!) {
        if (output || !live.has(target)) continue;
        const distance = add(distances.get(key)!, weight);
        if (distance < (distances.get(target) ?? Infinity)) {
          distances.set(target, distance);
          if (!queued.has(target)) {
            pending.push(target);
            queued.add(target);
          }
        }
      }
    }
    const expanded = new Map<string, Edge>();
    let final = Infinity;
    const stack = [source];
    const seen = new Set<number>();
    while (stack.length) {
      const key = stack.pop()!;
      if (seen.has(key)) continue;
      seen.add(key);
      const distance = distances.get(key)!;
      final = Math.min(final, add(distance, finals.get(key) ?? Infinity));
      for (const [target, weight, output] of adjacency.get(key)!) {
        if (!live.has(target)) continue;
        if (!output) {
          if (!seen.has(target)) stack.push(target);
        } else {
          const id = `${target}\u0000${output}`;
          const old = expanded.get(id);
          const value = Math.min(old ? old[1] : Infinity, add(distance, weight));
          if (old) old[1] = value;
          else expanded.set(id, [target, value, output]);
        }
      }
    }
    adjacency.set(source, [...expanded.values()].reverse());
    finals.set(source, final);
  }
  const seen = new Set([start]);
  const order: number[] = [];
  const stack: Array<[number, Edge[], number]> = [[start, adjacency.get(start)!, 0]];
  while (stack.length) {
    const top = stack[stack.length - 1];
    if (top[2] >= top[1].length) {
      order.push(top[0]);
      stack.pop();
      continue;
    }
    const target = top[1][top[2]][0];
    top[2] += 1;
    if (!seen.has(target)) {
      seen.add(target);
      stack.push([target, adjacency.get(target)!, 0]);
    }
  }
  return [adjacency, finals, order];
}

function decodeBytes(text: string): string {
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i += 1) bytes[i] = text.charCodeAt(i);
  return decodeUTF8(bytes);
}
