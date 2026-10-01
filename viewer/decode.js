// The field decode the viewer runs: rankfield's planes for one chunk -> per-streamline group
// and its two margins. One statement of the algorithm; export_check.mjs holds the
// best-class half to rankfield.decode_groups bit for bit.
//
// Field chunk layout (export.py): u32 lines, u32 depth, u16 ranks[depth][lines] (class + 1,
// 0 = absent, sentinels form a suffix), u16 tail[lines] (dropped mass / 65535),
// u8 support[depth-1][lines] (byte -> gap in logits through levels; 0 = absent).

export const TAIL_MAX = 65535;

export function parseField(buf) {
  const v = new DataView(buf);
  const lines = v.getUint32(0, true), depth = v.getUint32(4, true);
  const ranks = new Uint16Array(buf, 8, depth * lines);
  const tail = new Uint16Array(buf, 8 + depth * lines * 2, lines);
  const support = new Uint8Array(buf, 8 + depth * lines * 2 + lines * 2, (depth - 1) * lines);
  return { lines, depth, ranks, tail, support };
}

// For each streamline, its own group (the group of its winning class) and:
//   best: the winner's lead over the best kept class OUTSIDE the group, floored at the clip
//         (rankfield.decode_groups, disjoint groups) - reads `clip` when no outsider was kept.
//         An upper bound where the depth cut dropped the nearest outsider.
//   mass: log P(group) - log P(not group), with the stored classes' probabilities and the
//         dropped mass (the tail) counted against the group - a lower bound. Floored at half
//         a tail quantum, as encode.py's check does.
export function ownGroupMargins(field, classToGroup, levels, clip, out = null) {
  const { lines, depth, ranks, tail, support } = field;
  const group = out?.group ?? new Int32Array(lines);
  const best = out?.best ?? new Float32Array(lines);
  const mass = out?.mass ?? new Float32Array(lines);
  const floor = 0.5 / TAIL_MAX;
  for (let i = 0; i < lines; i++) {
    const own = classToGroup[ranks[i] - 1];
    let outB = 0, wIn = 1, wOut = 0, wAll = 1;          // the winner: gap 0, weight exp(0)
    for (let j = 1; j < depth; j++) {
      const r = ranks[j * lines + i];
      if (r === 0) break;
      const s = support[(j - 1) * lines + i];
      const w = Math.exp(-levels[s]);
      wAll += w;
      if (classToGroup[r - 1] === own) wIn += w;
      else { wOut += w; if (s > outB) outB = s; }
    }
    const t = tail[i] / TAIL_MAX;
    const scale = (1 - t) / wAll;
    group[i] = own;
    best[i] = Math.min(levels[outB], clip);
    mass[i] = Math.log(wIn * scale) - Math.log(Math.max(wOut * scale + t, floor));
  }
  return { group, best, mass };
}
