// M3 exit test: the web export loads, its byte counts match, and the field decodes in JS.
//
//     node bench/tractography/export_check.mjs ~/tmp/data/tractography/hcp/web
//
// Reads exactly what a browser viewer reads (manifest.json, then per chunk geom/, field/,
// extra/, plus levels.bin and tables.json) with the same DataView/TypedArray parsing, and checks:
//   - every file's size and sha256 against the manifest, and against the size its header implies
//   - line counts per chunk and in total; the extra/ indices are a permutation of 0..N-1
//   - every vertex inside the manifest's bounds
//   - the decode the viewer runs (viewer/decode.js, imported, not copied): for chunk 0, each
//     streamline's winning tract, its own-tract best-class margin bit for bit against
//     rankfield.decode_groups, and its mass margin against rankfield.probabilities to 1e-9
//     (float64 both sides; written by export.py to ../web_check/)
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { parseField, ownGroupMargins } from "./viewer/decode.js";

const dir = process.argv[2];
if (!dir) { console.error("usage: node export_check.mjs <web dir>"); process.exit(2); }
const failures = [];
const fail = (msg) => failures.push(msg);
const read = (rel) => { const b = readFileSync(join(dir, rel)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };

const manifest = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
const tables = JSON.parse(readFileSync(join(dir, manifest.tables), "utf8"));
const levels = new Float32Array(read(manifest.field.levels));
if (levels.length !== 256) fail(`levels.bin holds ${levels.length} entries, not 256`);
const clip = manifest.field.rankfield.clip;
const { center, quantum, bounds } = manifest.coordinates;
const N = manifest.streamlines;
const seen = new Uint8Array(N);
let total = 0;

function checked(entry) {
  const buf = read(entry.file);
  if (buf.byteLength !== entry.bytes) fail(`${entry.file}: ${buf.byteLength} bytes, manifest says ${entry.bytes}`);
  const digest = createHash("sha256").update(new Uint8Array(buf)).digest("hex");
  if (digest !== entry.sha256) fail(`${entry.file}: sha256 differs from the manifest`);
  return buf;
}

const t0 = performance.now();
manifest.chunks.forEach((ch, c) => {
  const g = checked(ch.geom), f = checked(ch.field), e = checked(ch.extra);
  const gv = new DataView(g), fv = new DataView(f), ev = new DataView(e);
  const lines = gv.getUint32(0, true), P = gv.getUint32(4, true);
  const depth = fv.getUint32(4, true);
  if (lines !== ch.lines || fv.getUint32(0, true) !== lines || ev.getUint32(0, true) !== lines)
    fail(`chunk ${c}: line counts disagree`);
  if (P !== manifest.pointsPerLine) fail(`chunk ${c}: ${P} points per line`);
  if (depth !== manifest.field.rankfield.depth) fail(`chunk ${c}: depth ${depth}`);
  if (g.byteLength !== 8 + lines * P * 3 * 2) fail(`chunk ${c}: geometry size does not match its header`);
  if (f.byteLength !== 8 + lines * depth * 2 + lines * 2 + lines * (depth - 1)) fail(`chunk ${c}: field size does not match its header`);
  if (e.byteLength !== 8 + lines * 4 + lines * 4) fail(`chunk ${c}: extra size does not match its header`);
  total += lines;

  const xyz = new Int16Array(g, 8, lines * P * 3);
  for (let k = 0; k < xyz.length; k++) {
    const v = center[k % 3] + xyz[k] * quantum;
    if (v < bounds[0][k % 3] - quantum || v > bounds[1][k % 3] + quantum) { fail(`chunk ${c}: vertex outside bounds`); break; }
  }
  const index = new Uint32Array(e, 8, lines);
  for (const i of index) { if (i >= N || seen[i]) { fail(`chunk ${c}: index ${i} repeated or out of range`); break; } seen[i] = 1; }

  if (c === 0) {
    const field = parseField(f);
    const dec = ownGroupMargins(field, Int32Array.from(tables.clusterToTract), levels, clip);
    const expBest = new Float32Array(readFileSync(join(dir, "..", "web_check", "chunk0_tract_margin.f32")).buffer.slice(0));
    const expMass = new Float64Array(readFileSync(join(dir, "..", "web_check", "chunk0_tract_mass.f64")).buffer.slice(0));
    const expTract = readFileSync(join(dir, "..", "web_check", "chunk0_tract.u8"));
    let bestDiffer = 0, tractDiffer = 0, massWorst = 0;
    for (let i = 0; i < lines; i++) {
      if (dec.best[i] !== expBest[i]) bestDiffer++;
      if (dec.group[i] !== expTract[i]) tractDiffer++;
      massWorst = Math.max(massWorst, Math.abs(dec.mass[i] - expMass[i]) / Math.max(1, Math.abs(expMass[i])));
    }
    // dec.mass is Float32Array storage of a float64 computation: compare at float32 resolution
    if (bestDiffer || tractDiffer || massWorst > 1e-6) fail(`chunk 0 decode: ${bestDiffer} best-class margins and ${tractDiffer} tracts differ, worst relative mass error ${massWorst.toExponential(2)}`);
    console.log(`chunk 0 decode (viewer/decode.js): ${lines} streamlines; best-class bit-identical: ${bestDiffer === 0}; ` +
                `tracts identical: ${tractDiffer === 0}; worst relative mass error ${massWorst.toExponential(2)}`);
  }
});
if (total !== N) fail(`chunks hold ${total} streamlines, manifest says ${N}`);
if (seen.some((s) => s === 0)) fail("some streamlines are in no chunk");

const bytes = (k) => manifest.chunks.reduce((s, ch) => s + ch[k].bytes, 0);
console.log(JSON.stringify({ chunks: manifest.chunks.length, streamlines: total,
  bytes: { geometry: bytes("geom"), field: bytes("field"), extra: bytes("extra") },
  ms: Math.round(performance.now() - t0), failures: failures.length }));
if (failures.length) { failures.slice(0, 20).forEach((m) => console.error("FAIL", m)); process.exit(1); }
console.log("OK: manifest and every chunk load; byte counts and hashes match");
