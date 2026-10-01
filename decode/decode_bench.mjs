// Decode timing in JS (Node): TRAKO's raw Draco position buffer through the official draco3d
// WASM decoder, against the predictive encoding through numcodecs' Blosc (WASM) plus a plain
// JS reconstruction loop. Both produce Float32Array positions for all 21.6 M vertices.
//
// Inputs from decode_prep.py, in DATA/decode. Bare imports resolve next to the script, so it
// runs from there:
//     cd DATA/decode && npm install draco3d numcodecs
//     cp <repo>/bench/tractography/decode/decode_bench.mjs . && node decode_bench.mjs <repo>/bench/tractography/results/decode_js.json
import { readFileSync, writeFileSync } from "node:fs";
import draco3d from "draco3d";
import { Blosc } from "numcodecs";

const dm = await draco3d.createDecoderModule({});
const blosc = new Blosc();

function readNpyF32(path) {
  const b = readFileSync(path);
  const hlen = b.readUInt16LE(8);
  return new Float32Array(b.buffer.slice(b.byteOffset + 10 + hlen, b.byteOffset + b.byteLength));
}

function draco(bytes) {
  const buf = new dm.DecoderBuffer();
  buf.Init(new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength), bytes.byteLength);
  const dec = new dm.Decoder();
  const pc = new dm.PointCloud();
  const st = dec.DecodeBufferToPointCloud(buf, pc);
  if (!st.ok()) throw new Error(st.error_msg());
  const attr = dec.GetAttribute(pc, 0);
  const n = pc.num_points() * attr.num_components();
  const ptr = dm._malloc(n * 4);
  dec.GetAttributeDataArrayForAllPoints(pc, attr, dm.DT_FLOAT32, n * 4, ptr);
  const out = new Float32Array(dm.HEAPF32.buffer, ptr, n).slice();
  dm._free(ptr); dm.destroy(pc); dm.destroy(dec); dm.destroy(buf);
  return out;
}

async function ours(files, meta) {
  const t0 = performance.now();
  const [bz, hz, lz] = files;
  const [bodyB, headB, lensB] = await Promise.all([blosc.decode(bz), blosc.decode(hz), blosc.decode(lz)]);
  const t1 = performance.now();
  const body = meta.body_dtype === "<i1" ? new Int8Array(bodyB.buffer, bodyB.byteOffset, bodyB.byteLength)
                                          : new Int16Array(bodyB.buffer, bodyB.byteOffset, bodyB.byteLength / 2);
  const head = new Int16Array(headB.buffer, headB.byteOffset, headB.byteLength / 2);
  const lens = new Uint16Array(lensB.buffer, lensB.byteOffset, lensB.byteLength / 2);
  const q = meta.quantum, [ox, oy, oz] = meta.origin;
  const out = new Float32Array(meta.vertices * 3);
  let b = 0, h = 0, o = 0;
  for (let l = 0; l < lens.length; l++) {
    const n = lens[l];
    let x = head[h++], y = head[h++], z = head[h++];
    out[o++] = ox + x * q; out[o++] = oy + y * q; out[o++] = oz + z * q;
    if (n < 2) continue;
    let dx = head[h++], dy = head[h++], dz = head[h++];
    x += dx; y += dy; z += dz;
    out[o++] = ox + x * q; out[o++] = oy + y * q; out[o++] = oz + z * q;
    for (let i = 2; i < n; i++) {
      dx += body[b++]; dy += body[b++]; dz += body[b++];
      x += dx; y += dy; z += dz;
      out[o++] = ox + x * q; out[o++] = oy + y * q; out[o++] = oz + z * q;
    }
  }
  const t2 = performance.now();
  return { out, decompress: t1 - t0, reconstruct: t2 - t1 };
}

const dir = new URL(".", import.meta.url).pathname;
const results = [];
for (const bits of [14, 12, 11]) {
  const meta = JSON.parse(readFileSync(`${dir}/ours_${bits}.json`, "utf8"));
  const truth = readNpyF32(`${dir}/truth_${bits}.npy`);
  const drc = new Uint8Array(readFileSync(`${dir}/draco_pos_${bits}.drc`));
  const files = ["body", "head", "lens"].map((n) => new Uint8Array(readFileSync(`${dir}/ours_${bits}_${n}.blosc`)));   // read outside the timer, as drc
  const dT = [], oT = [];
  let dOut, oRes;
  for (let r = 0; r < 5; r++) {
    const t0 = performance.now(); dOut = draco(drc); dT.push(performance.now() - t0);
    oRes = await ours(files, meta); oT.push(oRes);
  }
  let oDiff = 0, dDiff = 0;
  for (let i = 0; i < truth.length; i++) { oDiff = Math.max(oDiff, Math.abs(oRes.out[i] - truth[i])); }
  const comparable = dOut.length === truth.length;
  if (comparable) for (let i = 0; i < truth.length; i++) dDiff = Math.max(dDiff, Math.abs(dOut[i] - truth[i]));
  const min = (a) => Math.min(...a);
  const r = { bits, vertices: meta.vertices,
    draco: { ms: +min(dT).toFixed(0), first_ms: +dT[0].toFixed(0), values: dOut.length, max_diff_from_grid_mm: comparable ? +dDiff.toExponential(2) : null },
    predictive: { ms: +min(oT.map((t) => t.decompress + t.reconstruct)).toFixed(0), decompress_ms: +min(oT.map((t) => t.decompress)).toFixed(0),
                  reconstruct_ms: +min(oT.map((t) => t.reconstruct)).toFixed(0), max_diff_from_grid_mm: +oDiff.toExponential(2) } };
  r.draco.mvert_per_s = +(meta.vertices / r.draco.ms / 1e3).toFixed(1);
  r.predictive.mvert_per_s = +(meta.vertices / r.predictive.ms / 1e3).toFixed(1);
  results.push(r);
  console.log(JSON.stringify(r));
}
if (process.argv[2]) writeFileSync(process.argv[2], JSON.stringify({ node: process.version, results }, null, 1));
