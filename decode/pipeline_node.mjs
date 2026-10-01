// The CPU half of render/render.js in Node, as a cross-check outside the browser: file bytes ->
// positions ready to upload, for TRAKO (.tko JSON -> base64 -> Draco WASM) and the predictive
// encoding (blosc WASM -> two running sums -> int16 grid coordinates). No GPU stages.
//     cd DATA/decode && cp <repo>/bench/tractography/decode/pipeline_node.mjs . && node pipeline_node.mjs [out.json]
import { readFileSync, writeFileSync } from "node:fs";
import draco3d from "draco3d";
import { Blosc } from "numcodecs";

const dm = await draco3d.createDecoderModule({});
const blosc = new Blosc();
const now = () => performance.now();

function dracoDecode(bytes) {
  const buf = new dm.DecoderBuffer(); buf.Init(bytes, bytes.byteLength);
  const dec = new dm.Decoder(), pc = new dm.PointCloud();
  const st = dec.DecodeBufferToPointCloud(buf, pc);
  if (!st.ok()) throw new Error(st.error_msg());
  const attr = dec.GetAttribute(pc, 0), n = pc.num_points() * attr.num_components();
  const ptr = dm._malloc(n * 4);
  dec.GetAttributeDataArrayForAllPoints(pc, attr, dm.DT_FLOAT32, n * 4, ptr);
  const out = new Float32Array(dm.HEAPF32.buffer, ptr, n).slice();
  dm._free(ptr); dm.destroy(pc); dm.destroy(dec); dm.destroy(buf);
  return out;
}

function reconstructGrid(lens, head, body, out) {
  let b = 0, h = 0, o = 0;
  for (let l = 0; l < lens.length; l++) {
    const n = lens[l];
    let x = head[h++], y = head[h++], z = head[h++];
    out[o++] = x; out[o++] = y; out[o++] = z;
    if (n < 2) continue;
    let dx = head[h++], dy = head[h++], dz = head[h++];
    x += dx; y += dy; z += dz; out[o++] = x; out[o++] = y; out[o++] = z;
    for (let i = 2; i < n; i++) {
      dx += body[b++]; dy += body[b++]; dz += body[b++];
      x += dx; y += dy; z += dz; out[o++] = x; out[o++] = y; out[o++] = z;
    }
  }
}
const typed = (u8, T) => new T(u8.buffer, u8.byteOffset, u8.byteLength / T.BYTES_PER_ELEMENT);

const results = [];
for (const bits of [14, 12]) {
  const meta = JSON.parse(readFileSync(`render_${bits}.json`, "utf8"));
  const om = JSON.parse(readFileSync(`ours_${bits}.json`, "utf8"));
  const tkoText = readFileSync(`hcp_${bits}.tko`, "utf8");
  const blobs = ["lens", "head", "body"].map((n) => new Uint8Array(readFileSync(`ours_${bits}_${n}.blosc`)));
  for (let r = 0; r < 3; r++) {
    const t = { bits, rep: r };
    let a = now();
    const g = JSON.parse(tkoText); t.trako_parse = now() - a;
    a = now();
    const bufs = g.buffers.map((b) => { const u = b.uri; return new Uint8Array(Buffer.from(u.slice(u.indexOf(",") + 1), "base64")); });
    t.trako_base64 = now() - a;
    const view = (acc) => { const bv = g.bufferViews[g.accessors[acc].bufferView]; return new Int8Array(bufs[bv.buffer].buffer, bufs[bv.buffer].byteOffset + (bv.byteOffset || 0), bv.byteLength); };
    const prim = g.meshes[0].primitives[0];
    a = now(); dracoDecode(view(prim.attributes.POSITION)); dracoDecode(view(prim.indices)); t.trako_decode = now() - a;
    t.trako_total = t.trako_parse + t.trako_base64 + t.trako_decode;
    a = now();
    const [lb, hb, bb] = await Promise.all(blobs.map((b) => blosc.decode(b)));
    t.ours_decompress = now() - a;
    a = now();
    reconstructGrid(typed(lb, Uint16Array), typed(hb, Int16Array), om.body_dtype === "<i1" ? typed(bb, Int8Array) : typed(bb, Int16Array), new Int16Array(meta.vertices * 3));
    t.ours_reconstruct = now() - a;
    t.ours_total = t.ours_decompress + t.ours_reconstruct;
    for (const k of Object.keys(t)) if (typeof t[k] === "number" && k !== "bits" && k !== "rep") t[k] = Math.round(t[k]);
    results.push(t);
    console.log(JSON.stringify(t));
  }
}
if (process.argv[2]) writeFileSync(process.argv[2], JSON.stringify({ node: process.version, results }, null, 1));
