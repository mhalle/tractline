// Time to display: bytes on the wire -> a drawn frame of the whole HCP tractogram (440,621
// streamlines, 21.6 M vertices), TRAKO against the predictive encoding, in the browser.
//
// TRAKO, as a glTF loader takes it: fetch the .tko (glTF JSON), parse it, decode its base64
// data-URI buffers (fetch(dataURI), as three.js' GLTFLoader does), Draco-decode positions and
// streamline lengths (Google's WASM decoder, main thread), upload float32 positions.
// Predictive: fetch the three blosc blobs, decompress (numcodecs' Blosc WASM), reconstruct with
// two running sums per streamline, upload; either float32 positions or int16 grid coordinates
// that the vertex shader dequantizes (half the GPU bytes). Progressive: 50 self-contained
// chunks fetched concurrently, each decoded and uploaded as it lands, first frame after chunk 0.
// Every pipeline ends with the same draw - one multiDrawArrays of LINE_STRIPs, no index buffer -
// and a 1-pixel readPixels, so a stage's time includes the GPU work it queued.
import Blosc from "/data/node_modules/numcodecs/dist/blosc.js";   // the index also pulls in fflate, unresolvable without a bundler

const log = document.getElementById("log");
const say = (s) => { log.textContent += "\n" + s; };
log.textContent = "";
const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: true });
const md = gl.getExtension("WEBGL_multi_draw");
if (!md) say("no WEBGL_multi_draw: drawing with one drawArrays per streamline would distort the timing");
const blosc = new Blosc();
const now = () => performance.now();
const px = new Uint8Array(4);
const sync = () => gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);

const t0Draco = now();
const draco = await DracoDecoderModule({ wasmBinary: await (await fetch("/data/draco/draco_decoder.wasm")).arrayBuffer() });
const dracoInitMs = now() - t0Draco;

function program(vs) {
  const fs = `#version 300 es
  precision mediump float; out vec4 o; void main() { o = vec4(0.62, 0.78, 0.86, 0.06); }`;
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.bindAttribLocation(p, 0, "a"); gl.linkProgram(p);
  return p;
}
// view from the left, orthographic: screen x = -y (anterior left), screen y = z
const VIEW = `uniform vec3 uC; uniform float uS;
  vec4 place(vec3 p) { vec3 d = (p - uC) * uS; return vec4(-d.y, d.z * 1.333, 0.0, 1.0); }`;
const progF = program(`#version 300 es
  layout(location=0) in vec3 a; ${VIEW} void main() { gl_Position = place(a); }`);
const progI = program(`#version 300 es
  layout(location=0) in vec3 a; uniform vec3 uO; uniform float uQ; ${VIEW} void main() { gl_Position = place(uO + a * uQ); }`);

function draw(prog, buf, type, firsts, counts, n, meta) {
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.043, 0.067, 0.106, 1); gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
  gl.useProgram(prog);
  const c = meta.origin.map((o) => o + meta.extent / 2);
  gl.uniform3fv(gl.getUniformLocation(prog, "uC"), c);
  gl.uniform1f(gl.getUniformLocation(prog, "uS"), 1.8 / meta.extent);
  if (prog === progI) { gl.uniform3fv(gl.getUniformLocation(prog, "uO"), meta.origin); gl.uniform1f(gl.getUniformLocation(prog, "uQ"), meta.quantum); }
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, type, false, 0, 0);
  md.multiDrawArraysWEBGL(gl.LINE_STRIP, firsts, 0, counts, 0, n);
  sync();
}

function layout(lens) {
  const n = lens.length, firsts = new Int32Array(n), counts = new Int32Array(n);
  let acc = 0;
  for (let i = 0; i < n; i++) { firsts[i] = acc; counts[i] = lens[i]; acc += lens[i]; }
  return { firsts, counts, n, vertices: acc };
}

function dracoDecode(bytes) {
  const buf = new draco.DecoderBuffer();
  buf.Init(bytes, bytes.byteLength);
  const dec = new draco.Decoder();
  const pc = new draco.PointCloud();
  const st = dec.DecodeBufferToPointCloud(buf, pc);
  if (!st.ok()) throw new Error(st.error_msg());
  const attr = dec.GetAttribute(pc, 0);
  const n = pc.num_points() * attr.num_components();
  const ptr = draco._malloc(n * 4);
  dec.GetAttributeDataArrayForAllPoints(pc, attr, draco.DT_FLOAT32, n * 4, ptr);
  const out = new Float32Array(draco.HEAPF32.buffer, ptr, n).slice();
  draco._free(ptr); draco.destroy(pc); draco.destroy(dec); draco.destroy(buf);
  return out;
}

async function trako(bits, meta) {
  const t = {}, T = now();
  let a = now();
  const text = await (await fetch(`/data/hcp_${bits}.tko`, { cache: "no-store" })).text(); t.fetch = now() - a;
  a = now(); const g = JSON.parse(text); t.parse = now() - a;
  a = now(); const bufs = await Promise.all(g.buffers.map((b) => fetch(b.uri).then((r) => r.arrayBuffer()))); t.base64 = now() - a;
  const view = (acc) => { const bv = g.bufferViews[g.accessors[acc].bufferView]; return new Int8Array(bufs[bv.buffer], bv.byteOffset || 0, bv.byteLength); };
  const prim = g.meshes[0].primitives[0];
  a = now(); const pos = dracoDecode(view(prim.attributes.POSITION)); const lensF = dracoDecode(view(prim.indices)); t.decode = now() - a;
  a = now(); const L = layout(Int32Array.from(lensF, Math.round)); t.layout = now() - a;
  if (L.vertices * 3 !== pos.length) throw new Error("TRAKO lengths do not match its positions");
  a = now(); const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW); sync(); t.upload = now() - a;
  a = now(); draw(progF, buf, gl.FLOAT, L.firsts, L.counts, L.n, meta); t.draw = now() - a;
  t.total = now() - T; t.gpuBytes = pos.byteLength;
  gl.deleteBuffer(buf);
  return t;
}

// residuals -> positions, two running sums per streamline; `out` float32 world mm or int16 grid.
// Two specialized loops, not one with a per-vertex callback: the callback cost 9x in Chromium.
function reconstruct(lens, head, body, out, grid, meta) {
  return grid ? reconstructGrid(lens, head, body, out) : reconstructWorld(lens, head, body, out, meta);
}
function reconstructGrid(lens, head, body, out) {
  let b = 0, h = 0, o = 0;
  for (let l = 0; l < lens.length; l++) {
    const n = lens[l];
    let x = head[h++], y = head[h++], z = head[h++];
    out[o++] = x; out[o++] = y; out[o++] = z;
    if (n < 2) continue;
    let dx = head[h++], dy = head[h++], dz = head[h++];
    x += dx; y += dy; z += dz;
    out[o++] = x; out[o++] = y; out[o++] = z;
    for (let i = 2; i < n; i++) {
      dx += body[b++]; dy += body[b++]; dz += body[b++];
      x += dx; y += dy; z += dz;
      out[o++] = x; out[o++] = y; out[o++] = z;
    }
  }
  return o;
}
function reconstructWorld(lens, head, body, out, meta) {
  const q = meta.quantum, ox = meta.origin[0], oy = meta.origin[1], oz = meta.origin[2];
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
  return o;
}

const typed = (u8, T) => new T(u8.buffer, u8.byteOffset, u8.byteLength / T.BYTES_PER_ELEMENT);

async function predictive(bits, meta, grid) {
  const t = {}, T = now();
  let a = now();
  const raw = await Promise.all(["lens", "head", "body"].map((n) => fetch(`/data/ours_${bits}_${n}.blosc`, { cache: "no-store" }).then((r) => r.arrayBuffer())));
  t.fetch = now() - a;
  a = now(); const [lb, hb, bb] = await Promise.all(raw.map((r) => blosc.decode(new Uint8Array(r)))); t.decode = now() - a;
  const lens = typed(lb, Uint16Array), head = typed(hb, Int16Array), body = meta.body_dtype === "<i1" ? typed(bb, Int8Array) : typed(bb, Int16Array);
  a = now();
  const out = grid ? new Int16Array(meta.vertices * 3) : new Float32Array(meta.vertices * 3);
  reconstruct(lens, head, body, out, grid, meta);
  const L = layout(lens);
  t.reconstruct = now() - a;
  a = now(); const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, out, gl.STATIC_DRAW); sync(); t.upload = now() - a;
  a = now(); draw(grid ? progI : progF, buf, grid ? gl.SHORT : gl.FLOAT, L.firsts, L.counts, L.n, meta); t.draw = now() - a;
  t.total = now() - T; t.gpuBytes = out.byteLength;
  gl.deleteBuffer(buf);
  return t;
}

async function progressive(bits, meta) {
  const t = {}, T = now();
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, meta.vertices * 3 * 2, gl.STATIC_DRAW);       // int16 grid coordinates
  const pending = Array.from({ length: meta.chunks }, (_, c) =>
    fetch(`/data/chunks_${bits}/c${String(c).padStart(2, "0")}.bin`, { cache: "no-store" }).then((r) => r.arrayBuffer()));
  const allLens = [];
  let v = 0;
  for (let c = 0; c < meta.chunks; c++) {
    const ab = await pending[c];
    const [lines, nl, nh, nb, item] = new Uint32Array(ab, 0, 5);
    const at = 20, u8 = new Uint8Array(ab);
    const [lb, hb, bb] = await Promise.all([blosc.decode(u8.subarray(at, at + nl)), blosc.decode(u8.subarray(at + nl, at + nl + nh)),
                                            blosc.decode(u8.subarray(at + nl + nh, at + nl + nh + nb))]);
    const lens = typed(lb, Uint16Array);
    const nv = lens.reduce((s, x) => s + x, 0);
    const out = new Int16Array(nv * 3);
    reconstruct(lens, typed(hb, Int16Array), item === 1 ? typed(bb, Int8Array) : typed(bb, Int16Array), out, true, meta);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferSubData(gl.ARRAY_BUFFER, v * 6, out);
    v += nv; allLens.push(lens);
    if (c === 0) {
      const L = layout(lens);
      draw(progI, buf, gl.SHORT, L.firsts, L.counts, L.n, meta);
      t.firstFrame = now() - T; t.firstFrameStreamlines = lines;
    }
  }
  const lensAll = new Uint16Array(allLens.reduce((s, x) => s + x.length, 0));
  allLens.reduce((o, x) => (lensAll.set(x, o), o + x.length), 0);
  const L = layout(lensAll);
  draw(progI, buf, gl.SHORT, L.firsts, L.counts, L.n, meta);
  t.total = now() - T; t.gpuBytes = meta.vertices * 6;
  gl.deleteBuffer(buf);
  return t;
}

const round = (o) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, typeof v === "number" ? Math.round(v * 10) / 10 : v]));
const results = { dracoInitMs: Math.round(dracoInitMs), renderer: (() => { const d = gl.getExtension("WEBGL_debug_renderer_info"); return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); })(), runs: [] };
window.results = results;
for (const bits of [14, 12]) {
  const meta = await (await fetch(`/data/render_${bits}.json`, { cache: "no-store" })).json();
  const om = await (await fetch(`/data/ours_${bits}.json`, { cache: "no-store" })).json();
  meta.extent = (2 ** bits - 1) * meta.quantum;                               // Draco's range: the largest bbox side
  meta.body_dtype = om.body_dtype;
  for (const [name, fn] of [["trako", () => trako(bits, meta)], ["predictive float32", () => predictive(bits, meta, false)],
                            ["predictive int16 grid", () => predictive(bits, meta, true)], ["predictive progressive", () => progressive(bits, meta)]]) {
    const reps = [];
    for (let r = 0; r < 3; r++) reps.push(await fn());
    const best = reps.reduce((m, x) => (x.total < m.total ? x : m));
    results.runs.push({ bits, pipeline: name, best: round(best), first: round(reps[0]) });
    say(`${bits} bits  ${name.padEnd(24)} ${JSON.stringify(round(best))}`);
  }
}
results.done = true;
say("done");
