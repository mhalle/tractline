// TractCloud margins viewer (M4): streamlines from the web export, colored by tract, with the
// rankfield margin decoded in the browser driving opacity. One draw call per loaded chunk;
// per-streamline values live in a float texture read by gl_VertexID, so changing the threshold,
// softness or opacity is a uniform change - nothing is re-uploaded.
import PicoGL from "https://cdn.jsdelivr.net/npm/picogl@0.17.9/build/module/picogl.js";
import { parseField, ownGroupMargins } from "./decode.js";

const DATA = "data/";
const TEX_W = 2048;                       // per-line texture width (lines -> texels)
// 0.2-logit bins: past ~5 logits rankfield's byte levels are spaced more widely than 0.1, which
// left empty bins (0.16 apart at the clip)
const HIST = { lo: -2, hi: 8, bins: 50 };
const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("en-US");

const state = {
  showOther: true, mode: 1, which: 0, thr: 2, soft: 0.5, faded: 0.015, alpha: 0.35, invert: false, shown: 11,
  yaw: Math.PI, pitch: 0.12, dist: 430, pan: [0, 0, 0], view: "L",
};

// ---------- small matrix helpers (column-major) ----------
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}
function lookAt(eye, at, up) {
  let zx = eye[0] - at[0], zy = eye[1] - at[1], zz = eye[2] - at[2];
  let l = Math.hypot(zx, zy, zz); zx /= l; zy /= l; zz /= l;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
  l = Math.hypot(xx, xy, xz); xx /= l; xy /= l; xz /= l;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
    -(xx * eye[0] + xy * eye[1] + xz * eye[2]), -(yx * eye[0] + yy * eye[1] + yz * eye[2]), -(zx * eye[0] + zy * eye[1] + zz * eye[2]), 1]);
}
function mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}
function cameraBasis() {
  const { yaw, pitch } = state, cp = Math.cos(pitch), sp = Math.sin(pitch);
  const dir = [cp * Math.cos(yaw), cp * Math.sin(yaw), sp];
  const up = [-sp * Math.cos(yaw), -sp * Math.sin(yaw), cp];               // orthogonal to dir, defined at the poles
  const right = [up[1] * dir[2] - up[2] * dir[1], up[2] * dir[0] - up[0] * dir[2], up[0] * dir[1] - up[1] * dir[0]];   // screen right
  return { dir, up, right };
}
// RAS: +x right, +y anterior, +z superior. Each view looks FROM that side; S and I keep anterior up.
const VIEWS = { R: [0, 0], L: [Math.PI, 0], A: [Math.PI / 2, 0], P: [-Math.PI / 2, 0], S: [-Math.PI / 2, Math.PI / 2], I: [Math.PI / 2, -Math.PI / 2] };
const VIEW_NAMES = { R: "Right", L: "Left", A: "Anterior", P: "Posterior", S: "Superior", I: "Inferior" };

// "Other" is half the brain (unannotated clusters and every outlier class); Slicer's pink for it
// would paint the whole view, so it is drawn in a neutral gray, as context rather than a tract.
const OTHER_RGB = [107, 115, 133];
const tractRGB = (i) => (tables.tracts[i].name === "Other" ? OTHER_RGB : tables.tracts[i].color.map((v) => Math.round(v * 255)));

// ---------- GL ----------
const canvas = $("gl");
let app;
try {
  app = PicoGL.createApp(canvas, { antialias: true, alpha: false });
} catch (e) {
  $("subject").innerHTML = `<span class="error">This viewer needs WebGL 2, which this browser did not provide.</span>`;
  throw e;
}
app.clearColor(0.059, 0.09, 0.141, 1).noDepthTest().blend().blendFunc(PicoGL.SRC_ALPHA, PicoGL.ONE_MINUS_SRC_ALPHA);

const VS = `#version 300 es
layout(location=0) in vec3 aPos;
uniform mat4 uViewProj;
uniform vec3 uCenter;
uniform float uQuantum;
uniform int uPoints;
uniform highp sampler2D uLine;     // per streamline: group, best margin, mass margin, tract changes
uniform highp sampler2D uColors;   // per group: RGB
uniform int uMode, uWhich, uInvert, uHideGroup;
uniform float uThr, uSoft, uFaded, uAlpha;
out vec4 vColor;
void main() {
  gl_Position = uViewProj * vec4(uCenter + aPos * uQuantum, 1.0);
  int line = gl_VertexID / uPoints;
  vec4 d = texelFetch(uLine, ivec2(line % ${TEX_W}, line / ${TEX_W}), 0);
  vec3 c = texelFetch(uColors, ivec2(int(d.r), 0), 0).rgb;
  float a = uAlpha;
  if (uMode == 1) {
    float m = uWhich == 0 ? d.g : d.b;
    float s = smoothstep(uThr - uSoft, uThr + uSoft + 1e-4, m);
    if (uInvert == 1) s = 1.0 - s;
    a = mix(uFaded, uAlpha, s);
  }
  if (int(d.r) == uHideGroup) a = 0.0;
  vColor = vec4(c, a);
}`;
const FS = `#version 300 es
precision mediump float;
in vec4 vColor;
out vec4 fragColor;
void main() { fragColor = vColor; }`;
const program = app.createProgram(VS, FS);

let manifest, tables, levels, classToGroup, colorTex, indexBuf, P;
const chunks = [];                     // loaded chunks, in manifest order
let loading = 0;

async function fetchBuf(rel) {
  const r = await fetch(DATA + rel);
  if (!r.ok) throw new Error(`${rel}: HTTP ${r.status}`);
  return r.arrayBuffer();
}

async function init() {
  [manifest, tables] = await Promise.all([fetch(DATA + "manifest.json").then((r) => r.json()),
                                          fetch(DATA + "tables.json").then((r) => r.json())]);
  levels = new Float32Array(await fetchBuf(manifest.field.levels));
  classToGroup = Int32Array.from(tables.clusterToTract);
  P = manifest.pointsPerLine;
  $("subject").textContent = `${manifest.subject}. Field from run 0 of ${manifest.runs.count}, rankfield depth ${manifest.field.rankfield.depth}.`;
  $("chunks").max = manifest.chunkCount;

  const rgba = new Uint8Array(64 * 4);
  tables.tracts.forEach((t, i) => { rgba.set([...tractRGB(i), 255], i * 4); });
  colorTex = app.createTexture2D(rgba, 64, 1, { internalFormat: PicoGL.RGBA8, minFilter: PicoGL.NEAREST, magFilter: PicoGL.NEAREST });

  const maxLines = Math.max(...manifest.chunks.map((c) => c.lines));
  const idx = new Uint32Array(maxLines * (P - 1) * 2);
  for (let l = 0, k = 0; l < maxLines; l++) for (let s = 0; s < P - 1; s++) { idx[k++] = l * P + s; idx[k++] = l * P + s + 1; }
  indexBuf = app.createIndexBuffer(PicoGL.UNSIGNED_INT, 2, idx);

  buildLegend();
  setView("L");
  bindControls();
  resize();
  await ensureLoaded();
}

async function loadChunk(c) {
  const entry = manifest.chunks[c];
  const [g, f, e] = await Promise.all([fetchBuf(entry.geom.file), fetchBuf(entry.field.file), fetchBuf(entry.extra.file)]);
  const lines = new DataView(g).getUint32(0, true);
  const pos = new Int16Array(g, 8, lines * P * 3);
  const field = parseField(f);
  const t0 = performance.now();
  const dec = ownGroupMargins(field, classToGroup, levels, manifest.field.rankfield.clip);
  const decodeMs = performance.now() - t0;
  const changes = new Uint8Array(e, 8 + lines * 4, lines);

  const H = Math.ceil(lines / TEX_W);
  const tex = new Float32Array(TEX_W * H * 4);
  for (let i = 0; i < lines; i++) tex.set([dec.group[i], dec.best[i], dec.mass[i], changes[i]], i * 4);
  const lineTex = app.createTexture2D(tex, TEX_W, H, { internalFormat: PicoGL.RGBA32F, minFilter: PicoGL.NEAREST, magFilter: PicoGL.NEAREST,
                                                       wrapS: PicoGL.CLAMP_TO_EDGE, wrapT: PicoGL.CLAMP_TO_EDGE });
  const vao = app.createVertexArray()
    .vertexAttributeBuffer(0, app.createVertexBuffer(PicoGL.SHORT, 3, pos), { integer: false, normalized: false })
    .indexBuffer(indexBuf);
  const dc = app.createDrawCall(program, vao, PicoGL.LINES).drawRanges([0, lines * (P - 1) * 2])
    .texture("uLine", lineTex).texture("uColors", colorTex);
  chunks[c] = { lines, dc, group: dec.group, best: dec.best, mass: dec.mass, changes, decodeMs };
}

const inFlight = new Set();
async function ensureLoaded() {
  for (let c = 0; c < state.shown; c++) {
    if (chunks[c] || inFlight.has(c)) continue;      // a second caller skips what the first is fetching
    inFlight.add(c);
    loading++;
    updateStatus();
    try { await loadChunk(c); } catch (err) { $("st-load").textContent = `Could not load chunk ${c}: ${err.message}`; return; }
    finally { loading--; inFlight.delete(c); }
    refreshDerived();
    requestRender();
  }
  updateStatus();
}

function shownChunks() { return chunks.slice(0, state.shown).filter(Boolean); }

// ---------- rendering ----------
let renderPending = false, lastFrameMs = 0;
const syncPixel = new Uint8Array(4);
function requestRender() { if (!renderPending) { renderPending = true; requestAnimationFrame(render); } }
function render() {
  renderPending = false;
  const t0 = performance.now();
  const { dir, up } = cameraBasis();
  const target = state.pan;
  const aspect = canvas.width / canvas.height;
  const dist = state.dist * Math.max(1, 1.25 / aspect);           // a portrait screen: fit the width, not the height
  const eye = [target[0] + dir[0] * dist, target[1] + dir[1] * dist, target[2] + dir[2] * dist];
  // center the brain in the space beside the rail, not under it: shift clip x by the rail's share
  const rail = document.querySelector(".rail");
  const wide = window.innerWidth > 700;                             // otherwise the rail is a bottom sheet
  const side = wide ? (rail.offsetLeft + rail.offsetWidth) / window.innerWidth : 0;
  const lift = wide ? 0 : (window.innerHeight - rail.offsetTop) / window.innerHeight;
  const shift = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, side, lift, 0, 1]);
  const vp = mul(shift, mul(perspective(0.5, aspect, Math.max(1, dist - 400), dist + 400), lookAt(eye, target, up)));
  app.clear();
  const center = manifest.coordinates.center;
  for (const ch of shownChunks()) {
    ch.dc.uniform("uViewProj", vp).uniform("uCenter", center).uniform("uQuantum", manifest.coordinates.quantum)
      .uniform("uPoints", P).uniform("uMode", state.mode).uniform("uWhich", state.which).uniform("uInvert", state.invert ? 1 : 0)
      .uniform("uThr", state.thr).uniform("uSoft", state.soft).uniform("uFaded", state.faded).uniform("uAlpha", state.alpha)
      .uniform("uHideGroup", state.showOther ? -1 : tables.otherTract)
      .draw();
  }
  app.gl.readPixels(0, 0, 1, 1, app.gl.RGBA, app.gl.UNSIGNED_BYTE, syncPixel);   // blocks until the frame is drawn:
                                                                                  // finish() returned at once here
  lastFrameMs = performance.now() - t0;
  updateStatus();
}

// ---------- derived panels ----------
let histCounts = new Uint32Array(HIST.bins), totalShown = 0;
function refreshDerived() {
  const arrKey = state.which === 0 ? "best" : "mass";
  histCounts = new Uint32Array(HIST.bins);
  totalShown = 0;
  const perTract = new Uint32Array(tables.tracts.length);
  const w = (HIST.hi - HIST.lo) / HIST.bins;
  for (const ch of shownChunks()) {
    const m = ch[arrKey];
    totalShown += ch.lines;
    for (let i = 0; i < ch.lines; i++) {
      const b = Math.min(HIST.bins - 1, Math.max(0, Math.floor((m[i] - HIST.lo) / w)));
      histCounts[b]++;
      perTract[ch.group[i]]++;
    }
  }
  document.querySelectorAll(".legend .n").forEach((el) => { el.textContent = fmt.format(perTract[+el.dataset.t]); });
  drawHist();
  updateReadout();
}

function belowCount() {
  const arrKey = state.which === 0 ? "best" : "mass";
  let n = 0;
  for (const ch of shownChunks()) { const m = ch[arrKey]; for (let i = 0; i < ch.lines; i++) if (m[i] < state.thr) n++; }
  return n;
}

function drawHist() {
  const c = $("hist"), dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  if (c.width !== W * dpr) { c.width = W * dpr; c.height = H * dpr; }
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const css = getComputedStyle(document.documentElement);
  const max = Math.max(1, ...histCounts.slice(0, -1));    // the last bin holds everything at or past the clip
  const bw = W / HIST.bins;
  const xThr = ((state.thr - HIST.lo) / (HIST.hi - HIST.lo)) * W;
  // a filled step area, in two passes clipped at the threshold: bright where lines stay bright
  const area = () => {
    ctx.beginPath();
    ctx.moveTo(0, H);
    for (let b = 0; b < HIST.bins; b++) {
      const h = Math.min(1, Math.sqrt(histCounts[b] / max)) * (H - 4);   // sqrt: the low-margin tail stays visible
      ctx.lineTo(b * bw, H - h);
      ctx.lineTo((b + 1) * bw, H - h);
    }
    ctx.lineTo(W, H);
    ctx.closePath();
  };
  for (const [x0, x1, bright] of [[0, xThr, state.invert], [xThr, W, !state.invert]]) {
    ctx.save();
    ctx.beginPath(); ctx.rect(x0, 0, x1 - x0, H); ctx.clip();
    area();
    ctx.fillStyle = css.getPropertyValue(bright ? "--bar-below" : "--bar");
    ctx.fill();
    ctx.restore();
  }
  ctx.fillStyle = css.getPropertyValue("--ink");
  ctx.fillRect(Math.round(xThr) - 1, 0, 2, H);
  $("ax-lo").textContent = HIST.lo;
  $("ax-hi").textContent = `${HIST.hi}+`;
}

function updateReadout() {
  const n = belowCount();
  $("below").innerHTML = `<strong>${fmt.format(n)}</strong> of ${fmt.format(totalShown)} streamlines (${totalShown ? Math.round((100 * n) / totalShown) : 0}%) have a margin below ${state.thr.toFixed(2)} logits`;
}

function updateStatus() {
  const n = shownChunks().reduce((s, c) => s + c.lines, 0);
  $("st-lines").textContent = `${fmt.format(n)} streamlines`;
  $("st-frame").textContent = lastFrameMs ? `${lastFrameMs.toFixed(1)} ms per frame` : "";
  $("st-load").textContent = loading ? `Loading chunk ${chunks.filter(Boolean).length + 1} of ${state.shown}…` : "";
  $("orient").textContent = state.view ? `Viewed from ${VIEW_NAMES[state.view].toLowerCase()}` : "";
}

function buildLegend() {
  const legend = $("legend");
  for (const cat of tables.categories) {
    const tracts = tables.tracts.map((t, i) => ({ ...t, i })).filter((t) => t.category === cat);
    const block = document.createElement("div");
    block.innerHTML = `<div class="cat">${cat}</div><div class="tracts">${tracts.map((t) =>
      `<span class="tract" title="${t.fullName}"><span class="sw" style="background: rgb(${tractRGB(t.i).join(",")})"></span>${t.name} <span class="n" data-t="${t.i}"></span></span>`).join("")}</div>`;
    legend.appendChild(block);
  }
}

// ---------- controls ----------
function setView(v) {
  [state.yaw, state.pitch] = VIEWS[v];
  state.view = v;
  state.pan = [0, 0, 0];
  requestRender();
}
function bindControls() {
  const bindRange = (id, key, digits, after) => {
    const el = $(id), out = $(id + "-out");
    const sync = () => { state[key] = +el.value; out.textContent = (+el.value).toFixed(digits); after?.(); requestRender(); };
    el.addEventListener("input", sync);
    out.textContent = (+el.value).toFixed(digits);
    state[key] = +el.value;
  };
  bindRange("thr", "thr", 2, () => { drawHist(); updateReadout(); });
  bindRange("soft", "soft", 2);
  bindRange("faded", "faded", 3);
  bindRange("alpha", "alpha", 2);
  const chunksEl = $("chunks");
  const syncChunks = () => {
    state.shown = +chunksEl.value;
    const n = manifest.chunks.slice(0, state.shown).reduce((s, c) => s + c.lines, 0);
    $("chunks-out").textContent = fmt.format(n);
  };
  chunksEl.addEventListener("input", () => { syncChunks(); refreshDerived(); requestRender(); ensureLoaded(); });
  syncChunks();

  document.querySelectorAll("[data-mode]").forEach((b) => b.addEventListener("click", () => {
    state.mode = +b.dataset.mode;
    document.querySelectorAll("[data-mode]").forEach((x) => x.setAttribute("aria-pressed", x === b));
    $("margin-controls").hidden = state.mode !== 1;
    requestRender();
  }));
  document.querySelectorAll("[data-emph]").forEach((b) => b.addEventListener("click", () => {
    state.invert = b.dataset.emph === "1";
    document.querySelectorAll("[data-emph]").forEach((x) => x.setAttribute("aria-pressed", x === b));
    drawHist();
    requestRender();
  }));
  $("show-other").addEventListener("change", (e) => { state.showOther = e.target.checked; requestRender(); });
  $("which").addEventListener("change", (e) => { state.which = +e.target.value; refreshDerived(); requestRender(); });
  document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  window.addEventListener("keydown", (e) => {
    if (e.target.closest("input, select")) return;
    const v = e.key.toUpperCase();
    if (VIEWS[v]) setView(v);
  });

  // threshold by dragging on the histogram
  const hist = $("hist");
  const setFromHist = (e) => {
    const r = hist.getBoundingClientRect();
    const v = HIST.lo + ((e.clientX - r.left) / r.width) * (HIST.hi - HIST.lo);
    $("thr").value = Math.round(Math.min(HIST.hi, Math.max(HIST.lo, v)) * 20) / 20;
    $("thr").dispatchEvent(new Event("input"));
  };
  hist.addEventListener("pointerdown", (e) => { hist.setPointerCapture(e.pointerId); setFromHist(e); });
  hist.addEventListener("pointermove", (e) => { if (hist.hasPointerCapture(e.pointerId)) setFromHist(e); });

  // orbit: drag rotates, shift-drag or right-drag pans, wheel zooms
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => { canvas.setPointerCapture(e.pointerId); drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey || e.button === 2 }; });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    if (drag.pan) {
      const { up, right } = cameraBasis(), k = state.dist * 0.0012;
      state.pan = state.pan.map((p, i) => p - right[i] * dx * k + up[i] * dy * k);
    } else {
      state.yaw -= dx * 0.006;
      state.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, state.pitch + dy * 0.006));
      state.view = null;
    }
    requestRender();
  });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); state.dist = Math.min(1500, Math.max(60, state.dist * Math.exp(e.deltaY * 0.001))); requestRender(); }, { passive: false });
  window.addEventListener("resize", resize);
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  app.resize(Math.round(window.innerWidth * dpr), Math.round(window.innerHeight * dpr));
  drawHist();
  requestRender();
}

// ---------- measurement hook (M4 exit test) ----------
// viewer.benchmark(): sweep the threshold through `steps` values, rendering each frame
// synchronously, and report the frame times - the cost of a live threshold change.
window.viewer = {
  state, chunks,
  async ready() { while (loading || shownChunks().length < state.shown) await new Promise((r) => setTimeout(r, 50)); },
  benchmark(steps = 60) {
    const times = [];
    const saved = state.thr;
    for (let k = 0; k < steps; k++) {
      state.thr = -1 + (7 * k) / (steps - 1);
      const t0 = performance.now();
      render();
      times.push(performance.now() - t0);
    }
    state.thr = saved;
    requestRender();
    times.sort((a, b) => a - b);
    const lines = shownChunks().reduce((s, c) => s + c.lines, 0);
    const decodeMs = shownChunks().reduce((s, c) => s + c.decodeMs, 0);
    return { lines, steps, medianMs: +times[steps >> 1].toFixed(2), p95Ms: +times[Math.floor(steps * 0.95)].toFixed(2),
             maxMs: +times[steps - 1].toFixed(2), decodeMsTotal: +decodeMs.toFixed(1), canvas: [canvas.width, canvas.height] };
  },
};

init().catch((err) => { $("subject").innerHTML = `<span class="error">Could not start: ${err.message}</span>`; console.error(err); });
