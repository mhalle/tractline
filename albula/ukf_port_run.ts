// Runs Ron Kikinis's UKF port (albula-diffusion ukf.ts, commit ac1644e) on inputs written by
// albula_compare.py: UKF's own normalized signal, gradients, mask and seed points, so the tracker
// is the only thing that differs from the C++ program.
//   deno run -A --config bench/tractography/albula/deno.json bench/tractography/albula/ukf_port_run.ts DATA/albula [module]
import type { UkfData } from "../../../../../../../../../tmp/data/tractography/albula/src/ukf.ts";
// the module under test: the port as published, or a diagnostic copy beside it (second argument)
const { trackUkf } = await import(Deno.args[1] ?? "../../../../../../../../../tmp/data/tractography/albula/src/ukf.ts");

const dir = Deno.args[0];
const meta = JSON.parse(await Deno.readTextFile(`${dir}/meta.json`));
const f64 = async (n: string) => new Float64Array((await Deno.readFile(`${dir}/${n}`)).buffer);
const data: UkfData = {
  dims: meta.dims, voxel: meta.voxel, ijkToRAS: meta.ijkToRAS, G: meta.G,
  g: await f64("g.f64"), b: await f64("b.f64"),
  signal: new Float32Array((await Deno.readFile(`${dir}/signal.f32`)).buffer),
  mask: await Deno.readFile(`${dir}/mask.u8`),
};
const sp = await f64("seeds.f64");
const seeds: number[][] = [];
for (let i = 0; i < sp.length; i += 3) seeds.push([sp[i], sp[i + 1], sp[i + 2]]);
const res = trackUkf(data, seeds, meta.opts);
const n = res.fibers.length, offsets = new Uint32Array(n + 1), seedIdx = new Int32Array(n);
res.fibers.forEach((f, i) => { offsets[i + 1] = offsets[i] + f.points.length / 3; seedIdx[i] = f.seed; });
const pts = new Float32Array(offsets[n] * 3);
res.fibers.forEach((f, i) => pts.set(f.points, offsets[i] * 3));
await Deno.writeFile(`${dir}/port_offsets.u32`, new Uint8Array(offsets.buffer));
await Deno.writeFile(`${dir}/port_points.f32`, new Uint8Array(pts.buffer));
await Deno.writeFile(`${dir}/port_seed.i32`, new Uint8Array(seedIdx.buffer));
console.log(JSON.stringify({ fibers: n, seedsUsed: res.seedsUsed, seedsRejected: res.seedsRejected, steps: res.steps, ms: Math.round(res.ms) }));
