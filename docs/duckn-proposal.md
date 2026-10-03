# Proposal to duckn 2.0: the field a series is corrected by, and phase encoding carried from 1.x

**From:** tractline (its reader of duckn stores), 2026-10-03.
**For:** duckn convention 2.0 (draft, `docs/proposals/duckn-2.0.md`, revision 17) and dwmri 2.0
(`docs/proposals/dwi-extension-2.0.md`).
**Status:** a proposal; nothing here is adopted.

dwmri 2.0 already carries almost everything susceptibility correction needs: per series, the phase
encoding as a dimension and a polarity (§7), the total readout time (`acquisition`, as 1.0), the
gradients in a stated `frame` (§3); and between series, a shared frame of reference
(`world.reference`, duckn 2.0 §3.2), which is how a reversed-phase-encoding series acquired as a
differently placed slab is put onto the diffusion series' grid. Two things are missing: which series
together measure a field and which series it corrects (§1), and the phase encoding of 1.x files whose
layout is in fact known (§2).

---

## 1. The field a series is corrected by

### Why

Correction from a reversed pair (FSL topup's model; tractline's `susceptibility.estimate`) estimates
one off-resonance field from b0 images acquired with different phase encodings - usually the
diffusion series' own b0s and a short series with the opposite polarity - and applies it to the
diffusion series. A store holding both says nothing that pairs them. A reader can guess (one
frame of reference, opposite polarity along the same world direction), and the guess fails exactly
where it matters: several diffusion runs in one session, a pair acquired for a different series, a
dedicated field-map series, a second pair along another direction.

BIDS states the pairing with two sidecar fields (since BIDS 1.7): `B0FieldIdentifier` on every image
that contributes to measuring a field, and `B0FieldSource` on every image to be corrected by one
(the older `IntendedFor` points from the field map to the images, by path). dcm2niix cannot infer
them (the pairing is a protocol fact, not a DICOM one); BIDS curation tools and users write them.

### Fields

In the `acquisition` object (dwmri 1.0 §4.1, as 2.0 keeps it):

| Field | Type | Meaning | BIDS |
|---|---|---|---|
| `b0_field_identifier` | string, or array of strings | This series' images contribute to measuring the B0 field(s) so named. | `B0FieldIdentifier` |
| `b0_field_source` | string, or array of strings | This series is to be corrected by the field(s) so named. | `B0FieldSource` |

On the array, not the group: duckn 2.0 §12 keeps every array complete on its own, a string key
survives a store being split or merged, and BIDS keeps the fields per file. Names are compared
exactly, among the arrays a reader is given together (a store, or files named together); they are
not global identifiers.

### Rules

Reported by a reader of this extension, as dwmri 2.0 §4's are (the core does not refuse them):

- **An array that names a field states `acquisition.phase_encoding` and
  `acquisition.total_readout_time`**: a field cannot be estimated from it without both.
- **Arrays that name one field are measured in one frame**: their worlds carry the same prefixed
  `reference`, or transforms reaching one (duckn 2.0 §3.2). Otherwise their positions do not
  compare and a field estimated from them means nothing.
- **Their phase encodings are compared in the world**: an array's direction is its phase-encoding
  dimension's `step`, normalized, times the polarity (`+` along increasing index), with the worlds'
  axes matched as §3.2 matches them. A field needs at least two distinct directions; arrays that
  all encode one way are reported. (Slabs placed differently encode along nearly, not exactly,
  opposite directions - ds001226 has patients rotated 0.8 degrees - so the comparison takes a
  tolerance, which the extension should state; tractline uses the directions as given.)
- **The corrected series may name the field too**: its own b0s usually contribute (the common use
  of topup, and tractline's).
- **A source naming a field that no array in hand identifies** is reported by a reader that needs
  the field; it is not an error in the store, whose field may be elsewhere.

### Example

A diffusion series (its b0s contribute) and a reversed-phase b0 series, one frame of reference (the values
illustrative):

```json
"extensions": { "dwmri": { "version": "2.0", "b_value": 2800, "frame": [[1,0,0],[0,1,0],[0,0,1]],
  "acquisition": { "phase_encoding": { "dimension": 2, "polarity": "-" }, "total_readout_time": 0.0475,
                   "b0_field_identifier": "pepolar1", "b0_field_source": "pepolar1" } } }
```

```json
"extensions": { "dwmri": { "version": "2.0", "b_value": 0, "frame": [[1,0,0],[0,1,0],[0,0,1]],
  "acquisition": { "phase_encoding": { "dimension": 2, "polarity": "+" }, "total_readout_time": 0.0475,
                   "b0_field_identifier": "pepolar1" } } }
```

### Open: a home outside `dwmri`

The reversed series above is written as a diffusion series of b0s (b-value 0, zero gradients,
directionality `"none"`), which is honest but narrow: a field can be measured by an EPI series of any
kind (BIDS `fmap/*_epi`), by a dedicated field map (magnitude and phase difference, or a field map in
Hz), and can correct functional series as well as diffusion ones. `acquisition`, `phase_encoding` and
these two fields describe an MR acquisition, not a diffusion encoding; an `mr` extension (or a core
`acquisition` for MR intents) would be their natural home, with `dwmri` keeping only what is
diffusion's. Not needed for tractline's case; worth deciding before 2.0 is released, since moving
fields later is a major version.

---

## 2. Phase encoding from a 1.x block

### Now

`convention2_write` drops a 1.x `acquisition.phase_encoding_direction` (`"j-"`), because the letter
names an image axis in a layout the file does not fix (dwmri 2.0 §7).

### Proposed

Carry it when the file does fix the layout: when the 1.x array's Zarr `dimension_names` name its
spatial dimensions `i`, `j` and `k`, each exactly once, the letter names the dimension so named -
`phase_encoding.dimension` is that dimension's index in 2.0's `dimensions`, and `polarity` is `-` for a
trailing `-`, otherwise `+`. Otherwise the field is dropped with a finding, as now.

duckn's own NIfTI import writes these names, in either order (`["i", "j", "k", "diffusion"]` in dwmri
1.0's examples; `["k", "j", "i"]` from `nifti_convert`'s zip path, which reverses the order for Zarr and
says so), and neither reverses the index direction along a dimension. The rule holds only for writers
that keep a dimension's name where they keep its direction: a 1.x writer that flipped a dimension's
index order and kept its NIfTI name would make the carried polarity wrong. No duckn writer does; the
rule should say that the names are read as NIfTI's voxel axes, `+` along increasing index.

A converter from BIDS needs none of this: it knows which of its dimensions NIfTI's `i`, `j`, `k` became
and writes `{ "dimension", "polarity" }` directly, from `PhaseEncodingDirection`.

---

## 3. What tractline would read

| tractline's subject (`pipeline.run`) | duckn 2.0 |
|---|---|
| `dwi`, `affine`, `vox` | the diffusion array: `dimensions`, `origin`, `world` |
| `bval`, `bvec` | `dwmri`: `b_values`, `gradients` in world axes by `frame` (a missing `frame`: the gradients are unusable, and tractline would refuse the series) |
| `pe_axis`, `pe_sign` | `acquisition.phase_encoding` |
| `readout_s` | `acquisition.total_readout_time` |
| `b0s`, `pe_vectors` | the diffusion series' b0s and every array naming the field its `b0_field_source` names, resampled onto its grid through the shared `reference`; each one's direction in the world as §1 computes it |

---

## 4. Decisions asked

1. `acquisition.b0_field_identifier` and `acquisition.b0_field_source`, with BIDS's meaning (§1).
2. Whether MR acquisition fields move to a home outside `dwmri` before 2.0 is released (§1, Open).
3. Carrying a 1.x `phase_encoding_direction` when `dimension_names` state the layout (§2).
