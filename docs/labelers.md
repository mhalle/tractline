# Labelers

The pipeline's last stage names the streamlines. A labeler is any object called as
`labeler(fibers, draws=(0,), logp=False)` that returns `labelers.base.Labels`; `pipeline.run(s, labeler, ...)`
takes one, and `labeler=None` means the default, `pipeline.default_labeler(device)`.

| labeler | module | what it is | default |
|---|---|---|---|
| RapidParc | `labelers.rapidparc.Labeler(device, model="rapidparc" or "hemiaug")` | its inference written here (~100 lines of torch), its released weights | **yes** |
| TractCloud | `labelers.tractcloud.Labeler(device, upstream=False)` | upstream's model code and weights; the resampling, the CPU network (`MatmulDGCNN`) and the context built here | no |

Both use the same 43-class scheme - the ORG atlas's 800 clusters and their 800 outlier twins (1,600
classes) mapped to 42 tracts and Other (class 42) - shipped as `labelers/scheme_43.json` (from RapidParc's
release files; identical to TractCloud's mapping, checked). Shared in `labelers/base.py`: `Labels`
(`keep`: the streamlines labeled, those of 40 mm or more; `length_mm`; `tract`; `logp`: the 1,600
cluster log-probabilities as float16, when asked for), the 40 mm cut, streamline lengths (per streamline:
a NaN or empty streamline affects only itself), and `LogMean` (several draws' probabilities averaged in
the log domain, without underflow).

`draws` are seeds (any iterable): TractCloud's context sample, RapidParc's shuffle. Several draws average
their cluster probabilities. A tractogram with no streamline of 40 mm or more returns empty Labels.

## RapidParc (the default)

von Bornhaupt, Bisten, ..., Schultz, "RapidParc: A Global-Context Transformer for Parallel, Accurate, and
Lesion-Robust Tractogram Parcellation", Imaging Neuroscience 2026; github.com/MedVisBonn/RapidParc (BSD-3).
Per draw: 15 points per streamline, evenly spaced by index; the set scaled to [-1, 1] per axis; shuffled
(torch.Generator seeded with the draw) and cut into groups of 2,000, each group re-scaled to [-1, 1]
inside the network; a linear embedding, an 8-layer transformer encoder (d_model 128, 1 head), a two-layer
classifier to 1,600 logits. Trained and run at the same context size. Weights (`rapidparc`, and `hemiaug`
trained with one-sided augmentation for lesions and surgery; 6.6 MB each) from its release v1.0.0 in
`$TRACTOGRAPHY_DATA/RapidParc`, sha256-checked when loaded.

Differences from its package, all deliberate: the 40 mm cut first (its package labels every streamline;
its training data were 40 mm or longer); the last group padded by repeating the shuffled streamlines as
often as needed (its package pads once and fails under 1,000 streamlines - the same rows at 1,000 or
more); probabilities and log-probabilities returned (its package returns only the argmax); TF32 left off
on CUDA (its package enables it); default draw 0 (its default seed is 42).

Checked: identical 1,600-cluster argmax to its package for both models and seeds 0 and 42, on 32 x86
cores and an A10G (`modal_rapidparc_check.py`) and on the M2's CPU; on the M2's GPU one streamline of
32,264 differs by cluster (hemiaug, seed 0; the other three runs identical), every tract identical
(`rapidparc_check.py`). On
TractCloud's labeled test split, 94.5 % tract accuracy / 93.2 % macro F1 - its paper's 94.44 / 93.2
(`accuracy_tractcloud_test.py`).

## TractCloud (optional)

Xue, Zhang, O'Donnell et al., MICCAI 2023; github.com/SlicerDMRI/TractCloud. Needs its code
(`$TRACTOGRAPHY_DATA/TractCloud/src`) and weights (`TrainedModel/`, `HCP_mass_center.npy`). By default it
runs with the context the released model was trained with (`trained_context`): each streamline, then its
19 nearest among one random set of 10,000 candidates from the whole tractogram, and 500 global
streamlines shared by the draw. `upstream=True` uses upstream's packaged inference context instead (80
global streamlines; a 10 % local subsample taken within consecutive ~10,000-streamline chunks, i.e.
slabs of the brain in seed order). On TractCloud's own test split (10,000 streamlines a subject, where
the two local contexts coincide): 92.0 % at the trained context (its paper: 92.12, as RapidParc's paper
tabulates it), 86.6 % at upstream's.

## Not built

The interface sketched before RapidParc was added - several label sets per labeler, "pair" schemes for
connectome labels, a declared input space with a registration stage - was not needed for the two
labelers above. DeepMultiConnectome (two heads of region pairs, 3,655 and 13,695 classes, trained in
MNI space on iFOD2 tractograms) would need it, a T1 -> MNI registration, and an accuracy study with
FreeSurfer parcellations (HCP has them, under its data-use agreement).
