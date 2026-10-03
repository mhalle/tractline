"""tractline: diffusion MRI to named white-matter tracts.

Susceptibility correction from a reversed phase-encoding pair (FSL topup's model, by Gauss-Newton),
UKF two-tensor tractography (the Slicer UKFTractography algorithm, with Metal and Triton kernels),
and tract labeling (RapidParc by default, or TractCloud), in torch; one pipeline from the scan to labeled streamlines
(`tractline.pipeline`). Each stage is checked against its original in `bench/`.
"""
__version__ = "0.1.0"
