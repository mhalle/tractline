"""A registration check for ensemble_endpoints.py: FastSurfer's label boundaries (cortex yellow, deep gray cyan,
ventricles magenta), carried from the T1 onto each subject's corrected mean b0 by the rigid T1-to-DWI transform, on three
axial slices.

    uv run --group dipy python bench/endpoints_overlay.py OUT.png
"""
import sys, numpy as np, nibabel as nib, nrrd
sys.path.insert(0, 'bench')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from tractline.data import DATA
FS = DATA / 'fastsurfer'
def labmap(s):
    lab, h = nrrd.read(str(FS / f'{s}_asegdkt.seg.nrrd'))
    lps = np.eye(4); lps[:3, :3], lps[:3, 3] = np.asarray(h['space directions'], float).T, h['space origin']
    return lab, np.diag([-1., -1, 1, 1]) @ lps
def onto(lab, laff, T, shape, aff, kk_list):   # labels onto the DWI grid at 4x in-plane
    ii, jj, kk = np.meshgrid(np.arange(0, shape[0], .25), np.arange(0, shape[1], .25), kk_list, indexing='ij')
    vox = np.stack([ii, jj, kk], -1).reshape(-1, 3)
    w = vox @ aff[:3, :3].T + aff[:3, 3]
    wt = w @ np.linalg.inv(T)[:3, :3].T + np.linalg.inv(T)[:3, 3]
    v = (wt - laff[:3, 3]) @ np.linalg.inv(laff[:3, :3]).T
    return map_coordinates(lab, v.T, order=0).reshape(ii.shape)
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
from _masivar import scan_files
for row, s in enumerate(['cIIIs04', 'PAT16']):
    if s == 'cIIIs04':
        d = DATA / 'ensemble/cIIIs04/retest/cIIIs04-2.1'; b0 = np.load(d / 'A_meanb0.npy')
        aff = nib.load(scan_files('cIIIs04', 's1Bx1', 1, None)['APP'][0][1]).affine
        T = np.load(FS / 'cIIIs04_t1_to_dwi.npz')['t1_to_dwi']
    else:
        z = np.load(DATA / 'ensemble/PAT16/report_smooth.npz'); T = np.load(DATA / 'ensemble/PAT16/tumor_dwi.npz')['t1_to_dwi']
        aff = z['affine']; from ensemble_pat16_viewer import mean_b0; b0 = mean_b0()
    lab, laff = labmap(s)
    Z = b0.shape[2]; kk_list = [int(Z * f) for f in (0.35, 0.5, 0.65)]
    L = onto(lab, laff, T, b0.shape, aff, kk_list)
    for c, k in enumerate(kk_list):
        ax = axs[row, c]
        img = map_coordinates(b0, np.stack(np.meshgrid(np.arange(0, b0.shape[0], .25), np.arange(0, b0.shape[1], .25), [k], indexing='ij'), -1).reshape(-1, 3).T, order=1).reshape(L.shape[:2])
        ax.imshow(img.T, cmap='gray', origin='lower')
        l = L[:, :, c]
        edge = (np.diff(l, axis=0, prepend=l[:1]) != 0) | (np.diff(l, axis=1, prepend=l[:, :1]) != 0)
        ctx = (l >= 1000); sub = np.isin(l, [10, 49, 11, 50, 12, 51, 4, 43])
        ov = np.zeros(l.shape + (4,)); ov[edge & ctx] = [1, .8, 0, .8]; ov[edge & sub] = [0, 1, 1, .9]; ov[edge & np.isin(l, [4, 43, 5, 44])] = [1, 0, 1, .9]
        ax.imshow(np.transpose(ov, (1, 0, 2)), origin='lower'); ax.set_title(f'{s} slice {k}'); ax.axis('off')
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=80)
