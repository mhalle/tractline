"""Where the tractography benches keep data: OUTSIDE the repo and outside Dropbox.

medseg lives in Dropbox, so anything written next to these scripts syncs (one run's captured
field alone is ~1.4 GB). TractCloud's checkout, its trained model and test data, the captured
fields and every derived cache go to $TRACTOGRAPHY_DATA (default ~/tmp/data/tractography).
Not backed up; everything there regenerates.

Layout under DATA (TractCloud's own model-cache layout, so TRACTCLOUD_DATA_DIR can point here):
    TractCloud/                          the upstream checkout (pip install -e'd into .venv)
    TrainedModel/                        TrainedModel.tar.gz, release v1.0.0
    TrainData_800clu800ol/HCP_mass_center.npy   the one file kept from TrainData_800clu800ol.tar.gz
    TestData/HCP/101006_ukf_pp_with_region.vtp  from TestData.tar.gz
    .venv/                               python 3.12: tractcloud, rankfield v0.3.10, matplotlib, sklearn
    hcp/                                 this experiment's outputs for the HCP subject
"""
import os
from pathlib import Path

DATA = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))
DATA.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TRACTCLOUD_DATA_DIR", str(DATA))

MODEL = DATA / "TrainedModel"
MASS_CENTER = DATA / "TrainData_800clu800ol" / "HCP_mass_center.npy"
HCP_VTP = DATA / "TestData" / "HCP" / "101006_ukf_pp_with_region.vtp"
HCP = DATA / "hcp"
HCP.mkdir(exist_ok=True)

# The ensemble: run r is seeded with SEEDS[r]. Fixed here so every stage agrees on it.
SEEDS = (0, 1, 2, 3, 4)
# The M0/M1 subsample of target streamlines, and the seed that draws it.
SUBSAMPLE = 100_000
SUBSAMPLE_SEED = 20261001
