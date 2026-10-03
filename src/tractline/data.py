"""Where tractline's data lives: outside the repository, in $TRACTOGRAPHY_DATA (default
~/tmp/data/tractography). Nothing here is created on import; whatever writes there creates what it needs.

Layout under DATA:
    RapidParc/                           the default labeler's weights (github.com/MedVisBonn/RapidParc release
                                         v1.0.0: rapidparc.safetensors, hemiaug.safetensors; sha256-checked on load)
    TractCloud/src/                      TractCloud's code (github.com/SlicerDMRI/TractCloud), for the optional
                                         TractCloud labeler
    TrainedModel/                        TractCloud's weights (its release v1.0.0, TrainedModel.tar.gz)
    TrainData_800clu800ol/               from its TrainData_800clu800ol.tar.gz: HCP_mass_center.npy (the labeler's
                                         centering); test.pickle (its labeled test split, bench/accuracy_tractcloud_test.py)
    TestData/HCP/101006_ukf_pp_with_region.vtp   from its TestData.tar.gz (bench/resample_check.py)
    ds001226/                            OpenNeuro ds001226 (BTC_preop, CC0) and what the bench derives from it
    ukf/hardi/                           the Stanford HARDI scan (dipy's stanford_hardi) for the tracker's checks
    hcp/                                 bench outputs for HCP subject 101006
    .venv/                               the bench's environment (tractline installed editable)
"""
import os
from pathlib import Path

DATA = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))

MODEL = DATA / "TrainedModel"
MASS_CENTER = DATA / "TrainData_800clu800ol" / "HCP_mass_center.npy"
HCP_VTP = DATA / "TestData" / "HCP" / "101006_ukf_pp_with_region.vtp"
HCP = DATA / "hcp"
