import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))                            # phantom.py


# In CI the RapidParc weights are fetched first (ci.yml): their absence there is a failure, not a skip.
import os                                                                    # noqa: E402
if os.environ.get("CI"):
    from tractline.data import DATA                                          # noqa: E402
    if not (DATA / "RapidParc" / "rapidparc.safetensors").exists():
        raise RuntimeError(f"CI: RapidParc's weights are not in {DATA / 'RapidParc'} - the fetch step failed")
