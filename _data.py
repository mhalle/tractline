"""Where the tractography benches keep data: OUTSIDE the repo and outside Dropbox.

medseg lives in Dropbox, so anything written next to these scripts syncs (one run's captured
field alone is ~1.4 GB). TractCloud's checkout, its trained model and test data, the captured
fields and every derived cache go to $TRACTOGRAPHY_DATA (default ~/tmp/data/tractography).
Not backed up; everything there regenerates.
"""
import os
from pathlib import Path

DATA = Path(os.environ.get("TRACTOGRAPHY_DATA", Path.home() / "tmp/data/tractography"))
DATA.mkdir(parents=True, exist_ok=True)
