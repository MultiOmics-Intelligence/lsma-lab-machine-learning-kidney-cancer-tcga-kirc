# Second TCGA-KIRC cohort: 200 primary tumours from patients NOT in the first import, fetched with the lab's own
# TCGA importer (same files, same counts column, same sample sheet).
import json, sys
from pathlib import Path
from backend.services import tcga
recs = tcga.files("TCGA-KIRC", "rna")
first = tcga.select(recs, 200, ["Primary Tumor"])
used = {r["patient"] for r in first}
tum = [r for r in recs if r.get("sample_type") == "Primary Tumor"]
rest = sorted({r["patient"] for r in tum} - used)
print("tumour patients", len({r['patient'] for r in tum}), "first", len(used), "rest", len(rest))
out = Path("/data/scratch/cohort2"); out.mkdir(exist_ok=True)
res = tcga.fetch_tcga("TCGA-KIRC", out, {"data_type": "rna", "per_group": 200, "sample_types": ["Primary Tumor"], "patients": rest})
import pandas as pd
sh = pd.read_csv(res["files"][1]["path"], sep="\t")
print("overlap with first cohort:", len(set(sh["patient"]) & used), "n", len(sh))
print(sh["patient"].str[5:7].value_counts().to_dict())
print(json.dumps([str(f["path"]) for f in res["files"]]))
