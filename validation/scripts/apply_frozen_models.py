# Frozen-model test: the models trained in the lab on cohort 1 predict cohort 2 (other TCGA patients, mostly other
# hospitals) without any refitting. Feature values are built by the lab's own input stage on cohort 2.
import glob, json, sys
import joblib, numpy as np, pandas as pd
sys.path.insert(0, "/app/bioinformatics/python")
from lsma_bio.ml_common import c_index
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
S = json.load(open("/data/scratch/showcase_state.json"))
def find(run_id, stage, name):
    hits = glob.glob(f"/data/**/{run_id}/**/{stage}/objects/{name}", recursive=True)
    if not hits: raise SystemExit(f"not found {run_id} {stage} {name}")
    return hits[0]
out = {}
_t = pd.read_csv("/data/scratch/cohort2/TCGA-KIRC_rna_counts.tsv.gz", sep="\t")
_c = _t.iloc[:, 2:].to_numpy(dtype=float); _l = _c.sum(axis=0)
ALL = pd.DataFrame(np.log2(_c / _l * 1e6 + 1).T, index=_t.columns[2:], columns=_t["gene_name"].astype(str))
ALL = ALL.loc[:, ~ALL.columns.duplicated()]
def frame(d): return pd.DataFrame(d["x"], index=d["samples"], columns=d["features"])
for task, rk in (("survival", "surv"), ("classification", "stage")):
    fit = joblib.load(find(S["ml_rna"][f"{rk}_id"], "training", "model.joblib"))
    d1 = joblib.load(find(S["ml_rna"][f"{rk}_id"], "input", "dataset.joblib"))
    d2 = joblib.load(find(S["ml_c2"][f"{rk}_id"], "input", "dataset.joblib"))
    x2 = frame(d2).reindex(columns=d1["features"])
    miss = [f for f in x2.columns if x2[f].isna().all()]
    # genes below the expression filter in cohort 2: take their real log2(CPM + 1) from the raw counts
    for f in miss:
        if f in ALL.columns: x2[f] = ALL.loc[x2.index, f].to_numpy()
    missing = int(x2.isna().all().sum()); x2 = x2.fillna(0.0).to_numpy(dtype=np.float32)
    r = {"cohort2_samples": int(x2.shape[0]), "features_missing_in_cohort2": missing}
    if task == "survival":
        m = fit["model"]; used = d1["features"][m.cols_]
        r["features_missing_in_model"] = int(sum(f not in set(d2["features"]) for f in used))
        risk = m.risk(x2); t, e = d2["outcome"]["time"], d2["outcome"]["event"]
        r["c_index_cohort2"] = round(c_index(t, e, risk), 4)
        boot = []; rng = np.random.default_rng(1)
        for _ in range(1000):
            i = rng.integers(0, len(t), len(t)); boot.append(c_index(t[i], e[i], risk[i]))
        r["c_index_interval"] = [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]
        high = risk > fit["risk_cutoff"]
        lr = logrank_test(t[high], t[~high], e[high], e[~high]); r["logrank_p"] = float(lr.p_value)
        r["n_high"], r["n_low"], r["events"] = int(high.sum()), int((~high).sum()), int(e.sum())
        curves = {}
        for lab, mask in (("High risk", high), ("Low risk", ~high)):
            k = KaplanMeierFitter().fit(t[mask] / 365.25, e[mask]); sf = k.survival_function_
            curves[lab] = {"years": [float(v) for v in sf.index], "survival": [float(v) for v in sf.iloc[:, 0]]}
        r["km"] = curves
        pd.DataFrame({"patient": d2["samples"], "risk_score": risk, "group": np.where(high, "high", "low"), "years": t / 365.25, "died": e}).to_csv("/data/scratch/ml_apply_survival_cohort2.tsv", sep="\t", index=False)
    else:
        y2 = d2["outcome"]["y"]; p = fit["model"].predict_proba(x2)[:, 1]
        r["model"] = fit["model_key"]; r["auc_cohort2"] = round(float(roc_auc_score(y2, p)), 4)
        rng = np.random.default_rng(1); boot = []
        for _ in range(1000):
            i = rng.integers(0, len(y2), len(y2))
            if len(set(y2[i])) == 2: boot.append(roc_auc_score(y2[i], p[i]))
        r["auc_interval"] = [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]
        r["balanced_accuracy_0.5"] = round(float(balanced_accuracy_score(y2, p >= 0.5)), 4)
        r["n_stage_iv"], r["n_stage_i"] = int(y2.sum()), int((1 - y2).sum())
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y2, p); r["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        pd.DataFrame({"patient": d2["samples"], "probability_stage_iv": p, "true_stage": np.where(y2 == 1, "Stage IV", "Stage I")}).to_csv("/data/scratch/ml_apply_stage_cohort2.tsv", sep="\t", index=False)
    out[task] = r
json.dump(out, open("/data/scratch/ml_apply.json", "w"), indent=1)
print(json.dumps({k: {a: b for a, b in v.items() if a not in ("km", "roc")} for k, v in out.items()}, indent=1))
