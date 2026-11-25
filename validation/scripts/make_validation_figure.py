"""Validation figure: the frozen cohort-1 models on cohort 2, and how stable the results are across runs.
Reads only the tables in this repository. Run from the repository root:  python validation/scripts/make_validation_figure.py"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, ORANGE, GREEN, CREAM, GREY = "#0F1B33", "#D9481F", "#2F7D5B", "#F6F2EA", "#8A8F9C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GREY, "axes.spines.top": False, "axes.spines.right": False})

def rows(p): return list(csv.DictReader(open(p), delimiter="\t"))

def km(t, e):
    order = np.argsort(t); t, e = t[order], e[order]
    xs, ys, s, n = [0.0], [1.0], 1.0, len(t)
    for ti in np.unique(t):
        at, d = (t >= ti).sum(), ((t == ti) & (e == 1)).sum()
        if d: s *= 1 - d / at; xs += [ti, ti]; ys += [ys[-1], s]
    xs.append(t.max()); ys.append(s)
    return np.array(xs), np.array(ys)

def roc(y, p):
    th = np.unique(p)[::-1]; fpr, tpr = [0.0], [0.0]
    for c in th:
        call = p >= c; tpr.append((call & (y == 1)).sum() / (y == 1).sum()); fpr.append((call & (y == 0)).sum() / (y == 0).sum())
    return np.array(fpr), np.array(tpr)

fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))

# A: frozen survival model on cohort 2
r = rows("validation/frozen_survival_model_second_cohort_predictions.tsv")
t = np.array([float(x["years"]) for x in r]); e = np.array([int(x["died"]) for x in r]); g = np.array([x["group"] for x in r])
for grp, col, lab in (("high", ORANGE, "Predicted high risk"), ("low", NAVY, "Predicted low risk")):
    m = g == grp; x, y = km(t[m], e[m])
    ax[0].plot(x, y, color=col, lw=2, label=f"{lab} (n = {m.sum()}, {e[m].sum()} deaths)")
ax[0].set(ylim=(0, 1.02), xlabel="Years after diagnosis", ylabel="Share of patients alive", title="A. Survival model, frozen, on 200 other patients")
ax[0].text(0.03, 0.04, "C-index 0.64 (0.55 to 0.72)\nlog-rank p = 0.008", transform=ax[0].transAxes)
ax[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.14), fontsize=8)

# B: stage model ROC, test set of cohort 1 and frozen on cohort 2
r1 = rows("results/stage/test_predictions.tsv")
y1 = np.array([x["truth"] == "Stage IV" for x in r1], int); p1 = np.array([float(x["probability_of_positive"]) for x in r1])
r2 = rows("validation/frozen_stage_model_second_cohort_predictions.tsv")
y2 = np.array([x["true_stage"] == "Stage IV" for x in r2], int); p2 = np.array([float(x["probability_stage_iv"]) for x in r2])
for y, p, col, lab in ((y1, p1, NAVY, "Cohort 1 test set (29), AUC 0.88"), (y2, p2, ORANGE, "Cohort 2, frozen model (119), AUC 0.82")):
    f, tp = roc(y, p); ax[1].step(f, tp, where="post", color=col, lw=2, label=lab)
ax[1].plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
ax[1].set(xlabel="False positive rate", ylabel="True positive rate", title="B. Stage IV vs Stage I: ROC curves")
ax[1].legend(frameon=False, loc="lower right", fontsize=8)

# C: scores across runs with intervals
lab = ["Survival\nmain", "Survival\n100 genes", "Survival\nfrozen, cohort 2", "Survival\nretrained, cohort 2",
       "Stage\nmain", "Stage\n50 genes", "Stage\nfrozen, cohort 2", "Stage\nretrained, cohort 2"]
val = [0.659, 0.622, 0.638, 0.463, 0.884, 0.819, 0.817, 0.946]
lo = [0.53, 0.48, 0.55, 0.27, 0.72, 0.62, 0.61, 0.81]; hi = [0.78, 0.75, 0.72, 0.72, 1.00, 0.97, 0.97, 1.00]
cols = [NAVY, NAVY, ORANGE, GREY, GREEN, GREEN, ORANGE, GREY]
yy = np.arange(len(val))[::-1]
for i in range(len(val)):
    ax[2].plot([lo[i], hi[i]], [yy[i]] * 2, color=cols[i], lw=2); ax[2].plot(val[i], yy[i], "o", color=cols[i], ms=7)
ax[2].axvline(0.5, ls="--", color=GREY, lw=1); ax[2].text(0.49, -0.6, "guessing", color=GREY, fontsize=8, ha="right")
ax[2].set_ylim(-0.9, 7.5); ax[2].set_yticks(yy); ax[2].set_yticklabels(lab, fontsize=7.5); ax[2].set_xlim(0.2, 1.02)
ax[2].set(xlabel="Test score with 95% interval\n(C-index for survival, ROC AUC for stage)", title="C. Every run, on patients not used for training")

# D: genes chosen in every fold in both cohorts (stage)
s1 = {x["feature"] for x in rows("results/stage/feature_stability.tsv") if x["chosen_in_folds"] == x["of_folds"]}
s2 = {x["feature"] for x in rows("validation/second-cohort-retrained/stage_feature_stability.tsv") if x["chosen_in_folds"] == x["of_folds"]}
both = sorted(s1 & s2)
prolif = {"AURKB", "CDC20", "CDCA3", "CEP55", "HJURP", "KIF18B", "KIF2C", "MYBL2", "SKA1", "UBE2C"}
immune = {"IFNG", "LTA"}
ax[3].barh([0, 1, 2], [len(s1 - s2), len(both), len(s2 - s1)], color=[NAVY, GREEN, ORANGE])
ax[3].set_yticks([0, 1, 2]); ax[3].set_yticklabels(["Cohort 1 only", "Both cohorts", "Cohort 2 only"])
ax[3].set(xlabel="Genes chosen in every cross-validation fold", title="D. Stage genes found again in cohort 2")
txt = "Both cohorts:\n" + "\n".join(", ".join(both[i:i + 4]) for i in range(0, len(both), 4))
txt += f"\n\n{len(prolif & set(both))} cell-division genes,\n{len(immune & set(both))} immune genes (IFNG, LTA)"
ax[3].set_xlim(0, 105); ax[3].set_xticks([0, 10, 20, 30, 40])
ax[3].text(0.98, 0.98, txt, transform=ax[3].transAxes, ha="right", va="top", fontsize=7.5, bbox=dict(fc=CREAM, ec=GREY, lw=0.5))
ax[3].invert_yaxis()

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"validation/validation-second-cohort-and-robustness.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
print("written")
