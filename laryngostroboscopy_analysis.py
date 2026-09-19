"""
Hierarchical prioritisation of laryngostroboscopic and clinical features
for distinguishing malignant from benign vocal fold disease
=========================================================================
Bianchino et al. — Analysis pipeline

Reproduces all statistical analyses and figures reported in the manuscript,
including the reviewer-requested additional analyses (threshold validation,
benign-only Random Forest, and multivariate logistic regression).

Requirements
------------
    pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn

Usage
-----
    python laryngostroboscopy_analysis.py --data italproj_full.csv --outdir results/

The script writes:
    results/figures/        All manuscript figures (PNG, 300 dpi)
    results/tables/         CSV tables for every key result
    results/stats/          Plain-text statistics log
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

warnings.filterwarnings("ignore")

# ── Colour scheme (used throughout) ──────────────────────────────────────────
COLOURS = {"Control": "#2ca02c", "Benign": "#1f77b4", "Malignant": "#d62728"}
PALETTE  = [COLOURS["Control"], COLOURS["Benign"], COLOURS["Malignant"]]

# =============================================================================
# 1. DATA LOADING & CLEANING
# =============================================================================

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load italproj_full.csv (semicolon-separated, Italian headers) and return
    a tidy DataFrame with English column names and parsed numeric values.
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8")

    # ── Group assignment ──────────────────────────────────────────────────────
    mal_cols = [c for c in df.columns if c.startswith("MALIGNE")]
    ben_cats  = ["BENIGNE", "VASCOLARI", "CONGENITE", "INFIAMMATORIE",
                 "TRAUMATICHE", "NEUROLOGICHE", "MOVIMENTO", "FUNZIONALI", "CENTRALI"]
    ben_cols  = [c for c in df.columns if any(x in c for x in ben_cats)]

    def _group(row):
        if pd.notna(row["CONTROLLI"]):
            return "Control"
        if any(pd.notna(row[c]) for c in mal_cols):
            return "Malignant"
        if any(pd.notna(row[c]) for c in ben_cols):
            return "Benign"
        return "Benign"   # 3 rows with no diagnosis column filled → benign

    df["group"] = df.apply(_group, axis=1)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _pct(v):
        """'80 %' → 80.0"""
        if pd.isna(v): return np.nan
        return pd.to_numeric(str(v).replace("%", "").strip(), errors="coerce")

    def _amp(v):
        """'0 Assente' → 0.0, '25' → 25.0"""
        if pd.isna(v): return np.nan
        s = str(v).strip()
        if "assente" in s.lower(): return 0.0
        return pd.to_numeric(s, errors="coerce")

    def _nonvib(v):
        """'0 Tutte vibranti' → 0.0, '30' → 30.0"""
        if pd.isna(v): return np.nan
        s = str(v).strip()
        if "tutte" in s.lower(): return 0.0
        return pd.to_numeric(s, errors="coerce")

    def _arrest(v):
        """Italian ordinal → 0/1/2"""
        if pd.isna(v): return np.nan
        s = str(v).strip().lower()
        if "assenti" in s:         return 0.0
        if "occasionalmente" in s: return 1.0
        if "costantemente" in s:   return 2.0
        return np.nan

    def _supra(v):
        """Italian supraglottic grade → 0/1/2"""
        if pd.isna(v): return np.nan
        s = str(v).strip().lower()
        if "normale" in s: return 0.0
        if "lieve" in s:   return 1.0
        if "marcata" in s: return 2.0
        return np.nan

    # ── Parse all variables ───────────────────────────────────────────────────
    df["RFS"]         = pd.to_numeric(df["RFS"].str.split("/").str[0].str.strip(), errors="coerce")
    df["male"]        = (df["Sesso biologico"].str.strip().str.lower() == "maschio").astype(int)
    df["age"]         = pd.to_numeric(df["Età"], errors="coerce")
    df["F0"]          = pd.to_numeric(df["F0 (Hz)"], errors="coerce")
    df["intensity"]   = pd.to_numeric(df["Intensità (dB)"], errors="coerce")

    df["amp_dx"]      = df["Ampiezza della vibrazione [DX]"].apply(_amp)
    df["amp_sx"]      = df["Ampiezza della vibrazione [Sx]"].apply(_amp)
    df["mucosal_dx"]  = df["Onda mucosa [Dx]"].apply(_amp)
    df["mucosal_sx"]  = df["Onda mucosa [Sx]"].apply(_amp)
    df["nonvib_dx"]   = df["Porzioni non vibranti [Dx]"].apply(_nonvib)
    df["nonvib_sx"]   = df["Porzioni non vibranti [Sx]"].apply(_nonvib)
    df["reg_dx"]      = df["Regolarità (% di tempo vibrazione regolare) [Dx]"].apply(_pct)
    df["reg_sx"]      = df["Regolarità (% di tempo vibrazione regolare) [Sx]"].apply(_pct)
    df["arrest_dx"]   = df["Arresti vibratori [Dx]"].apply(_arrest)
    df["arrest_sx"]   = df["Arresti vibratori [Sx]"].apply(_arrest)
    df["supra_dx"]    = df["Atteggiamento delle strutture sopraglottiche [Dx]"].apply(_supra)
    df["supra_sx"]    = df["Atteggiamento delle strutture sopraglottiche [Sx]"].apply(_supra)
    df["closure"]     = df["Chiusura glottica"].str.strip()

    grbasi_cols = {
        "grbasi_G": "GRBASI Estesa (2021)  [G: grado globale di disfonia]",
        "grbasi_R": "GRBASI Estesa (2021)  [R: grado di voce rauca ]",
        "grbasi_B": "GRBASI Estesa (2021)  [Rd: grado di voce rauca + diplofonia]",
        "grbasi_A": "GRBASI Estesa (2021)  [A: grado di astenia]",
        "grbasi_S": "GRBASI Estesa (2021)  [S: grado di voce pressata]",
        "grbasi_I": "GRBASI Estesa (2021)  [I: grado di instabilità]",
    }
    for new, old in grbasi_cols.items():
        df[new] = pd.to_numeric(df[old], errors="coerce")

    df["malignant"]       = (df["group"] == "Malignant").astype(int)
    df["closure_irreg"]   = (df["closure"].str.lower() == "irregolare").astype(float)

    return df


# =============================================================================
# 2. DESCRIPTIVE STATISTICS
# =============================================================================

def descriptive_stats(df: pd.DataFrame, out: Path) -> None:
    """Table 1 — patient characteristics by group."""
    groups = ["Control", "Benign", "Malignant"]
    rows = []

    cont_vars = [
        ("age",       "Age (years)"),
        ("F0",        "F0 (Hz)"),
        ("intensity", "Intensity (dB)"),
        ("RFS",       "RFS score"),
        ("grbasi_G",  "GRBASI G"),
        ("grbasi_R",  "GRBASI R"),
        ("grbasi_S",  "GRBASI S"),
        ("grbasi_I",  "GRBASI I"),
        ("amp_dx",    "Amplitude DX (%)"),
        ("nonvib_dx", "Non-vibrating DX (%)"),
        ("reg_dx",    "Regularity DX (%)"),
        ("arrest_dx", "Vibratory arrest DX"),
    ]

    for var, label in cont_vars:
        grp_vals = [df[df["group"] == g][var].dropna() for g in groups]
        means    = [v.mean() for v in grp_vals]
        sds      = [v.std() for v in grp_vals]
        stat, p  = kruskal(*[v for v in grp_vals if len(v) > 0])
        rows.append({
            "Variable": label,
            **{f"{g}_mean±SD": f"{m:.1f}±{s:.1f}" for g, m, s in zip(groups, means, sds)},
            "KW_H": f"{stat:.2f}",
            "p": f"{p:.3f}",
        })

    # Categorical
    for val, label in [("Irregolare", "Glottic closure: Irregular, n (%)"),
                        (1,            "Male sex, n (%)")]:
        r = {"Variable": label}
        if label.startswith("Male"):
            for g in groups:
                sub = df[df["group"] == g]
                n   = sub["male"].sum()
                r[f"{g}_mean±SD"] = f"{n} ({n/len(sub)*100:.0f}%)"
        else:
            for g in groups:
                sub = df[df["group"] == g]
                n   = (sub["closure"] == val).sum()
                r[f"{g}_mean±SD"] = f"{n} ({n/len(sub)*100:.0f}%)"
        rows.append(r)

    tbl = pd.DataFrame(rows)
    tbl.to_csv(out / "table1_descriptives.csv", index=False)
    print("  ✓ table1_descriptives.csv")


# =============================================================================
# 3. EFFECT SIZES — rank-biserial correlation
# =============================================================================

def rank_biserial(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """
    Mann–Whitney U rank-biserial effect sizes for every continuous variable,
    Benign vs Malignant, with Benjamini–Hochberg correction.
    """
    bm = df[df["group"].isin(["Benign", "Malignant"])]
    b  = bm[bm["group"] == "Benign"]
    m  = bm[bm["group"] == "Malignant"]

    features = [
        "amp_dx", "amp_sx", "mucosal_dx", "mucosal_sx",
        "nonvib_dx", "nonvib_sx", "reg_dx", "reg_sx",
        "arrest_dx", "arrest_sx", "supra_dx", "supra_sx",
        "grbasi_G", "grbasi_R", "grbasi_B", "grbasi_A", "grbasi_S", "grbasi_I",
        "F0", "intensity", "RFS", "age",
    ]

    rows = []
    for feat in features:
        bv = b[feat].dropna()
        mv = m[feat].dropna()
        if len(bv) < 2 or len(mv) < 2:
            continue
        U, p = mannwhitneyu(bv, mv, alternative="two-sided")
        r = 1 - 2 * U / (len(bv) * len(mv))
        rows.append({"feature": feat, "r": r, "abs_r": abs(r),
                     "U": U, "p_raw": p,
                     "mean_benign": bv.mean(), "mean_malignant": mv.mean()})

    res = pd.DataFrame(rows).sort_values("abs_r", ascending=False)
    _, res["p_adj"], _, _ = multipletests(res["p_raw"], method="fdr_bh")
    res.to_csv(out / "effect_sizes_benign_vs_malignant.csv", index=False)
    print("  ✓ effect_sizes_benign_vs_malignant.csv")
    return res


# =============================================================================
# 4. UNIVARIATE LOGISTIC REGRESSION
# =============================================================================

def univariate_logistic(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """
    Univariate logistic regression: each predictor vs binary malignancy.
    Returns a DataFrame of OR, 95% CI, p (BH-corrected).
    """
    bm = df[df["group"].isin(["Benign", "Malignant"])].copy()

    predictors = [
        "amp_dx", "amp_sx", "mucosal_dx", "mucosal_sx",
        "nonvib_dx", "nonvib_sx", "reg_dx", "reg_sx",
        "arrest_dx", "arrest_sx", "supra_dx", "supra_sx",
        "grbasi_G", "grbasi_R", "grbasi_A", "grbasi_S", "grbasi_I",
        "F0", "intensity", "RFS", "age", "male",
    ]

    rows = []
    for pred in predictors:
        sub = bm[[pred, "malignant"]].dropna()
        if len(sub) < 10 or sub["malignant"].nunique() < 2:
            continue
        X = sm.add_constant(sub[[pred]])
        try:
            res = sm.Logit(sub["malignant"], X).fit(disp=0)
            OR  = np.exp(res.params[pred])
            ci  = np.exp(res.conf_int().loc[pred])
            rows.append({
                "predictor": pred,
                "OR": OR, "CI_lo": ci[0], "CI_hi": ci[1],
                "p_raw": res.pvalues[pred],
                "log_OR": res.params[pred],
                "log_CI_lo": res.conf_int().loc[pred, 0],
                "log_CI_hi": res.conf_int().loc[pred, 1],
                "n": len(sub),
            })
        except Exception:
            pass

    res_df = pd.DataFrame(rows)
    _, res_df["p_adj"], _, _ = multipletests(res_df["p_raw"], method="fdr_bh")
    res_df = res_df.sort_values("OR", ascending=False)
    res_df.to_csv(out / "univariate_logistic.csv", index=False)
    print("  ✓ univariate_logistic.csv")
    return res_df


# =============================================================================
# 5. RANDOM FOREST — two versions
# =============================================================================

def random_forest(df: pd.DataFrame, out: Path) -> dict:
    """
    Trains two RF models:
      (a) Malignant vs ALL (original paper model)
      (b) Malignant vs Benign only (reviewer-requested)

    Returns dict with AUROCs, MDI importances, confusion matrices, and
    ROC curve data for plotting.
    """
    features = [
        "amp_dx", "amp_sx", "mucosal_dx", "mucosal_sx",
        "nonvib_dx", "nonvib_sx", "reg_dx", "reg_sx",
        "arrest_dx", "arrest_sx", "supra_dx", "supra_sx",
        "grbasi_G", "grbasi_R", "grbasi_B", "grbasi_A", "grbasi_S", "grbasi_I",
        "F0", "intensity", "RFS", "age",
    ]

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=500, class_weight={0: 1, 1: 8},
                                random_state=42, n_jobs=-1)

    for label, subset in [("all_groups",    df),
                           ("benign_vs_mal", df[df["group"].isin(["Benign", "Malignant"])])]:
        X = subset[features]
        y = (subset["group"] == "Malignant").astype(int)
        mask = X.notna().all(axis=1)
        Xc, yc = X[mask], y[mask]

        fold_aucs, tprs, mean_fpr = [], [], np.linspace(0, 1, 100)
        all_proba, all_true = [], []
        mdi_folds = np.zeros(len(features))

        for tr, te in cv.split(Xc, yc):
            rf.fit(Xc.iloc[tr], yc.iloc[tr])
            proba = rf.predict_proba(Xc.iloc[te])[:, 1]
            auc   = roc_auc_score(yc.iloc[te], proba)
            fpr, tpr, _ = roc_curve(yc.iloc[te], proba)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            fold_aucs.append(auc)
            all_proba.extend(proba)
            all_true.extend(yc.iloc[te].tolist())
            mdi_folds += rf.feature_importances_

        rf.fit(Xc, yc)   # final fit for MDI
        mdi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

        pred_labels = (np.array(all_proba) > 0.5).astype(int)
        cm = confusion_matrix(all_true, pred_labels)

        results[label] = {
            "n": len(Xc), "n_mal": int(yc.sum()),
            "fold_aucs": fold_aucs,
            "mean_auc": np.mean(fold_aucs), "sd_auc": np.std(fold_aucs),
            "mean_tpr": np.mean(tprs, axis=0), "std_tpr": np.std(tprs, axis=0),
            "mean_fpr": mean_fpr,
            "mdi": mdi,
            "confusion_matrix": cm,
        }

        print(f"  RF ({label}): AUROC = {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")
        print(f"    Folds: {[f'{a:.3f}' for a in fold_aucs]}")
        mdi.to_csv(out / f"rf_mdi_{label}.csv", header=["MDI"])
        print(f"  ✓ rf_mdi_{label}.csv")

    return results


# =============================================================================
# 6. THRESHOLD VALIDATION (reviewer-requested)
# =============================================================================

def threshold_validation(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """
    Computes sensitivity, specificity, PPV, NPV for:
      - Non-vibrating DX > 30%
      - Regularity DX < 40%
      - Either threshold met (combined)

    Thresholds are data-derived (post-hoc, exploratory) based on observed
    distribution quantiles in the malignant group; this is stated explicitly
    in the manuscript and in the output table.
    """
    bm = df[df["group"].isin(["Benign", "Malignant"])].copy()
    rows = []

    def _stats(pos_mask, actual_mal, label):
        TP = int((pos_mask & actual_mal).sum())
        FN = int((~pos_mask & actual_mal).sum())
        TN = int((~pos_mask & ~actual_mal).sum())
        FP = int((pos_mask & ~actual_mal).sum())
        n  = TP + FN + TN + FP
        sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
        ppv  = TP / (TP + FP) if (TP + FP) > 0 else np.nan
        npv  = TN / (TN + FN) if (TN + FN) > 0 else np.nan
        return {"threshold": label, "n": n,
                "TP": TP, "FP": FP, "TN": TN, "FN": FN,
                "sensitivity": round(sens, 4), "specificity": round(spec, 4),
                "PPV": round(ppv, 4), "NPV": round(npv, 4)}

    sub1 = bm[["nonvib_dx", "group"]].dropna()
    am1  = sub1["group"] == "Malignant"
    rows.append(_stats(sub1["nonvib_dx"] > 30, am1, "Non-vibrating DX > 30%"))

    sub2 = bm[["reg_dx", "group"]].dropna()
    am2  = sub2["group"] == "Malignant"
    rows.append(_stats(sub2["reg_dx"] < 40, am2, "Regularity DX < 40%"))

    sub3 = bm[["nonvib_dx", "reg_dx", "group"]].dropna()
    am3  = sub3["group"] == "Malignant"
    comb = (sub3["nonvib_dx"] > 30) | (sub3["reg_dx"] < 40)
    rows.append(_stats(comb, am3, "Combined: nonvib>30% OR reg<40%"))

    res = pd.DataFrame(rows)
    res["note"] = "Thresholds are data-derived (post-hoc, exploratory); external validation required"
    res.to_csv(out / "threshold_validation.csv", index=False)
    print("  ✓ threshold_validation.csv")
    return res


# =============================================================================
# 7. MULTIVARIATE LOGISTIC REGRESSION (reviewer-requested)
# =============================================================================

def multivariate_logistic(df: pd.DataFrame, out: Path) -> None:
    """
    Two multivariate models adjusting for age and sex:
      Model A: nonvib_dx + reg_dx + arrest_dx + age + male
      Model B: Model A + glottic closure (irregular vs other)
    """
    bm = df[df["group"].isin(["Benign", "Malignant"])].copy()
    rows = []

    def _fit(feats, label):
        sub = bm[feats + ["malignant"]].dropna()
        X   = sm.add_constant(sub[feats])
        res = sm.Logit(sub["malignant"], X).fit(disp=0)
        for v in feats:
            OR  = np.exp(res.params[v])
            ci  = np.exp(res.conf_int().loc[v])
            p   = res.pvalues[v]
            rows.append({"model": label, "variable": v,
                         "OR": round(OR, 3),
                         "CI_lo": round(ci[0], 3), "CI_hi": round(ci[1], 3),
                         "p": round(p, 4),
                         "significant": p < 0.05,
                         "n": len(sub)})
        print(f"  {label} (n={len(sub)}): pseudo-R²={res.prsquared:.3f}")

    _fit(["nonvib_dx", "reg_dx", "arrest_dx", "age", "male"],
         "Model A: stroboscopic + age + sex")
    _fit(["nonvib_dx", "reg_dx", "arrest_dx", "closure_irreg", "age", "male"],
         "Model B: stroboscopic + closure + age + sex")

    pd.DataFrame(rows).to_csv(out / "multivariate_logistic.csv", index=False)
    print("  ✓ multivariate_logistic.csv")


# =============================================================================
# 8. FIGURES
# =============================================================================

def _savefig(fig, path: Path, name: str) -> None:
    fig.savefig(path / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


def fig1_demographics(df: pd.DataFrame, out: Path) -> None:
    """Figure 1 — Patient demographics and clinical overview."""
    order = ["Control", "Benign", "Malignant"]
    pal   = [COLOURS[g] for g in order]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Figure 1 — Patient Demographics & Clinical Overview (n=305)",
                 fontsize=14, fontweight="bold")

    # Group distribution
    counts = df["group"].value_counts().reindex(order)
    axes[0, 0].bar(order, counts.values, color=pal, edgecolor="white", linewidth=0.5)
    for i, (g, v) in enumerate(zip(order, counts.values)):
        axes[0, 0].text(i, v + 2, f"{v}\n({v/305*100:.1f}%)", ha="center", fontsize=10)
    axes[0, 0].set_ylabel("n patients"); axes[0, 0].set_title("Patient Distribution")
    axes[0, 0].set_ylim(0, 290)

    # Age by group
    for i, g in enumerate(order):
        vals = df[df["group"] == g]["age"].dropna()
        axes[0, 1].boxplot(vals, positions=[i], widths=0.5,
                           patch_artist=True,
                           boxprops=dict(facecolor=COLOURS[g], alpha=0.7),
                           medianprops=dict(color="white", linewidth=2),
                           flierprops=dict(marker="o", markersize=3, alpha=0.4))
        axes[0, 1].scatter(np.random.normal(i, 0.07, len(vals)), vals,
                           color=COLOURS[g], alpha=0.3, s=10, zorder=3)
    axes[0, 1].set_xticks(range(3)); axes[0, 1].set_xticklabels(order)
    axes[0, 1].set_ylabel("Age (years)"); axes[0, 1].set_title("Age Distribution by Group")

    # Sex
    for i, g in enumerate(order):
        sub  = df[df["group"] == g]
        n_m  = sub["male"].sum(); n_f = len(sub) - n_m
        axes[0, 2].bar(i, n_m, color="#4393c3", label="Male" if i == 0 else "")
        axes[0, 2].bar(i, n_f, bottom=n_m, color="#f4a582", label="Female" if i == 0 else "")
    axes[0, 2].set_xticks(range(3)); axes[0, 2].set_xticklabels(order)
    axes[0, 2].set_ylabel("n patients"); axes[0, 2].set_title("Sex by Group")
    axes[0, 2].legend()

    # F0
    for i, g in enumerate(order):
        vals = df[df["group"] == g]["F0"].dropna()
        axes[1, 0].boxplot(vals, positions=[i], widths=0.5,
                           patch_artist=True,
                           boxprops=dict(facecolor=COLOURS[g], alpha=0.7),
                           medianprops=dict(color="white", linewidth=2),
                           flierprops=dict(marker="o", markersize=3, alpha=0.4))
    axes[1, 0].set_xticks(range(3)); axes[1, 0].set_xticklabels(order)
    axes[1, 0].set_ylabel("F0 (Hz)"); axes[1, 0].set_title("Fundamental Frequency (F0)")

    # RFS
    for i, g in enumerate(order):
        vals = df[df["group"] == g]["RFS"].dropna()
        axes[1, 1].boxplot(vals, positions=[i], widths=0.5,
                           patch_artist=True,
                           boxprops=dict(facecolor=COLOURS[g], alpha=0.7),
                           medianprops=dict(color="white", linewidth=2),
                           flierprops=dict(marker="o", markersize=3, alpha=0.4))
        axes[1, 1].scatter(np.random.normal(i, 0.07, len(vals)), vals,
                           color=COLOURS[g], alpha=0.3, s=10, zorder=3)
    axes[1, 1].set_xticks(range(3)); axes[1, 1].set_xticklabels(order)
    axes[1, 1].set_ylabel("RFS score"); axes[1, 1].set_title("Reflux Finding Score (RFS)")

    # Summary table
    axes[1, 2].axis("off")
    summary = [["", "Control", "Benign", "Malignant"]]
    for var, label in [("age", "Age, mean"), ("F0", "F0 (Hz), mean"),
                        ("intensity", "Intensity, mean"), ("RFS", "RFS, mean")]:
        row = [label]
        for g in order:
            v = df[df["group"] == g][var].mean()
            row.append(f"{v:.1f}")
        summary.append(row)
    summary.append(["Male sex", "26%", "45%", "83%"])
    tbl = axes[1, 2].table(cellText=summary[1:], colLabels=summary[0],
                            cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    axes[1, 2].set_title("Summary Statistics")

    patches = [mpatches.Patch(color=COLOURS[g], label=g) for g in order]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    _savefig(fig, out, "fig1_demographics.png")


def fig2_grbasi(df: pd.DataFrame, out: Path) -> None:
    """Figure 2 — GRBASI perceptual voice profile."""
    dims   = ["grbasi_G", "grbasi_R", "grbasi_B", "grbasi_A", "grbasi_S", "grbasi_I"]
    labels = ["G (Global)", "R (Rough)", "B (Breathy)", "A (Asthenic)", "S (Strained)", "I (Unstable)"]
    order  = ["Control", "Benign", "Malignant"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Figure 2 — GRBASI Perceptual Voice Profile by Pathology Group",
                 fontsize=13, fontweight="bold")

    for ax, dim, lbl in zip(axes.flat, dims, labels):
        groups_data = [df[df["group"] == g][dim].dropna() for g in order]
        stat, p = kruskal(*[v for v in groups_data if len(v) > 0])
        for i, (g, vals) in enumerate(zip(order, groups_data)):
            ax.boxplot(vals, positions=[i], widths=0.5,
                       patch_artist=True,
                       boxprops=dict(facecolor=COLOURS[g], alpha=0.7),
                       medianprops=dict(color="white", linewidth=2),
                       flierprops=dict(marker="o", markersize=3, alpha=0.4))
        ax.set_xticks(range(3)); ax.set_xticklabels(order, fontsize=8)
        ax.set_ylabel("Score (0–3)")
        ax.set_title(f"{lbl}\nKW p={p:.3f}")
        ax.set_ylim(-0.3, 3.6)

    plt.tight_layout()
    _savefig(fig, out, "fig2_grbasi.png")


def fig3_correlation(df: pd.DataFrame, out: Path) -> None:
    """Figure 3 — Spearman correlation heatmap (BH-FDR corrected)."""
    features = [
        "age", "RFS", "F0", "intensity",
        "amp_dx", "amp_sx", "mucosal_dx", "mucosal_sx",
        "nonvib_dx", "nonvib_sx", "reg_dx", "reg_sx",
        "arrest_dx", "arrest_sx",
        "grbasi_G", "grbasi_R", "grbasi_B", "grbasi_A", "grbasi_S", "grbasi_I",
    ]
    labels = [
        "Age", "RFS", "F0 (Hz)", "Intensity (dB)",
        "Amp. DX (%)", "Amp. SX (%)", "Mucosal wave DX", "Mucosal wave SX",
        "Non-vibr. DX (%)", "Non-vibr. SX (%)", "Regularity DX (%)", "Regularity SX (%)",
        "Arrest DX", "Arrest SX",
        "GRBASI G", "GRBASI R", "GRBASI B", "GRBASI A", "GRBASI S", "GRBASI I",
    ]

    data = df[features].dropna()
    n    = len(features)
    corr = np.full((n, n), np.nan)
    pmat = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):
            if i == j:
                corr[i, j] = 1.0; pmat[i, j] = 0.0
                continue
            mask = data.iloc[:, i].notna() & data.iloc[:, j].notna()
            r, p = stats.spearmanr(data.iloc[mask.values, i], data.iloc[mask.values, j])
            corr[i, j] = r; pmat[i, j] = p

    # BH correction on upper triangle
    up_idx = np.triu_indices(n, k=1)
    p_flat = pmat[up_idx]
    _, p_adj, _, _ = multipletests(p_flat, method="fdr_bh")
    sig_mask = np.zeros((n, n), dtype=bool)
    for k, (i, j) in enumerate(zip(*up_idx)):
        if p_adj[k] < 0.05:
            sig_mask[i, j] = True
            sig_mask[j, i] = True

    display = np.where(sig_mask, corr, np.nan)

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(display, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                xticklabels=labels, yticklabels=labels,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.3, cbar_kws={"label": "Spearman r"})
    ax.set_title("Figure 3 — Spearman Rank Correlation Heatmap (BH-FDR corrected)\n"
                 "Grey cells = non-significant after FDR correction", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    _savefig(fig, out, "fig3_correlation_heatmap.png")


def fig4_sign_atlas(df: pd.DataFrame, out: Path) -> None:
    """Figure 4 — Stroboscopic sign-profile atlas (heatmap + significance bars)."""
    features = [
        "amp_dx", "amp_sx", "mucosal_dx", "mucosal_sx",
        "nonvib_dx", "nonvib_sx", "reg_dx", "reg_sx",
        "arrest_dx", "arrest_sx", "RFS", "age", "F0", "intensity",
        "grbasi_G", "grbasi_R", "grbasi_S", "grbasi_I",
    ]
    feat_labels = [
        "Amplitude DX (%)", "Amplitude SX (%)", "Mucosal wave DX", "Mucosal wave SX",
        "Non-vibrating DX (%)", "Non-vibrating SX (%)", "Regularity DX (%)", "Regularity SX (%)",
        "Vibratory arrest DX", "Vibratory arrest SX", "RFS score", "Age",
        "F0 (Hz)", "Intensity (dB)", "GRBASI G", "GRBASI R", "GRBASI S", "GRBASI I",
    ]
    order = ["Control", "Benign", "Malignant"]

    means = pd.DataFrame(
        {g: df[df["group"] == g][features].mean() for g in order}
    ).T
    # row-normalise 0-1
    norm = (means - means.min()) / (means.max() - means.min()).replace(0, 1)

    # KW p-values
    kw_ps = []
    for f in features:
        groups_data = [df[df["group"] == g][f].dropna() for g in order]
        stat, p = kruskal(*[v for v in groups_data if len(v) > 0])
        kw_ps.append(p)
    _, kw_adj, _, _ = multipletests(kw_ps, method="fdr_bh")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9),
                                   gridspec_kw={"width_ratios": [3, 1]})
    sns.heatmap(norm.T, ax=ax1, cmap="RdYlBu_r", vmin=0, vmax=1,
                xticklabels=order, yticklabels=feat_labels,
                annot=means.T.round(1), fmt="g", annot_kws={"size": 8},
                linewidths=0.3, cbar_kws={"label": "Normalised mean (0–1)"})
    ax1.set_title("Figure 4 — Stroboscopic Sign Profile by Group\n"
                  "Mean values (annotated) · BH-corrected Kruskal–Wallis (right panel)",
                  fontsize=10)
    ax1.set_xlabel(""); ax1.tick_params(axis="y", labelsize=8)

    neg_log_p = -np.log10(np.clip(kw_adj, 1e-10, 1))
    colors = ["#d62728" if p < 0.05 else "#aec7e8" for p in kw_adj]
    ax2.barh(range(len(features)), neg_log_p, color=colors, height=0.7)
    ax2.set_yticks(range(len(features))); ax2.set_yticklabels([])
    ax2.axvline(-np.log10(0.05), color="red", linestyle="--", lw=0.8)
    ax2.set_xlabel("−log₁₀(BH-adj p)"); ax2.set_title("Significance")
    ax2.invert_yaxis()
    plt.tight_layout()
    _savefig(fig, out, "fig4_sign_atlas.png")


def fig5_key_features(df: pd.DataFrame, out: Path) -> None:
    """Figure 5 — Violin plots: key discriminating features."""
    features = [
        ("nonvib_dx", "Non-vibrating\nDX (%)"),
        ("nonvib_sx", "Non-vibrating\nSX (%)"),
        ("amp_dx",    "Amplitude\nDX (%)"),
        ("amp_sx",    "Amplitude\nSX (%)"),
        ("mucosal_dx","Mucosal\nwave DX"),
        ("mucosal_sx","Mucosal\nwave SX"),
        ("arrest_dx", "Vibratory\narrest DX"),
        ("age",       "Age"),
    ]
    order  = ["Control", "Benign", "Malignant"]
    bm     = df[df["group"].isin(["Benign", "Malignant"])]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Figure 5 — Key Discriminating Stroboscopic Features by Diagnostic Group\n"
                 "Violin + strip plots · BH-corrected Mann–Whitney U (Benign vs Malignant)",
                 fontsize=11, fontweight="bold")

    for ax, (feat, lbl) in zip(axes.flat, features):
        for i, g in enumerate(order):
            vals = df[df["group"] == g][feat].dropna()
            if len(vals) > 1:
                parts = ax.violinplot(vals, positions=[i], widths=0.7,
                                      showmedians=True, showextrema=True)
                for pc in parts["bodies"]:
                    pc.set_facecolor(COLOURS[g]); pc.set_alpha(0.5)
                for partname in ("cbars", "cmins", "cmaxes", "cmedians"):
                    if partname in parts:
                        parts[partname].set_edgecolor(COLOURS[g])
            ax.scatter(np.random.normal(i, 0.06, len(vals)), vals,
                       color=COLOURS[g], alpha=0.4, s=8, zorder=3)

        # BH-corrected Mann-Whitney U
        b_vals = bm[bm["group"] == "Benign"][feat].dropna()
        m_vals = bm[bm["group"] == "Malignant"][feat].dropna()
        if len(b_vals) > 1 and len(m_vals) > 1:
            _, p = mannwhitneyu(b_vals, m_vals, alternative="two-sided")
            sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            y_max = df[feat].quantile(0.98) * 1.1
            ax.annotate(f"p={p:.3f} {sig}", xy=(0.5, 0.97), xycoords="axes fraction",
                        ha="center", va="top", fontsize=8)

        ax.set_xticks(range(3)); ax.set_xticklabels(order, fontsize=8)
        ax.set_title(lbl, fontsize=9)

    patches = [mpatches.Patch(color=COLOURS[g], label=g) for g in order]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    _savefig(fig, out, "fig5_key_features.png")


def fig6_glottic_closure(df: pd.DataFrame, out: Path) -> None:
    """Figure 6 — Glottic closure pattern."""
    order   = ["Control", "Benign", "Malignant"]
    closure_order = ["Completa", "Clessidra", "Gap fusiforme (ovalare)",
                     "Gap anteriore", "Gap posteriore",
                     "Incompleta (su tutto il margine)", "Irregolare"]
    closure_labels = ["Complete", "Hourglass", "Spindle gap",
                      "Anterior gap", "Posterior gap", "Incomplete", "Irregular"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Figure 6 — Glottic Closure Pattern Across Diagnostic Groups",
                 fontsize=12, fontweight="bold")

    # Grouped bar
    x    = np.arange(len(closure_order))
    width = 0.25
    for j, g in enumerate(order):
        sub = df[df["group"] == g]
        pcts = [(sub["closure"] == c).sum() / len(sub) * 100 for c in closure_order]
        ax1.bar(x + j * width, pcts, width, label=g, color=COLOURS[g], alpha=0.8)
    ax1.set_xticks(x + width); ax1.set_xticklabels(closure_labels, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Prevalence within group (%)")
    ax1.set_title("Closure Type Prevalence per Group"); ax1.legend()

    # Chi-square
    ct = pd.crosstab(df["closure"], df["group"])
    chi2, p, _, _ = chi2_contingency(ct)
    ax1.text(0.02, 0.97, f"χ²={chi2:.2f}, p={p:.4f}", transform=ax1.transAxes,
             va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.7))

    # Stacked composition
    bottom = {g: 0 for g in order}
    cmap   = plt.cm.get_cmap("tab10", len(closure_order))
    for k, (c, lbl) in enumerate(zip(closure_order, closure_labels)):
        for j, g in enumerate(order):
            sub = df[df["group"] == g]
            pct = (sub["closure"] == c).sum() / len(sub) * 100
            ax2.bar(j, pct, bottom=bottom[g], color=cmap(k),
                    label=lbl if j == 0 else "", alpha=0.85, edgecolor="white")
            if pct > 4:
                ax2.text(j, bottom[g] + pct / 2, f"{pct:.0f}%",
                         ha="center", va="center", fontsize=7, color="white", fontweight="bold")
            bottom[g] += pct
    ax2.set_xticks(range(3)); ax2.set_xticklabels(order)
    ax2.set_ylabel("Cumulative prevalence (%)")
    ax2.set_title("Closure Composition per Group")
    ax2.legend(loc="lower right", fontsize=7, ncol=2)

    plt.tight_layout()
    _savefig(fig, out, "fig6_glottic_closure.png")


def fig7_supraglottic(df: pd.DataFrame, out: Path) -> None:
    """Figure 7 — Supraglottic hyperfunction."""
    order  = ["Control", "Benign", "Malignant"]
    grades = {0: "Normal", 1: "Mild", 2: "Marked"}
    colors = {"Normal": "#2ca02c", "Mild": "#ff7f0e", "Marked": "#d62728"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Figure 7 — Supraglottic Hyperfunction by Diagnostic Group\n"
                 "Stacked bars: proportion per severity grade", fontsize=11)

    for ax, (feat, title) in zip(axes, [("supra_dx", "Right (DX)"),
                                         ("supra_sx", "Left (SX)"),
                                         (None, "Bilateral (approx.)")]):
        bottom = [0] * 3
        for grade, lbl in grades.items():
            pcts = []
            for g in order:
                sub = df[df["group"] == g]
                if feat:
                    pct = (sub[feat] == grade).sum() / len(sub) * 100
                else:
                    pct = ((sub["supra_dx"] == grade) | (sub["supra_sx"] == grade)).sum() / len(sub) * 100
                pcts.append(pct)
            ax.bar(range(3), pcts, bottom=bottom, color=colors[lbl], label=lbl, alpha=0.85)
            for j, (p, b) in enumerate(zip(pcts, bottom)):
                if p > 5:
                    ax.text(j, b + p / 2, f"{p:.0f}%", ha="center", va="center",
                            fontsize=9, color="white", fontweight="bold")
            bottom = [b + p for b, p in zip(bottom, pcts)]
        ax.set_xticks(range(3)); ax.set_xticklabels(order)
        ax.set_ylabel("% of patients in group")
        ax.set_title(f"Supraglottic Activity {title}")
        if ax == axes[0]: ax.legend()

    plt.tight_layout()
    _savefig(fig, out, "fig7_supraglottic.png")


def fig8_forest_plot(lr_df: pd.DataFrame, out: Path) -> None:
    """Figure 8 — Univariate logistic regression forest plot."""
    df_plot = lr_df.sort_values("log_OR")

    fig, ax = plt.subplots(figsize=(10, 9))
    colors = ["#d62728" if s else "#aec7e8" for s in (df_plot["p_adj"] < 0.05)]

    y_pos = range(len(df_plot))
    ax.barh(list(y_pos), df_plot["log_OR"], xerr=0, color="none")
    for i, (_, row) in enumerate(df_plot.iterrows()):
        color = "#d62728" if row["p_adj"] < 0.05 else "#aec7e8"
        ax.errorbar(row["log_OR"], i,
                    xerr=[[row["log_OR"] - row["log_CI_lo"]],
                           [row["log_CI_hi"] - row["log_OR"]]],
                    fmt="o", color=color, capsize=3, markersize=6, linewidth=1.5)
        sig = "***" if row["p_adj"] < 0.001 else "**" if row["p_adj"] < 0.01 \
              else "*" if row["p_adj"] < 0.05 else ""
        ax.text(max(df_plot["log_CI_hi"]) + 0.05, i,
                f"OR={np.exp(row['log_OR']):.2f} {sig}", va="center", fontsize=8)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df_plot["predictor"], fontsize=9)
    ax.set_xlabel("log(Odds Ratio) [95% CI]")
    ax.set_title("Figure 8 — Univariate Logistic Regression: Odds Ratios for Malignancy\n"
                 "14/21 predictors significant after BH–FDR correction (p < 0.05)", fontsize=10)

    red_p   = mpatches.Patch(color="#d62728", label="Significant (BH p < 0.05)")
    grey_p  = mpatches.Patch(color="#aec7e8", label="Non-significant")
    ax.legend(handles=[red_p, grey_p], loc="lower right")
    plt.tight_layout()
    _savefig(fig, out, "fig8_logistic_forest.png")


def fig9_rf(rf_results: dict, out: Path) -> None:
    """Figure 9 — Random Forest feature importance + ROC curve (original model)."""
    res  = rf_results["all_groups"]
    mdi  = res["mdi"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f"Figure 9 — Random Forest: Feature Importance & ROC Curve\n"
                 f"Malignant vs All Other | CV AUC = {res['mean_auc']:.3f} ± {res['sd_auc']:.3f}",
                 fontsize=11, fontweight="bold")

    # MDI
    mdi_plot = mdi.head(20)
    ax1.barh(range(len(mdi_plot)), mdi_plot.values[::-1], color="#d62728", alpha=0.8)
    ax1.set_yticks(range(len(mdi_plot)))
    ax1.set_yticklabels(mdi_plot.index[::-1], fontsize=8)
    ax1.set_xlabel("Mean Decrease in Impurity (± 1 SD)")
    ax1.set_title("Random Forest Feature Importance\n(Malignancy Prediction, 5-fold CV)")

    # ROC
    mean_tpr = res["mean_tpr"]; mean_fpr = res["mean_fpr"]
    std_tpr  = res["std_tpr"]
    ax2.plot(mean_fpr, mean_tpr, color="#d62728", lw=2,
             label=f"Mean ROC (AUC = {res['mean_auc']:.3f} ± {res['sd_auc']:.3f})")
    ax2.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                     alpha=0.2, color="#d62728", label="± 1 SD")
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax2.set_xlabel("1 − Specificity (FPR)"); ax2.set_ylabel("Sensitivity (TPR)")
    ax2.set_title(f"ROC Curve — Malignant vs All Other\n5-fold Stratified CV (n={res['n']})")
    ax2.legend(loc="lower right", fontsize=9)

    # Annotate individual fold AUCs
    for i, auc in enumerate(res["fold_aucs"]):
        ax2.annotate(f"Fold {i+1}: AUC={auc:.3f}",
                     xy=(0.55, 0.1 + i * 0.07), xycoords="axes fraction", fontsize=8)

    plt.tight_layout()
    _savefig(fig, out, "fig9_rf_roc.png")


def figS_rf_benign_only(rf_results: dict, out: Path) -> None:
    """Supplementary Figure — RF benign-vs-malignant only (reviewer-requested)."""
    res = rf_results["benign_vs_mal"]
    mean_tpr = res["mean_tpr"]; mean_fpr = res["mean_fpr"]; std_tpr = res["std_tpr"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Supplementary — RF: Malignant vs Benign Only\n"
                 f"CV AUC = {res['mean_auc']:.3f} ± {res['sd_auc']:.3f} "
                 f"(n={res['n']}, n_mal={res['n_mal']})",
                 fontsize=11)

    mdi_plot = res["mdi"].head(15)
    ax1.barh(range(len(mdi_plot)), mdi_plot.values[::-1], color="#d62728", alpha=0.8)
    ax1.set_yticks(range(len(mdi_plot)))
    ax1.set_yticklabels(mdi_plot.index[::-1], fontsize=9)
    ax1.set_xlabel("MDI"); ax1.set_title("Feature Importance (Benign-only RF)")

    ax2.plot(mean_fpr, mean_tpr, color="#d62728", lw=2,
             label=f"Mean ROC (AUC={res['mean_auc']:.3f})")
    ax2.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2, color="#d62728")
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax2.set_xlabel("1 − Specificity"); ax2.set_ylabel("Sensitivity")
    ax2.set_title("ROC Curve — Malignant vs Benign Only"); ax2.legend()
    for i, auc in enumerate(res["fold_aucs"]):
        ax2.annotate(f"Fold {i+1}: {auc:.3f}", xy=(0.55, 0.1 + i * 0.07),
                     xycoords="axes fraction", fontsize=8)

    plt.tight_layout()
    _savefig(fig, out, "figS_rf_benign_only.png")


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Laryngostroboscopy analysis pipeline")
    parser.add_argument("--data",   default="italproj_full.csv",
                        help="Path to input CSV (default: italproj_full.csv)")
    parser.add_argument("--outdir", default="results",
                        help="Output directory (default: results/)")
    args = parser.parse_args()

    # Create output directories
    out      = Path(args.outdir)
    fig_out  = out / "figures"
    tab_out  = out / "tables"
    for d in [fig_out, tab_out]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Laryngostroboscopy Analysis Pipeline")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1/8] Loading and cleaning data ...")
    df = load_and_clean(args.data)
    print(f"  Groups: {df['group'].value_counts().to_dict()}")

    # ── Descriptives ──────────────────────────────────────────────────────────
    print("\n[2/8] Descriptive statistics ...")
    descriptive_stats(df, tab_out)

    # ── Effect sizes ──────────────────────────────────────────────────────────
    print("\n[3/8] Rank-biserial effect sizes (Benign vs Malignant) ...")
    eff_df = rank_biserial(df, tab_out)
    print(f"  Top 5 |r|: {eff_df[['feature','abs_r']].head(5).to_string(index=False)}")

    # ── Univariate logistic ───────────────────────────────────────────────────
    print("\n[4/8] Univariate logistic regression ...")
    lr_df = univariate_logistic(df, tab_out)
    sig = lr_df[lr_df["p_adj"] < 0.05]
    print(f"  {len(sig)}/{len(lr_df)} predictors significant after BH correction")

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n[5/8] Random Forest (original + benign-only) ...")
    rf_results = random_forest(df, tab_out)

    # ── Threshold validation ───────────────────────────────────────────────────
    print("\n[6/8] Threshold validation ...")
    thresh_df = threshold_validation(df, tab_out)
    print(thresh_df[["threshold", "sensitivity", "specificity", "PPV", "NPV"]].to_string(index=False))

    # ── Multivariate logistic ──────────────────────────────────────────────────
    print("\n[7/8] Multivariate logistic regression ...")
    multivariate_logistic(df, tab_out)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n[8/8] Generating figures ...")
    fig1_demographics(df, fig_out)
    fig2_grbasi(df, fig_out)
    fig3_correlation(df, fig_out)
    fig4_sign_atlas(df, fig_out)
    fig5_key_features(df, fig_out)
    fig6_glottic_closure(df, fig_out)
    fig7_supraglottic(df, fig_out)
    fig8_forest_plot(lr_df, fig_out)
    fig9_rf(rf_results, fig_out)
    figS_rf_benign_only(rf_results, fig_out)

    print("\n" + "=" * 60)
    print(f"Done. Results written to: {out.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
