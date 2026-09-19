# Laryngostroboscopy Malignancy Analysis

**Hierarchical prioritisation of laryngostroboscopic and clinical features for distinguishing malignant from benign vocal fold disease: a retrospective study of 305 patients**

Bianchino A, Mihai IS et al.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository contains the full analysis code for the manuscript, across two pipelines that reflect the chronological development of the project:

- **`analysis_R/`** — original exploratory pipeline (thesis stage). Broad, data-type-agnostic association mapping and first-pass modelling across all variable types.
- **`analysis_python/`** — extended manuscript pipeline. Hypothesis-driven analysis focused on binary malignancy discrimination, with the statistical methods and figures reported in the paper.

The input data for both pipelines is `italproj_full.csv` (305 patients, semicolon-separated, Italian headers). The file is not included in this repository; it is available from the corresponding author upon reasonable request.

---

## Repository structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── analysis_R/
│   ├── 00_utils.R              # Shared helpers: argument parsing, dir creation, I/O
│   ├── 01_clean_encode.R       # Data loading, cleaning, feature engineering, cohort split
│   ├── 02_assoc_maps.R         # Cross-type association matrices and heatmaps
│   ├── 03_models.R             # LM, multinomial, ordinal regression, random forest
│   └── 04_freq_tables_plots.R  # Frequency tables and stacked bar plots
└── analysis_python/
    └── laryngostroboscopy_analysis.py  # Complete manuscript analysis pipeline
```

---

## R pipeline — exploratory analysis

### What it does

The R pipeline was developed during the thesis stage as a broad, automated exploration of the dataset. Rather than targeting a specific outcome, it classifies every column by data type (numeric, binary, ordinal, categorical) and runs a comprehensive sweep of cross-type statistical associations and models. This was the stage at which the most informative variables were identified, motivating the focused binary-malignancy analysis in the Python pipeline.

| Script | Purpose |
|--------|---------|
| `00_utils.R` | CLI argument parsing, directory creation, safe save wrappers |
| `01_clean_encode.R` | Load CSV, replace blanks, parse RFS, harmonise factor/numeric columns, split cohorts (control vs disease), run Levene + Wilcoxon for RFS comparison, construct and save typed data panels as `.rds` |
| `02_assoc_maps.R` | Spearman correlations (numeric×numeric, numeric×ordinal, numeric×categorical); Cramér's V (binary×binary); point-biserial (binary×ordinal); Kruskal–Wallis η² (ordinal×categorical); all saved as CSV + PDF heatmaps |
| `03_models.R` | Linear models (numeric Y ~ categorical X); multinomial logistic (categorical Y ~ all numeric, with Tjur R²); ordinal regression via `polr` (ordinal Y ~ numeric X, univariate); random forest with MeanDecreaseAccuracy importance (ordinal Y) |
| `04_freq_tables_plots.R` | Count tables and stacked bar plots for binary, categorical, and ordinal variables |

### Requirements

R ≥ 4.1 and:

```r
install.packages(c(
  "dplyr", "ggplot2", "reshape2", "car", "Hmisc", "tidyr",
  "nnet", "MASS", "randomForest"
))
```

### Usage

Run scripts sequentially from the repo root (01 must run before 02–04):

```bash
Rscript analysis_R/01_clean_encode.R --in data/italproj_full.csv --outdir results --figdir figures
Rscript analysis_R/02_assoc_maps.R   --outdir results --figdir figures
Rscript analysis_R/03_models.R       --outdir results --figdir figures
Rscript analysis_R/04_freq_tables_plots.R --outdir results --figdir figures
```

### Outputs

**`results/`**
- `numeric_data.rds`, `binary_nonnumeric.rds`, `ordinal_nn.rds`, `ordinal_nn_mapped.rds`, `categoric_nonnumeric.rds`, `categoric_nonnumeric_mapped.rds` — typed data panels passed between scripts
- `significant_spearman_*.csv` — Spearman correlation matrices (significant pairs only)
- `binary_cramersV.csv`, `binary_vs_ordinal_pointbiserial_sig.csv`, `ordinal_vs_categoric_eta2_sig.csv`
- `lm_numericY_categoricX_significant.csv`, `multinom_categoricY_numericX_significant.csv`, `polr_ordinalY_numericX_significant.csv`, `rf_ordinalY_importance.csv`

**`figures/`**
- `rfs_group_boxplot.pdf` — RFS control vs disease
- `spearman_numeric.pdf`, `spearman_num_vs_ord.pdf`, `spearman_num_vs_cat.pdf`
- `cramersV_binary.pdf`, `pointbiserial_bin_vs_ord.pdf`, `kruskal_eta2_ord_vs_cat.pdf`
- `binary_counts.pdf`, `categoric_counts.pdf`, `ordinal_counts.pdf`

### Note on RFS parsing

In `01_clean_encode.R`, RFS is parsed as the ratio `a/b` (e.g. `"3 / 48"` → `0.0625`). The manuscript and the Python pipeline use the raw score `a` (e.g. `3.0`). Both representations are monotonically equivalent for ranking purposes; the raw score is used in all reported statistics.

---

## Python pipeline — manuscript analysis

### What it does

The Python pipeline was written for the manuscript and targets binary malignancy discrimination specifically. It reproduces all statistical analyses and figures in the paper, including the reviewer-requested additional analyses (threshold validation, benign-only random forest, multivariate logistic regression adjusted for age and sex).

### Requirements

Python ≥ 3.9 and:

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn
```

### Usage

```bash
python analysis_python/laryngostroboscopy_analysis.py \
    --data italproj_full.csv \
    --outdir results/
```

Both arguments are optional and default to `italproj_full.csv` and `results/`.

### Analysis steps

1. Data loading and cleaning — parses all Italian-format columns (e.g. `"80 %"` → 80, `"0 Tutte vibranti"` → 0, `"Presenti costantemente"` → 2)
2. Descriptive statistics — Table 1 with Kruskal–Wallis p-values
3. Rank-biserial effect sizes — Mann–Whitney U (Benign vs Malignant), BH–FDR corrected
4. Univariate logistic regression — ORs with 95% Wald CI, BH–FDR corrected
5. Random Forest — two models: malignant vs all (original); malignant vs benign only (reviewer-requested); both 5-fold stratified CV
6. Threshold validation — sensitivity, specificity, PPV, NPV at non-vibrating DX >30% and regularity DX <40%
7. Multivariate logistic regression — top stroboscopic features adjusted for age and sex
8. Figures — all 9 main manuscript figures + supplementary RF benign-only figure

### Key results

| Metric | Value |
|--------|-------|
| Strongest malignancy discriminator | Regularity DX (\|r\| = 0.70) |
| Second strongest | Non-vibrating portion DX (\|r\| = 0.67) |
| Highest univariate OR | Vibratory arrest DX (OR = 3.55, 95% CI 2.34–5.40) |
| RF AUROC — malignant vs all | 0.960 ± 0.024 |
| RF AUROC — malignant vs benign only | 0.960 ± 0.043 |
| Non-vibrating >30%: sensitivity / specificity / NPV | 70.0% / 87.5% / 96.1% |
| Regularity <40%: sensitivity / specificity / NPV | 66.7% / 84.3% / 95.6% |
| Combined threshold: sensitivity / specificity / NPV | 76.7% / 80.0% / 96.7% |

### Note on thresholds

The non-vibrating >30% and regularity <40% thresholds are **data-derived (post-hoc, exploratory)** and have not been validated in an independent cohort. The `threshold_validation.csv` output includes an explicit note to this effect.

---

## Relationship between the two pipelines

The R and Python pipelines are complementary, not redundant. The R pipeline is wider — it explores associations across all variable types without a fixed outcome — and was used to identify candidate features. The Python pipeline is narrower and deeper — it focuses on binary malignancy, applies BH correction throughout, computes effect sizes and ORs in familiar clinical units, and produces all manuscript figures. Running both pipelines on the same dataset demonstrates the consistency of the findings across methods and languages.

---

## Data availability

`italproj_full.csv` is not included in this repository to protect patient privacy. The data are available from the corresponding author upon reasonable request, as stated in the manuscript.

---

## Citation

If you use this code, please cite:

> Bianchino A, Mihai IS et al. Hierarchical prioritisation of laryngostroboscopic and clinical features for distinguishing malignant from benign vocal fold disease: a retrospective study of 305 patients. [Journal, Year, DOI — to be updated on acceptance]

---

## License

MIT — see [LICENSE](LICENSE).
