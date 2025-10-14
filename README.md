# biomarkers_rfs-otolaryngology-pipeline
Integrated R pipeline for voice/laryngoscopy data — RFS analysis, correlations, and models.


# Overview

**Integrated R pipeline for quantitative laryngology and voice-clinic data** — from raw tables to reproducible RFS-driven statistical and modeling outputs.

**voice-rfs-pipeline** is a modular and fully scripted R workflow for analyzing **laryngoscopic** and **phonatory datasets**, with a focus on **RFS** as a central outcome variable.

It performs data cleaning, encoding, and statistical mapping across **numeric**, **binary**, **ordinal**, and **categorical** features. The pipeline unifies exploratory statistics (Spearman, Kruskal, Cramér’s V, point-biserial) with first-pass modeling (LM, multinomial, ordinal regression, random forests), producing interpretable, publication-ready outputs.


## Installation and requirements

Requires **R ≥ 4.1** and:

```r
install.packages(c(
  "dplyr","ggplot2","reshape2","car","Hmisc","dunn.test",
  "nnet","MASS","randomForest","tidyr"
))
```

## Usage

Run stages from repo root.

```bash
# 1) Clean, engineer, split; saves RDS panels
Rscript scripts/01_clean_encode.R --in data/italproj_full.csv --outdir results --figdir figures

# 2) Association matrices + heatmaps (loads RDS from results/)
Rscript scripts/02_assoc_maps.R --outdir results --figdir figures

# 3) First-pass models: LM, multinomial, ordinal, random forest
Rscript scripts/03_models.R --outdir results --figdir figures

# 4) Frequency tables and stacked-bar plots
Rscript scripts/04_freq_tables_plots.R --outdir results --figdir figures
```

## Outputs

### `results/`
- `numeric_data.rds`, `binary_nonnumeric.rds`, `ordinal_nn*.rds`, `categoric_nonnumeric*.rds`
- Correlations and associations:
  - `significant_spearman_*`
  - `binary_cramersV.csv`
  - `binary_vs_ordinal_pointbiserial_*.csv`
  - `ordinal_vs_categoric_eta2_sig.csv`
- Modeling:
  - `lm_numericY_categoricX_significant.csv`
  - `multinom_categoricY_numericX_significant.csv`
  - `polr_ordinalY_numericX_significant.csv`
  - `rf_ordinalY_importance.csv`

### `figures/`
- `rfs_group_boxplot.pdf` (control vs disease)
- Heatmaps for numeric/ordinal/categoric cross-maps
- `cramersV_binary.pdf`, `kruskal_eta2_ord_vs_cat.pdf`
- Stacked bar plots for binary/categoric/ordinal counts

## Analytical design

| Stage | Type | Core method | Goal |
|------:|:-----|:------------|:-----|
| 01 | Preprocessing | Cleaning, encoding | Harmonize variables, derive RFS |
| 02 | Statistics | Spearman, χ², Kruskal, η² | Identify multi-type associations |
| 03 | Modeling | LM, multinomial, polr, RF | Find significant predictors |
| 04 | Visualization | Barplots, heatmaps | Summaries and distributions |

## Notes

- Uses **relative paths** (`results/`, `figures/`).
- Creates directories as needed and stores intermediate `.RDS` panels.
- Prefer **Benjamini–Hochberg** correction when aggregating many tests.
- Save reproducibility info:
  ```r
  capture.output(sessionInfo(), file = file.path("results","session_info.txt"))
  ```


