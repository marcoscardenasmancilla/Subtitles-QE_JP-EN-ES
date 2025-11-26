# Subtitles-QE [JP-EN-ES]

---

## Summary / Purpose
A CPU-first, OOM-resilient set of small pipelines to compute sentence-level similarity and automatic quality-estimation signals for Japanese source subtitles vs English and Spanish translations, aggregate metrics by group, visualize distributions, run paired group comparisons, and generate post-hoc plots, tables and bootstrap CIs for statistically significant contrasts. The scripts expect a prepared CSV dataset with columns including `group`, `sub_jap`, `sub_trad_en` and `sub_trad_sp` (see `1_sub_sbert_comet_pipeline.py`).

---

## Files and high-level responsibilities
- `1_sub_sbert_comet_pipeline.py` — **Compute per-group metrics from a CSV dataset** (reads `DATASET_PATH`, computes char‑ngram cosine, punctuation jaccard, digit matching, length ratios, optional SBERT and COMET‑QE scores, batch-local min‑max normalization and a `qe_mix` weighted composite; saves per‑group CSVs and HTML reports; writes `metrics_all_groups.csv`). Defaults and behaviour are implemented in this file.

- `2_make_wide_matrix.py` — **Pivot the combined `metrics_all_groups.csv` into wide format by time stamp**. It attempts to auto-detect the time column and the group column, maps desired base metric names to actual columns heuristically, and produces: `metrics_wide_by_timestamp.csv`, `metrics_wide_valid_minonepergroup.csv`, and (when possible) `metrics_wide_strict_allmetrics.csv`.

- `3_plots_qe.py` — **Generate violin + box + strip plots per metric × group** from `metrics_wide_strict_allmetrics.csv` (or adjust `WIDE_PATH`). Produces per-metric PNGs and `descriptives_by_metric_group.csv`. It detects groups by parsing column names of the form `<metric>__<group>`.

- `4_compare_groups_qe.py` — **Perform paired statistical comparisons** across groups using the wide table (or pivot from `metrics_all_groups.csv` if wide is missing). For each metric it selects a strict or looser subset (strict = all metrics present for every group; minone = at least one metric present per group), tests normality of paired differences (Shapiro), uses paired t‑test if normal else Wilcoxon signed‑rank, computes effect sizes (paired Cohen's d or Wilcoxon r approximation), writes `pairwise_tests_<timestamp>.csv`, descriptive CSVs and a basic HTML report with embedded plots. This script includes defaults for `metrics` and `DEFAULT_WIDE` and a fallback to pivot from the raw file if necessary.

- `5_qe_posthoc_pipeline.py` — **Post‑hoc utilities** that act on the pairwise results and the wide matrix. Actions implemented:
  - **A** — create paired violin/box + paired-line plots for significant contrasts (uses `pairwise` file to select contrasts).
  - **B** — assemble a manuscript-ready CSV + LaTeX table merging pairwise statistics with descriptives and optional bootstrap CIs.
  - **C** — recompute FDR (Benjamini–Hochberg) grouped by metric and add `signif_fdr_bh` / `p_fdr_bh` columns.
  - **D** — paired bootstrap for mean-difference confidence intervals (returns lower/upper CI and `n_pairs`). Defaults and helper functions are in this script.

---

## Input data expected (script `1_sub_sbert_comet_pipeline.py`)
- By default, `1_sub_sbert_comet_pipeline.py` reads `DATASET_PATH` (a CSV). The default in your uploaded script is a local Windows path: `DATASET_PATH = ".\dataset_final.csv"`. The script expects columns such as `group`, `sub_jap`, `sub_trad_en`, `sub_trad_sp` (see code).

---

## Important configuration defaults (from `1_sub_sbert_comet_pipeline.py`)
- `OUT_ROOT` default: `.\qe_outputs` (per-group outputs are written under `group__{group}` subfolders).
- SBERT: `SBERT_ENABLE = True`, model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `SBERT_BATCH_SIZE = 64`.
- COMET‑QE: `COMET_QE_ENABLE = True`, model `Unbabel/wmt20-comet-qe-da`, `COMET_QE_BATCH_SIZE = 8`.
- QE_mix weights: `W_COMET = 0.6`, `W_SBERT = 0.4`.
- Char‑ngram `CHAR_N = 3`.
- Flagging thresholds (defaults visible in code):
  - `TH_QE_MIX_LOW = 0.25`, `TH_LENGTH_RATIO_LOW = 0.5`, `TH_LENGTH_RATIO_HIGH = 1.8`, `TH_PUNC_JACCARD_LOW = 0.3`, `TH_DIGIT_MATCH = True`, `TH_CHAR_N_HIGH_SIM = 0.95`, `TH_DISCREPANCY = 0.5`.

---

## `1_sub_sbert_comet_pipeline.py` (per group)
Under `OUT_ROOT/group__{group}/` the script writes (names from the code):
- `metrics_group_{group}.csv` — full per-row metrics for that group.
- `report_group_{group}.html` — a self-contained HTML report per group (summary stats, top flags, a histogram embedded as base64, sample rows).
- When all groups processed, the script concatenates group frames and writes `metrics_all_groups.csv` under `OUT_ROOT`.

Each `metrics_group_{group}.csv` includes columns such as:
`sub_jap`, `sub_trad_en`, `sub_trad_sp`, `cosn_jap_en`, `cosn_jap_sp`, `punc_jaccard_en`, `punc_jaccard_sp`, `digits_match_en`, `digits_match_sp`, `len_ratio_en`, `len_ratio_sp`, `sbert_jap_en`, `sbert_jap_sp`, `comet_jap_en`, `comet_jap_sp`, normalized columns `*_norm`, `qe_mix_en`, `qe_mix_sp`, and `flags_en`, `flags_sp`, `flags_all`.

---

## `2_make_wide_matrix.py`
- Default input: the combined `metrics_all_groups.csv` path (the script has `DEFAULT_INPUT` pointing to `qe_outputs/metrics_all_groups.csv`). It reads the combined CSV with tolerant parsing.
- Auto-detection: it heuristically detects a time column (candidates like `time_stamp`, `timestamp`, `time`, `timecode`, `ts`) and a group column (candidates `group`, `group_type`, `variant`, `source`, `origin`). If it can't detect them, it raises an error.
- The script maps `DEFAULT_METRICS` (a concrete list defined in the file) to actual columns in the CSV using tolerant token matching, then pivots each metric into `<actual_col>__<group>` columns. Default metric list present in the code includes e.g. `sbert_jap_en`, `comet_jap_sp`, `qe_mix_en`, `qe_mix_sp`, and their `_norm` variants.
- Outputs (in `OUT_DIR` default): `metrics_wide_by_timestamp.csv`, `metrics_wide_valid_minonepergroup.csv`, and, if possible, `metrics_wide_strict_allmetrics.csv`. The strict variant will be omitted if required columns are missing.

---

## `3_plots_qe.py`
- Default WIDE file: `WIDE_PATH` points to `qe_outputs/metrics_wide_strict_allmetrics.csv` — verifying that the wide strict matrix exists is necessary, otherwise the script will find no columns to plot.
- Metric base list: `METRICS_BASE` defined in the script lists which bases are plotted (e.g., `sbert_jap_en`, `comet_jap_en`, `qe_mix_en`, and their SP counterparts). The script forms expected columns as `<metric>__<group>` and warns when missing.
- Outputs written to `qe_plots/` (default): per-metric PNG files such as `{metric}_violin_boxplot.png` and a combined grid `combined_metrics_violin_boxplots.png`, plus `descriptives_by_metric_group.csv`.

---

## `4_compare_groups_qe.py`
- Default wide: `DEFAULT_WIDE` points to `qe_outputs/metrics_wide_strict_allmetrics.csv`. If missing, the script will attempt to pivot from `metrics_all_groups.csv` (fallback behaviour).
- Group detection: parses columns named `<metric>__<group>` to find groups. The script expects at least two groups.
- Subsetting modes: `strict` (requires every metric present for every group) or `minone` (at least one metric per group). If `strict` fails due to missing columns it falls back to `minone`.
- Statistical testing logic: for each metric and pair of groups, the script:
  - computes paired differences and runs Shapiro on the differences,
  - if Shapiro p > alpha uses paired t-test and paired Cohen's d; else uses Wilcoxon signed‑rank and computes an r-like effect size approximation,
  - collects `n_pairs`, `test`, `stat`, `pvalue`, `normal_shapiro_p`, and `effect_size` for each contrast and writes them to `pairwise_tests_<timestamp>.csv`.
- Visual outputs: for each metric the script produces a boxplot across groups and paired-line plots per contrast (sampled to avoid overplotting), saved to `qe_compare_outputs/plots/`. It also produces a basic HTML report embedding top plots.

---

## `5_qe_posthoc_pipeline.py`
- Default pairwise and descriptives paths are set to previous-run specific timestamps in your environment (see `PAIRWISE_FP_DEFAULT` and `DESCRIPTIVES_FP_DEFAULT` in the file). The script expects to be pointed to the pairwise CSV generated by `4_compare_groups_qe.py`.
- Actions:
  - **C (recompute FDR)**: adds `p_fdr_bh` and `signif_fdr_bh` using `statsmodels.stats.multitest.multipletests` grouped by metric.
  - **D (bootstrap)**: `paired_bootstrap_ci` computes paired bootstrap CI for mean differences, saving `bootstrap_cis.csv`.
  - **A (plots)**: `plot_violin_box_paired` draws violin+box+strip and paired lines for significant contrasts and saves PNGs under an `plots/` folder.
  - **B (table)**: merges descriptives and pairwise results (and optional bootstrap CIs) into `posthoc_table.csv` and `posthoc_table.tex`.

---

## Quick start (minimal)
1. Prepare `dataset_final.csv` with the columns used by `1_sub_sbert_comet_pipeline.py` (at least `group`, `sub_jap`, and one or both `sub_trad_en`/`sub_trad_sp`).
2. Run per-group metric computation:
```bash
python 1_sub_sbert_comet_pipeline.py
```
This creates `qe_outputs/group__{group}/metrics_group_{group}.csv` and `qe_outputs/metrics_all_groups.csv`.
3. Pivot to wide:
```bash
python 2_make_wide_matrix.py --input /path/to/qe_outputs/metrics_all_groups.csv
```
4. Plot distributions:
```bash
python 3_plots_qe.py
```
5. Compare groups (strict mode recommended, fallback to minone):
```bash
python 4_compare_groups_qe.py --input /path/to/metrics_wide_strict_allmetrics.csv --mode strict
```
6. Post-hoc: recompute FDR, bootstrap and plots:
```bash
python 5_qe_posthoc_pipeline.py --pairwise path/to/pairwise.csv --descriptives path/to/descriptives.csv --wide path/to/metrics_wide_strict_allmetrics.csv --do C D A B
```

---

## Notes:
- **Group semantics**: The pipeline is set up to compare different *groups* (e.g., `official`, `fansub`, `ai`) as encoded in the `group` column. `4_compare_groups_qe.py`'s top comment already cites `official, fansub, ai`. Use the `group` column consistently.
- **Normalization scope**: `qe_mix` normalization is performed per-group/frame in `1_sub_sbert_comet_pipeline.py` using `minmax_series()` on the group-level frame (batch-local). This means `qe_mix` values are relative within each processed group frame (not globally comparable across runs) unless you enforce a global normalization step after concatenating groups.
- **COMET & SBERT availability**: Both models are optional (`SBERT_ENABLE`, `COMET_QE_ENABLE`); the scripts fallback to NaNs if models fail to load. Expect long first-run downloads for COMET.

---

# Licensing & contact
- GNU Affero General Public License v3.0.
- Author / maintainer: Dr. Marcos H. Cárdenas-Mancilla.
- Date of creation: November 25, 2025.
- Contact: marcoscardenasmancilla@gmail.com
- For reproducibility issues, include the output of `print_env_info()` when opening issues.

---

## Final notes
This README documents design decisions and implementation details to help new developers and researchers understand and adapt the pipeline. The code is intentionally conservative and defensive to make it robust for research usage on modest hardware.
