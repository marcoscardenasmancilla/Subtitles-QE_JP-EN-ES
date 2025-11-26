# ES–EN QE & SBERT Pipeline — README

Local pipeline that extracts sentence-level Quality Estimation (COMET-QE), semantic adequacy (SBERT), structural heuristics, then pivots, plots, compares groups and runs post-hoc analyses.

---

## Repository contents (main scripts)

- **`1_sub_sbert_comet_pipeline.py`**  
  Extracts per-pair / per-sentence metrics from parallel CoNLL-U files: COMET-QE (no reference), SBERT cosine (ES↔EN), structural heuristics (length ratio, digits, punctuation Jaccard, char-N cosines), and a composite `qe_mix`. Produces per-pair CSVs, plots, and an HTML report.

- **`2_make_wide_matrix.py`**  
  Takes a row-wise aggregate (e.g., `metrics_all_groups.csv`) and pivots it into *wide* matrices with columns named like `<metric>__<group>` required by plotting and comparison scripts. Produces several wide outputs (strict / loose variants).

- **`3_plots_qe.py`**  
  Reads a wide matrix and creates distribution figures (violin + box + strip) per metric × group, and a descriptive CSV (`descriptives_by_metric_group.csv`).

- **`4_compare_groups_qe.py`**  
  Performs paired comparisons (within `time_stamp`) between groups for chosen metrics. Auto-detects normality and selects paired t-test or Wilcoxon; computes effect sizes, saves CSV results, plots, and a basic HTML report.

- **`5_qe_posthoc_pipeline.py`**  
  Post-hoc utilities: generate plots for significant contrasts, assemble manuscript-ready tables (CSV/LaTeX), recompute FDR, and compute paired bootstrap CIs for mean differences.

---

## Quick start

1. Prepare your parallel data (CoNLL-U pairs) and run per-pair metric extraction:

```bash
python 1_sub_sbert_comet_pipeline.py
```

2. Consolidate per-pair outputs into a single `metrics_all_groups.csv` (one row per `time_stamp` × `group` × metrics).  
   *(This aggregation step is assumed to be performed outside these scripts — e.g., a small script or manual merge.)*

3. Pivot into a wide matrix:

```bash
python 2_make_wide_matrix.py --input /path/to/metrics_all_groups.csv
# Default outputs:
# - metrics_wide_by_timestamp.csv
# - metrics_wide_valid_minonepergroup.csv
# - metrics_wide_strict_allmetrics.csv
```

4. Create plots:

```bash
python 3_plots_qe.py --wide metrics_wide_strict_allmetrics.csv
```

5. Run group comparisons (example — via a Python wrapper or invoking the script if you add a CLI entry point):

```python
# Example wrapper usage if the script exposes a run_pipeline(...) function
from compare_groups_qe import run_pipeline
run_pipeline(input_path="metrics_wide_strict_allmetrics.csv",
             metrics=["comet_jap_sp_norm","sbert_jap_en_norm"],
             mode="strict")
```

6. Run post-hoc actions (plots, tables, FDR, bootstrap):

```bash
python 5_qe_posthoc_pipeline.py --do A B C D --pairwise path/to/pairwise.csv --wide metrics_wide_strict_allmetrics.csv
# --do accepts:
# A -> generate paired plots for significant contrasts
# B -> produce manuscript table (CSV + LaTeX)
# C -> recompute FDR across contrasts
# D -> compute paired bootstrap CIs (use --nboot to set number of resamples)
```

---

## Requirements

Install the Python packages below (example):

```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn jinja2 sentence-transformers unbabel-comet
```

Notes:
- If you only need plotting and stats (no SBERT / COMET), you can skip `sentence-transformers` and `unbabel-comet`. The scripts attempt graceful fallbacks and will proceed with `NaN`s for missing model outputs.
- The scripts default to **CPU-first** behavior and include batching and OOM retry heuristics. If you have a GPU, you can configure devices in the script headers.

---

## Configuration (what to edit)

Each script contains a top “CONFIG / DEFAULTS” section you can edit. Common configuration keys:

- Paths
  - `COMMON_SOURCE_DIR` — directory with input CoNLL-U pairs
  - `COMMON_OUTPUTS_DIR` — root for per-pair outputs
  - `WIDE_INPUT_PATH` — input wide CSV for plotting/comparing

- SBERT
  - `SBERT_MODEL` (e.g., `'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'`)
  - `SBERT_DEVICE` (`"cpu"` or `"cuda"`)
  - `SBERT_BATCH_SIZE`

- COMET-QE
  - `COMET_QE_MODEL` (e.g., `'Unbabel/wmt20-comet-qe-da'` or desired QE checkpoint)
  - `COMET_QE_DEVICE` (`"cpu"` or `"cuda"`)
  - `COMET_QE_BATCH_SIZE`

- QE_mix
  - `QE_MIX_WEIGHTS` (e.g., `{"comet":0.6, "sbert":0.4}`)
  - `QE_MIX_LOW` — lower bound threshold for alerting

- Pivot & comparison
  - `DEFAULT_METRICS` — list of metric base names to pivot or analyze
  - `PIVOT_MODE` — `"strict"` or `"minone"` options in pivot/comparison scripts

---

## Outputs (typical)

- Per-pair folder `parallel_outputs_{stem}_ES_EN/`  
  - `parallel_es_en_metrics.csv` — per sentence metrics  
  - `parallel_corpus_aligned.csv` — source/hypothesis alignment table  
  - `needs_review.csv` — flagged rows for human inspection  
  - `plots/` — per-sentence similarity and structure figures  
  - `{stem}_report.html` — single HTML report per pair

- Aggregation & pivot
  - `metrics_all_groups.csv` — aggregated raw table (input)  
  - `metrics_wide_strict_allmetrics.csv` — strict wide matrix  
  - `metrics_wide_valid_minonepergroup.csv` — looser wide matrix

- Figures & stats
  - `qe_plots/*.png` — per-metric/group distribution plots  
  - `descriptives_by_metric_group.csv` — summary stats per metric × group  
  - `qe_compare_outputs/` — pairwise tests CSVs, plots, HTML reports

- Post hoc
  - `qe_posthoc_outputs/posthoc_table.csv` (+ LaTeX)  
  - `qe_posthoc_outputs/bootstrap_cis.csv`  
  - `qe_posthoc_outputs/*.png`

---

## Statistical decisions implemented

- Normality test: **Shapiro–Wilk** on paired differences (or rank assumptions)  
- If normal: **paired t-test** + paired Cohen’s d  
- If not normal: **Wilcoxon signed-rank** test + rank-biserial or equivalent effect size  
- Multiple comparisons: **Benjamini–Hochberg (FDR)** correction option available in post-hoc script  
- Bootstrap: paired bootstrap for mean difference CIs (configurable `--nboot`)

---

## Troubleshooting & FAQs

**Q: COMET or SBERT imports fail / model won’t load.**  
A: Install missing packages (`sentence-transformers`, `unbabel-comet`). Scripts print warnings and continue with `NaN` values for missing model outputs so you can still use pivot/plot functionality.

**Q: OOM (Out of Memory) on GPU.**  
A: Reduce batch sizes in the CONFIG section, set `SBERT_DEVICE="cpu"` to force CPU, or decrease `COMET_QE_BATCH_SIZE`. Also consider setting environment variables to limit thread usage (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`).

**Q: `2_make_wide_matrix.py` fails because columns are missing.**  
A: Use the looser output `metrics_wide_valid_minonepergroup.csv` or add / rename columns to match expected metric base names. You can pass `--metrics` to explicitly control which metrics to pivot.

**Q: I want to include another metric.**  
A: Add metric extraction in `1_sub_sbert_comet_pipeline.py`, then update `DEFAULT_METRICS` in `2_make_wide_matrix.py`, `3_plots_qe.py`, and `5_qe_posthoc_pipeline.py`.

---

## Reproducibility & best practices

- Keep a copy of model names / commits used for SBERT and COMET in your experiment logs.  
- Save the `metrics_all_groups.csv` used to generate each wide matrix and analysis report (timestamped file naming recommended).  
- Version your environment (`requirements.txt` or `conda` environment) and GPU/CPU specs to help reproduce runs.

---

## Contributing

- Add new metrics as modular functions (follow existing structure).  
- If you change metric names, update `DEFAULT_METRICS` and any mapping logic in `2_make_wide_matrix.py`.  
- Add unit tests for pivoting and normality/effect-size computation if you extend statistical functionality.

---

## License & citation
