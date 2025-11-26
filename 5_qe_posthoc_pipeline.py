#!/usr/bin/env python3
# qe_posthoc_pipeline.py
# Pipeline implementing A (plots), B (table), C (recompute FDR), D (bootstrap CIs)
# Usage examples:
#   python qe_posthoc_pipeline.py --all
#   python qe_posthoc_pipeline.py --do A B --metrics comet_jap_sp_norm sbert_jap_sp_norm
#
# Defaults use the local files detected in your environment.

import os
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# optional seaborn; fallback to matplotlib-only styling
try:
    import seaborn as sns
    sns.set(style="whitegrid")
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

# ------------------ Defaults & paths ------------------
PAIRWISE_FP_DEFAULT = r"C:\Users\Marcos\Desktop\2025-2\Tesis_Samir\qe_outputs\qe_compare_outputs\stats\pairwise_tests_20251120T220239Z.csv"
DESCRIPTIVES_FP_DEFAULT = r"C:\Users\Marcos\Desktop\2025-2\Tesis_Samir\qe_outputs\qe_compare_outputs\stats\descriptives_20251120T220239Z.csv"
WIDE_FP_DEFAULT = r"C:\Users\Marcos\Desktop\2025-2\Tesis_Samir\qe_outputs\metrics_wide_strict_allmetrics.csv"

OUT_DIR = Path(r"C:\Users\Marcos\Desktop\2025-2\Tesis_Samir\qe_outputs\qe_posthoc_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Default metrics to process (the four normalized metrics you asked)
DEFAULT_METRICS = [
    "comet_jap_sp_norm",
    "comet_jap_en_norm",
    "sbert_jap_sp_norm",
    "sbert_jap_en_norm"
]

# ------------------ Helpers ------------------
def detect_pvalue_column(df):
    candidates = ["pvalue","p_value","p.value","pval","p_val"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: numeric column named like 'p'
    for c in df.columns:
        if c.lower().startswith("p") and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("Could not detect p-value column in pairwise file. Columns: " + ", ".join(df.columns))

def recompute_fdr(pairwise_df, p_col=None, alpha=0.05, new_cols_prefix="p_fdr_bh"):
    if p_col is None:
        p_col = detect_pvalue_column(pairwise_df)
    pvals = pairwise_df[p_col].fillna(1.0).astype(float).values
    pairwise_df[['signif_fdr_bh','p_fdr_bh']] = pairwise_df.groupby('metric')[p_col].apply(lambda s: pd.DataFrame(multipletests(s.fillna(1.0).astype(float).values, method='fdr_bh')[:2]).T.set_index(s.index).rename(columns={0:'signif_fdr_bh',1:'p_fdr_bh'})).reset_index(level=0, drop=True)
    return pairwise_df, p_col

def es_label_r(r):
    try:
        v = abs(float(r))
    except:
        return "NA"
    if v < 0.1: return "negligible (<0.1)"
    if v < 0.3: return "small (0.1-0.3)"
    if v < 0.5: return "medium (0.3-0.5)"
    return "large (>=0.5)"

def paired_bootstrap_ci(series1, series2, n_boot=2000, ci=95, random_state=42):
    # series1 and series2 are aligned (same index), both numeric (np arrays)
    x = np.asarray(series1, dtype=float)
    y = np.asarray(series2, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]; y = y[mask]
    n = len(x)
    if n == 0:
        return (np.nan, np.nan, n)
    rng = np.random.default_rng(random_state)
    diffs = x - y
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = diffs[idx].mean()
    lo = np.percentile(boot_means, (100 - ci) / 2.0)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2.0)
    return (lo, hi, n)

def plot_violin_box_paired(wide_df, metric_base, g1, g2, pair_row, outpath_png):
    """
    wide_df: wide table where columns are like '<metric>__<group>'
    metric_base: e.g. 'sbert_jap_en_norm'
    g1,g2: group names
    pair_row: a pd.Series row from pairwise table for annotations (p, p_fdr_bh, effect_size)
    """
    col1 = f"{metric_base}__{g1}"
    col2 = f"{metric_base}__{g2}"
    if col1 not in wide_df.columns or col2 not in wide_df.columns:
        print(f"[WARN] Missing columns for metric {metric_base} groups {g1}/{g2}. Skipping plot.")
        return False

    dfplot = wide_df[[col1, col2, "time_stamp"]].rename(columns={col1: "g1", col2: "g2"})
    # drop if both nan
    dfplot = dfplot[ dfplot[["g1","g2"]].notna().any(axis=1) ].copy()
    # build long format for violin
    rows = []
    for _, r in dfplot.iterrows():
        if not pd.isna(r["g1"]):
            rows.append({"time_stamp": r["time_stamp"], "group": g1, "value": r["g1"]})
        if not pd.isna(r["g2"]):
            rows.append({"time_stamp": r["time_stamp"], "group": g2, "value": r["g2"]})
    if not rows:
        print("[WARN] No rows to plot for", metric_base, g1, g2)
        return False
    long = pd.DataFrame(rows)

    plt.figure(figsize=(8,5))
    if HAS_SEABORN:
        sns.violinplot(data=long, x="group", y="value", inner=None, cut=0)
        sns.boxplot(data=long, x="group", y="value", width=0.12, showcaps=True,
                    boxprops={'facecolor':'white','zorder':2}, showfliers=False)
        sns.stripplot(data=long, x="group", y="value", color="k", size=2, jitter=True, alpha=0.3)
    else:
        # matplotlib fallback: violinplot + boxplot + jitter
        groups = [g1, g2]
        data = [long.loc[long["group"]==g, "value"].dropna().values for g in groups]
        parts = plt.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']: pc.set_alpha(0.6)
        plt.boxplot(data, widths=0.12, positions=[1,2], patch_artist=True, boxprops=dict(facecolor='white'))
        # jitter points
        for i, ys in enumerate(data):
            if len(ys) == 0: continue
            x = np.random.normal(loc=i+1, scale=0.06, size=len(ys))
            plt.scatter(x, ys, color='k', s=6, alpha=0.3)
        plt.xticks([1,2], groups)

    # Paired lines for a random subset to avoid clutter
    paired = dfplot.dropna(subset=["g1","g2"])
    sample_n = min(len(paired), 200)
    if sample_n > 0:
        sample = paired.sample(n=sample_n, random_state=1)
        for _, r in sample.iterrows():
            plt.plot([0,1],[r["g1"], r["g2"]], color="gray", alpha=0.25)

    # annotations: means, p, p_fdr, effect
    mean_g1 = dfplot["g1"].mean()
    mean_g2 = dfplot["g2"].mean()
    mean_diff = mean_g1 - mean_g2
    p_raw = pair_row.get("pvalue", np.nan)
    p_fdr = pair_row.get("p_fdr_bh", np.nan)
    es = pair_row.get("effect_size", np.nan)
    title = f"{metric_base} — {g1} vs {g2}"
    subtitle = f"mean {g1}={mean_g1:.4f}, {g2}={mean_g2:.4f} | diff={mean_diff:.4f}\n p={p_raw:.3g}, p_fdr={p_fdr:.3g}, r={es}"
    plt.title(title + "\n" + subtitle)
    plt.tight_layout()
    plt.savefig(outpath_png, dpi=200)
    plt.close()
    return True

# ------------------ Core actions A,B,C,D ------------------

def action_C_recompute_fdr(pairwise_fp, alpha=0.05, out_fp=None):
    df = pd.read_csv(pairwise_fp, engine="python", encoding="utf-8")
    orig_pcol = detect_pvalue_column(df)
    df, pcol = recompute_fdr(df, p_col=orig_pcol, alpha=alpha)
    if out_fp is None:
        out_fp = Path(OUT_DIR) / "pairwise_with_recomputed_fdr.csv"
    df.to_csv(out_fp, index=False, encoding="utf-8")
    print("[C] Wrote recomputed FDR to:", out_fp)
    return df, pcol, out_fp

def action_A_plots(pairwise_df, wide_fp, metrics, out_folder):
    wide = pd.read_csv(wide_fp, encoding="utf-8", on_bad_lines="warn")
    if "time_stamp" not in wide.columns:
        # assume first column is time identifier
        wide = wide.rename(columns={wide.columns[0]: "time_stamp"})
    out_folder = Path(out_folder); out_folder.mkdir(parents=True, exist_ok=True)
    made = []
    for m in metrics:
        # find all relevant pairwise rows for this metric that are significant after FDR (if present) else p<0.05
        rows = pairwise_df[pairwise_df["metric"] == m].copy()
        if "signif_fdr_bh" in rows.columns:
            rows = rows[rows["signif_fdr_bh"] == True]
        else:
            pv = detect_pvalue_column(rows)
            rows = rows[rows[pv] < 0.05]
        if rows.empty:
            print("[A] No significant contrasts to plot for metric:", m)
            continue
        # for each contrast generate a plot
        for _, pr in rows.iterrows():
            g1 = pr.get("group1") or pr.get("group_1") or pr.get("g1")
            g2 = pr.get("group2") or pr.get("group_2") or pr.get("g2")
            if pd.isna(g1) or pd.isna(g2):
                print("[A] Can't detect group columns for a row: skip")
                continue
            name = f"{m}__{g1}_vs_{g2}.png"
            outpng = out_folder / name
            ok = plot_violin_box_paired(wide, m, str(g1), str(g2), pr, outpng)
            if ok:
                made.append(str(outpng))
    print("[A] Plots saved:", len(made), "files to", out_folder)
    return made

def action_D_bootstrap(pairwise_df, wide_fp, metrics, out_folder, n_boot=2000):
    wide = pd.read_csv(wide_fp, encoding="utf-8", on_bad_lines="warn")
    if "time_stamp" not in wide.columns:
        wide = wide.rename(columns={wide.columns[0]: "time_stamp"})
    results = []
    for m in metrics:
        rows = pairwise_df[pairwise_df["metric"] == m].copy()
        if "signif_fdr_bh" in rows.columns:
            rows = rows[rows["signif_fdr_bh"] == True]
        else:
            pv = detect_pvalue_column(rows)
            rows = rows[rows[pv] < 0.05]
        for _, r in rows.iterrows():
            g1 = r.get("group1") or r.get("group_1") or r.get("g1")
            g2 = r.get("group2") or r.get("group_2") or r.get("g2")
            col1 = f"{m}__{g1}"; col2 = f"{m}__{g2}"
            if col1 not in wide.columns or col2 not in wide.columns:
                print("[D] Missing columns for bootstrap:", col1, col2)
                lo, hi, n = (np.nan, np.nan, 0)
            else:
                lo, hi, n = paired_bootstrap_ci(wide[col1], wide[col2], n_boot=n_boot)
            results.append({
                "metric": m,
                "group1": g1,
                "group2": g2,
                "bootstrap_ci_lower": lo,
                "bootstrap_ci_upper": hi,
                "n_pairs": int(n)
            })
    outp = Path(out_folder) / "bootstrap_cis.csv"
    pd.DataFrame(results).to_csv(outp, index=False, encoding="utf-8")
    print("[D] Bootstrap CIs saved to:", outp)
    return outp

def action_B_table(pairwise_df, descriptives_fp, bootstrap_fp=None, out_folder=None):
    # Combine pairwise with descriptives means and optional bootstrap CIs to produce manuscript table
    desc = pd.read_csv(descriptives_fp, engine="python", encoding="utf-8")
    desc_lookup = {(r.metric, r.group): r.mean for r in desc.itertuples()}
    # canonical p-value column
    pv_col = detect_pvalue_column(pairwise_df)
    rows = []
    for _, r in pairwise_df.iterrows():
        g1 = r.get("group1") or r.get("group_1") or r.get("g1")
        g2 = r.get("group2") or r.get("group_2") or r.get("g2")
        m = r.get("metric")
        mean_g1 = desc_lookup.get((m, g1), np.nan)
        mean_g2 = desc_lookup.get((m, g2), np.nan)
        mean_diff = mean_g1 - mean_g2 if not (math.isnan(mean_g1) or math.isnan(mean_g2)) else r.get("mean_diff_g1_g2", np.nan)
        p_raw = r.get(pv_col, np.nan)
        p_fdr = r.get("p_fdr_bh", np.nan) if "p_fdr_bh" in r.index else np.nan
        es = r.get("effect_size", np.nan)
        rows.append({
            "metric": m,
            "group1": g1, "group2": g2,
            "n_pairs": int(r.get("n_pairs", np.nan)) if not pd.isna(r.get("n_pairs", np.nan)) else np.nan,
            "mean_g1": mean_g1, "mean_g2": mean_g2, "mean_diff": mean_diff,
            "test": r.get("test", ""),
            "p_raw": p_raw, "p_fdr": p_fdr,
            "effect_size": es, "es_magnitude": es_label_r(es)
        })
    out_df = pd.DataFrame(rows)
    # merge bootstrap if provided
    if bootstrap_fp and Path(bootstrap_fp).exists():
        boot = pd.read_csv(bootstrap_fp, engine="python", encoding="utf-8")
        out_df = out_df.merge(boot, how="left", left_on=["metric","group1","group2"], right_on=["metric","group1","group2"])
    # save CSV and LaTeX
    out_folder = Path(out_folder or OUT_DIR); out_folder.mkdir(parents=True, exist_ok=True)
    csv_fp = out_folder / "posthoc_table.csv"
    tex_fp = out_folder / "posthoc_table.tex"
    out_df.to_csv(csv_fp, index=False, encoding="utf-8")
    # produce a simple LaTeX table (use pandas)
    try:
        with open(tex_fp, "w", encoding="utf-8") as fh:
            fh.write(out_df.to_latex(index=False, float_format="%.4f", na_rep="NA"))
        print("[B] LaTeX saved to:", tex_fp)
    except Exception as e:
        print("[B] Could not write LaTeX:", e)
    print("[B] CSV saved to:", csv_fp)
    return csv_fp, tex_fp, out_df

# ------------------ CLI & main ------------------

def parse_args():
    p = argparse.ArgumentParser(description="QE posthoc pipeline: A plots, B table, C recompute FDR, D bootstrap CIs")
    p.add_argument("--pairwise", type=str, default=PAIRWISE_FP_DEFAULT, help="Pairwise enriched CSV (default uses: " + PAIRWISE_FP_DEFAULT + ")")
    p.add_argument("--descriptives", type=str, default=DESCRIPTIVES_FP_DEFAULT, help="Descriptives CSV (metric x group means)")
    p.add_argument("--wide", type=str, default=WIDE_FP_DEFAULT, help="Wide per-pair metrics (time_stamp rows)")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="Metric bases to process")
    p.add_argument("--do", nargs="+", choices=["A","B","C","D","all"], default=["all"], help="Which actions to run")
    p.add_argument("--out", type=str, default=str(OUT_DIR), help="Output folder")
    p.add_argument("--nboot", type=int, default=2000, help="Bootstrap resamples")
    return p.parse_args()

def main():
    args = parse_args()
    out_folder = Path(args.out); out_folder.mkdir(parents=True, exist_ok=True)

    # load pairwise:
    pairwise_df = pd.read_csv(args.pairwise, engine="python", encoding="utf-8")
    # ensure p_fdr present (safe)
    if "p_fdr_bh" not in pairwise_df.columns:
        pairwise_df, pcol = recompute_fdr(pairwise_df)
        # save copy
        pairwise_df.to_csv(Path(out_folder)/"pairwise_with_recomputed_fdr_initial.csv", index=False)

    run_all = ("all" in args.do)
    doA = run_all or ("A" in args.do)
    doB = run_all or ("B" in args.do)
    doC = run_all or ("C" in args.do)
    doD = run_all or ("D" in args.do)

    # C first if requested explicitly
    if doC:
        pairwise_df, pcol, recomputed_fp = action_C_recompute_fdr(args.pairwise, out_fp=Path(out_folder)/"pairwise_recomputed_fdr.csv")

    # D: bootstrap CIs (depends on pairwise_df and wide)
    boot_fp = None
    if doD:
        boot_fp = action_D_bootstrap(pairwise_df, args.wide, args.metrics, out_folder, n_boot=args.nboot)

    # A: generate plots for significant contrasts
    if doA:
        made = action_A_plots(pairwise_df, args.wide, args.metrics, Path(out_folder)/"plots")
        print("[A] Created", len(made), "plots")

    # B: produce consolidated table (merge with bootstrap if computed)
    if doB:
        csv_fp, tex_fp, out_df = action_B_table(pairwise_df, args.descriptives, bootstrap_fp=boot_fp, out_folder=out_folder)

    print("Pipeline finished. Outputs in:", out_folder)

if __name__ == "__main__":
    main()
