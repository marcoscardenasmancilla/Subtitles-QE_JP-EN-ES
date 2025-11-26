#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_groups_qe.py
Pipeline local para comparar calidad de traducción entre grupos (official, fansub, ai)
Entrada por defecto:
 - /mnt/data/metrics_wide_strict_allmetrics.csv  (archivo wide ya creado)
Si no existe, el script intentará cargar /mnt/data/metrics_all_groups.csv y pivotar a wide.
Salida por defecto:
 - ./qe_compare_outputs/ (CSV, plots, report HTML)

Dependencias:
pip install pandas numpy scipy statsmodels matplotlib seaborn jinja2

Uso:
python compare_groups_qe.py --input /mnt/data/metrics_wide_strict_allmetrics.csv
"""
import os
import sys
import argparse
import math
from pathlib import Path
import base64
import io
from datetime import datetime

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

# ---------------- CONFIG DEFAULTS ----------------
DEFAULT_WIDE = r"C:\Users\Marcos\Desktop\2025-2\Tesis_Samir\qe_outputs\metrics_wide_strict_allmetrics.csv"
FALLBACK_RAW = r"C:\Users\Marcos\Desktop\2025-2\Tesis_Samir\qe_outputs\metrics_all_groups.csv"
OUT_ROOT = r"C:\Users\Marcos\Desktop\2025-2\Tesis_Samir\qe_outputs\qe_compare_outputs"
ALPHA = 0.05

# default metrics to analyze (must match column base names in the wide file)
DEFAULT_METRICS = [
    "sbert_jap_en", "sbert_jap_sp",
    "comet_jap_en", "comet_jap_sp",
    "comet_jap_en_norm", "sbert_jap_en_norm", "qe_mix_en",
    "comet_jap_sp_norm", "sbert_jap_sp_norm", "qe_mix_sp"
]

# pairs to compare (will auto-detect groups present)
DEFAULT_PAIRS = None  # will be set after reading groups

# ---------------- Helpers ----------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)

def read_wide_or_pivot(input_path):
    """
    If input_path exists: read it as wide.
    Else: try to read metrics_all_groups.csv and pivot to wide.
    Returns dataframe wide.
    """
    if Path(input_path).exists():
        print(f"[INFO] Reading wide input: {input_path}")
        wide = pd.read_csv(input_path, engine="python", encoding="utf-8")
        return wide
    # fallback pivot
    raw_fp = Path(FALLBACK_RAW)
    if not raw_fp.exists():
        raise FileNotFoundError(f"Neither {input_path} nor fallback {FALLBACK_RAW} were found.")
    print(f"[WARN] Wide file not found. Pivoting from raw file: {FALLBACK_RAW}")
    df = pd.read_csv(raw_fp, engine="python", encoding="utf-8")
    if "time_stamp" not in df.columns or "group" not in df.columns:
        raise ValueError("Fallback raw file missing 'time_stamp' or 'group' columns.")
    # detect metric columns (pattern)
    metric_candidates = [c for c in df.columns if any(tok in c.lower() for tok in ["sbert","comet","qe_mix", "qe-mix", "qe"]) ]
    # pivot each metric into metric__group columns
    time_col = "time_stamp"
    groups = sorted(df["group"].dropna().unique().tolist())
    wide_frames = []
    for m in metric_candidates:
        tmp = df[[time_col, "group", m]].copy()
        pivot = tmp.pivot(index=time_col, columns="group", values=m)
        pivot.columns = [f"{m}__{g}" for g in pivot.columns]
        wide_frames.append(pivot)
    if not wide_frames:
        raise ValueError("No metric candidates found to pivot from raw file.")
    wide = pd.concat(wide_frames, axis=1).reset_index().rename(columns={ "index": time_col })
    # ensure time_stamp column name
    if "time_stamp" not in wide.columns:
        # maybe first column is the index name
        wide = wide.rename(columns={wide.columns[0]: "time_stamp"})
    return wide

def detect_groups_from_wide(wide):
    # detect groups by parsing columns pattern <metric>__<group>
    groups = set()
    for c in wide.columns:
        if "__" in c:
            try:
                metric, grp = c.rsplit("__", 1)
                groups.add(grp)
            except:
                continue
    groups = sorted(list(groups))
    return groups

def build_strict_subset(wide, metrics, groups):
    """
    Strict subset: require that for each time_stamp and for all groups,
    the metrics specified (core metrics) are non-null.
    """
    required_cols = []
    for m in metrics:
        for g in groups:
            col = f"{m}__{g}"
            if col in wide.columns:
                required_cols.append(col)
            else:
                raise KeyError(f"Expected column {col} not found in wide dataframe.")
    mask = wide[required_cols].notna().all(axis=1)
    return wide.loc[mask].reset_index(drop=True)

def build_minone_subset(wide, metrics, groups):
    """
    Looser subset: require that for each time_stamp and for each group,
    at least one metric among provided metrics is non-null.
    """
    rows = []
    for idx, row in wide.iterrows():
        ok = True
        for g in groups:
            has = False
            for m in metrics:
                col = f"{m}__{g}"
                if col in wide.columns:
                    if pd.notna(row[col]):
                        has = True
                        break
            if not has:
                ok = False
                break
        if ok:
            rows.append(True)
        else:
            rows.append(False)
    mask = pd.Series(rows, index=wide.index)
    return wide.loc[mask].reset_index(drop=True)

def paired_test_and_effect(x, y, alpha=ALPHA):
    """
    For two arrays x,y (paired), compute normality of differences (Shapiro),
    then use paired t-test if normal, else Wilcoxon signed-rank test.
    Returns dict with test name, stat, pvalue, effect_size (Cohen's d for paired or r for Wilcoxon), and normal_flag.
    """
    res = {"test": None, "stat": None, "pvalue": None, "effect_size": None, "method": None, "n": 0}
    # drop nan pairs
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x2 = np.array(x[mask]); y2 = np.array(y[mask])
    n = len(x2)
    res["n"] = n
    if n < 3:
        res.update({"method":"insufficient_n"})
        return res
    diff = x2 - y2
    # normality test on diff
    try:
        sh_w, sh_p = stats.shapiro(diff)
    except Exception:
        # fallback to normal if shapiro errors
        sh_w, sh_p = (None, 1.0)
    normal = (sh_p is not None and sh_p > alpha)
    res["normal_shapiro_p"] = sh_p
    if normal:
        # paired t-test
        t_stat, p_val = stats.ttest_rel(x2, y2, nan_policy='omit')
        # Cohen's d for paired: mean(diff) / std(diff)
        md = np.nanmean(diff)
        sd = np.nanstd(diff, ddof=1)
        d = md / sd if sd and not np.isnan(sd) else np.nan
        res.update({"test":"paired_ttest","stat": float(t_stat), "pvalue": float(p_val), "effect_size": float(d), "method":"t-test paired"})
    else:
        # Wilcoxon signed-rank (two-sided)
        try:
            w_stat, p_val = stats.wilcoxon(x2, y2, zero_method='wilcox', correction=False)
            # effect size r = z / sqrt(N)
            # approximate z from w: use normal approximation: z = (w - n(n+1)/4)/sqrt(n(n+1)(2n+1)/24)
            n_ = n
            mean_w = n_*(n_+1)/4.0
            sd_w = math.sqrt(n_*(n_+1)*(2*n_+1)/24.0)
            z = (w_stat - mean_w) / sd_w if sd_w>0 else 0.0
            r = abs(z) / math.sqrt(n_) if n_>0 else np.nan
            res.update({"test":"wilcoxon","stat": float(w_stat), "pvalue": float(p_val), "effect_size": float(r), "method":"wilcoxon (r)"})
        except Exception as e:
            # fallback
            res.update({"test":"wilcoxon_failed","stat":None,"pvalue":None,"effect_size":None,"method":"wilcoxon_failed","error":str(e)})
    return res

def descriptive_stats_by_group(wide_subset, metric, groups):
    """
    Return a dataframe with mean, sd, median, n per group for a given metric (base name).
    """
    rows = []
    for g in groups:
        col = f"{metric}__{g}"
        if col not in wide_subset.columns:
            rows.append({"group":g,"metric":metric,"n":0,"mean":np.nan,"sd":np.nan,"median":np.nan})
            continue
        series = wide_subset[col].dropna().astype(float)
        rows.append({
            "group": g,
            "metric": metric,
            "n": int(series.count()),
            "mean": float(series.mean()) if len(series)>0 else np.nan,
            "sd": float(series.std(ddof=1)) if len(series)>1 else np.nan,
            "median": float(series.median()) if len(series)>0 else np.nan
        })
    return pd.DataFrame(rows)

# ---------------- Pipeline Core ----------------
def run_pipeline(input_path=None, metrics=DEFAULT_METRICS, mode="strict", out_root=OUT_ROOT, alpha=ALPHA):
    out_root = ensure_dir(out_root)
    # 1. Read wide or pivot raw
    wide = read_wide_or_pivot(input_path if input_path else DEFAULT_WIDE)

    # 2. detect groups
    groups = detect_groups_from_wide(wide)
    if len(groups) < 2:
        raise ValueError("Not enough groups detected in wide file. Found groups: " + ", ".join(groups))
    print(f"[INFO] Groups detected: {groups}")

    # 3. Validate metrics presence
    missing = []
    for m in metrics:
        for g in groups:
            if f"{m}__{g}" not in wide.columns:
                missing.append(f"{m}__{g}")
    if missing:
        print(f"[WARN] Missing metric columns: {len(missing)} (first 10 shown): {missing[:10]}")
        # Continue but user may want to adjust metric list. We will attempt to proceed using available columns only.
    # 4. subset selection
    if mode == "strict":
        try:
            subset = build_strict_subset(wide, metrics, groups)
        except KeyError as e:
            print("[ERROR] Strict subset build failed:", e)
            print("Falling back to minone mode.")
            subset = build_minone_subset(wide, metrics, groups)
            mode = "minone"
    else:
        subset = build_minone_subset(wide, metrics, groups)
    print(f"[INFO] Subset mode: {mode} -> {len(subset)} time_stamp rows selected")

    # 5. Prepare outputs
    ensure_dir(out_root)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    stats_folder = ensure_dir(os.path.join(out_root, "stats"))
    plots_folder = ensure_dir(os.path.join(out_root, "plots"))

    # 6. Compute descriptive stats per metric per group
    desc_frames = []
    for m in metrics:
        try:
            df_desc = descriptive_stats_by_group(subset, m, groups)
            desc_frames.append(df_desc)
        except Exception as e:
            print(f"[WARN] Descriptive for {m} failed:", e)
    if desc_frames:
        desc_all = pd.concat(desc_frames, ignore_index=True)
        desc_csv = os.path.join(stats_folder, f"descriptives_{timestamp}.csv")
        desc_all.to_csv(desc_csv, index=False, encoding="utf-8")
        print("[OK] Saved descriptives:", desc_csv)

    # 7. Pairwise comparisons for all metric/group pairs
    from itertools import combinations
    pair_results = []
    for m in metrics:
        for g1, g2 in combinations(groups, 2):
            col1 = f"{m}__{g1}"
            col2 = f"{m}__{g2}"
            if col1 not in subset.columns or col2 not in subset.columns:
                print(f"[SKIP] Missing columns for pair {col1} vs {col2}; skipping.")
                continue
            x = subset[col1].astype(float).to_numpy()
            y = subset[col2].astype(float).to_numpy()
            res = paired_test_and_effect(x, y, alpha=alpha)
            row = {
                "metric": m,
                "group1": g1,
                "group2": g2,
                "n_pairs": int(res.get("n", 0)),
                "test": res.get("test"),
                "method": res.get("method"),
                "stat": res.get("stat"),
                "pvalue": res.get("pvalue"),
                "normal_shapiro_p": res.get("normal_shapiro_p"),
                "effect_size": res.get("effect_size"),
            }
            pair_results.append(row)
    pair_df = pd.DataFrame(pair_results)
    pair_csv = os.path.join(stats_folder, f"pairwise_tests_{timestamp}.csv")
    pair_df.to_csv(pair_csv, index=False, encoding="utf-8")
    print("[OK] Saved pairwise test results:", pair_csv)

    # 8. Generate plots
    # Boxplots per metric across groups
    for m in metrics:
        # build long table for seaborn
        frames = []
        for g in groups:
            col = f"{m}__{g}"
            if col not in subset.columns:
                continue
            ser = subset[[ "time_stamp", col ]].rename(columns={col:"value"})
            ser["group"] = g
            ser["metric"] = m
            frames.append(ser)
        if not frames:
            continue
        long = pd.concat(frames, ignore_index=True)
        long = long.dropna(subset=["value"])
        if long.empty:
            continue
        plt.figure(figsize=(8,4))
        sns.boxplot(data=long, x="group", y="value")
        sns.stripplot(data=long, x="group", y="value", color="0.2", alpha=0.4, jitter=True, size=3)
        plt.title(f"{m} — distribution por grupo")
        plt.ylabel(m)
        plt.tight_layout()
        plot_fp = os.path.join(plots_folder, f"{m}_boxplot_{timestamp}.png")
        plt.savefig(plot_fp, dpi=200)
        plt.close()

        # Paired line plot: draw lines for each time_stamp for pair comparisons (only for first pair of groups)
        # We'll generate pairwise line plots for each combination
        for g1, g2 in combinations(groups, 2):
            col1 = f"{m}__{g1}"
            col2 = f"{m}__{g2}"
            if col1 not in subset.columns or col2 not in subset.columns:
                continue
            df_plot = subset[[ "time_stamp", col1, col2 ]].dropna()
            if df_plot.shape[0] == 0:
                continue
            plt.figure(figsize=(8,4))
            # sample to max 200 lines to avoid overplotting
            sample = df_plot.sample(n=min(len(df_plot), 200), random_state=42)
            for _, r in sample.iterrows():
                plt.plot([0,1],[r[col1], r[col2]], color="gray", alpha=0.4)
            # overlay means
            mean1 = df_plot[col1].mean()
            mean2 = df_plot[col2].mean()
            plt.scatter([0,1],[mean1, mean2], color="red", zorder=5, s=60)
            plt.xlim(-0.5,1.5)
            plt.xticks([0,1],[g1,g2])
            plt.title(f"Paired lines: {m} — {g1} vs {g2}")
            plt.ylabel(m)
            plt.tight_layout()
            plot_fp2 = os.path.join(plots_folder, f"{m}_paired_{g1}_vs_{g2}_{timestamp}.png")
            plt.savefig(plot_fp2, dpi=200)
            plt.close()

    print("[OK] Plots saved to:", plots_folder)

    # 9. Build HTML report (basic)
    report_fp = os.path.join(out_root, f"qe_compare_report_{timestamp}.html")
    html_template = """
    <!doctype html><html><head><meta charset="utf-8"><title>QE compare report</title></head><body>
    <h1>QE Compare Report</h1>
    <p>Generated: {{ generated }}</p>
    <h2>Settings</h2>
    <ul><li>Input (wide): {{ input }}</li><li>Mode: {{ mode }}</li><li>Groups: {{ groups }}</li><li>Metrics: {{ metrics }}</li></ul>
    <h2>Descriptives (summary)</h2>
    {{ desc_table | safe }}
    <h2>Pairwise test summary</h2>
    {{ pair_table | safe }}
    <h2>Plots (boxplots and paired)</h2>
    {% for p in plots %}
        <h3>{{ p.title }}</h3>
        <img src="data:image/png;base64,{{ p.b64 }}" style="max-width:900px;"/>
    {% endfor %}
    </body></html>
    """
    # prepare summary html snippets
    desc_html = ""
    if 'desc_all' in locals():
        desc_html = desc_all.pivot(index="metric", columns="group")[["mean","sd","n"]].to_html()
    pair_html = pair_df.to_html(index=False)

    # embed top N plots
    plots = []
    # list png files in plots_folder, take up to 12
    pngs = sorted(Path(plots_folder).glob("*.png"))[:12]
    for p in pngs:
        with open(p, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("utf-8")
        plots.append({"title": p.name, "b64": b64})

    tpl = Template(html_template)
    html = tpl.render(generated=datetime.utcnow().isoformat()+"Z",
                      input=(input_path if input_path else DEFAULT_WIDE),
                      mode=mode,
                      groups=",".join(groups),
                      metrics=",".join(metrics),
                      desc_table=desc_html,
                      pair_table=pair_html,
                      plots=plots)
    with open(report_fp, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("[OK] Report saved:", report_fp)

    # 10. Save subset used and wide used
    subset_fp = os.path.join(out_root, f"subset_used_{timestamp}.csv")
    wide_fp = os.path.join(out_root, f"wide_used_{timestamp}.csv")
    subset.to_csv(subset_fp, index=False, encoding="utf-8")
    wide.to_csv(wide_fp, index=False, encoding="utf-8")
    print("[OK] Saved subset and wide copies:", subset_fp, wide_fp)

    print("[DONE] Pipeline finished. Outputs in:", out_root)
    return {
        "out_root": out_root,
        "desc_csv": desc_csv if 'desc_csv' in locals() else None,
        "pair_csv": pair_csv,
        "plots_folder": plots_folder,
        "report": report_fp,
        "subset": subset_fp,
        "wide_used": wide_fp
    }

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Compare QE metrics across groups (local pipeline)")
    p.add_argument("--input", type=str, default=DEFAULT_WIDE, help="Path to wide CSV (default: metrics_wide_strict_allmetrics.csv). If missing, will pivot from metrics_all_groups.csv")
    p.add_argument("--mode", type=str, default="strict", choices=["strict","minone"], help="Subset mode: strict (all metrics present) or minone (at least one per group)")
    p.add_argument("--out", type=str, default=OUT_ROOT, help="Output root folder")
    p.add_argument("--metrics", type=str, nargs="+", default=DEFAULT_METRICS, help="List of metric base names to compare")
    p.add_argument("--alpha", type=float, default=ALPHA, help="Alpha for tests")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    results = run_pipeline(input_path=args.input, metrics=args.metrics, mode=args.mode, out_root=args.out, alpha=args.alpha)
    print("Outputs:", results)
