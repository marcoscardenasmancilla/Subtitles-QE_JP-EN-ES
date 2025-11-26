#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate violin + boxplots per metric and group
Input: .\data\metrics_wide_strict_allmetrics.csv
Output: .\qe_plots\*.png  and .\qe_plots\descriptives_by_metric_group.csv
Dependencies: pandas numpy matplotlib seaborn
pip install pandas numpy matplotlib seaborn
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Config
WIDE_PATH = ".\qe_outputs\metrics_wide_strict_allmetrics.csv"
OUT_DIR = Path(".\qe_outputs\qe_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set(style="whitegrid")

# Base metrics we want to visualize (adjust if necessary)
METRICS_BASE = [
    "sbert_jap_en", "sbert_jap_sp",
    "comet_jap_en", "comet_jap_sp",
    "comet_jap_en_norm", "sbert_jap_en_norm", "qe_mix_en",
    "comet_jap_sp_norm", "sbert_jap_sp_norm", "qe_mix_sp"
]

# 1. Read wide
df = pd.read_csv(WIDE_PATH, engine="python", encoding="utf-8", on_bad_lines="warn")
print("Wide shape:", df.shape)
if "time_stamp" not in df.columns:
    # if it doesn't exist, take the first column as id
    df = df.rename(columns={df.columns[0]: "time_stamp"})
print("Columns sample:", df.columns[:10].tolist())

# 2. Detect groups (parsing columns with pattern <metric>__<group>)
groups = sorted({c.rsplit("__",1)[1] for c in df.columns if "__" in c})
print("Groups detected:", groups)

# 3. Create long table with rows (time_stamp, metric, group, value)
rows = []
for m in METRICS_BASE:
    for g in groups:
        col = f"{m}__{g}"
        if col in df.columns:
            ser = df[["time_stamp", col]].copy()
            ser = ser.rename(columns={col: "value"})
            ser["metric"] = m
            ser["group"] = g
            rows.append(ser)
        else:
            # warn if the column is missing
            print(f"[WARN] columna no encontrada: {col}  (se omitirá esta combinación)")
if not rows:
    raise RuntimeError("No se encontraron columnas para las métricas base solicitadas en el wide file.")
long = pd.concat(rows, ignore_index=True)
# convert value to float
long["value"] = pd.to_numeric(long["value"], errors="coerce")

# 4. Descriptive statistics by metric and group
desc = long.groupby(["metric","group"]).value.agg(["count","mean","std","median","min","max"]).reset_index()
desc.to_csv(OUT_DIR / "descriptives_by_metric_group.csv", index=False)
print("Descriptivos guardados en:", OUT_DIR / "descriptives_by_metric_group.csv")

# 5. Generate individual plots: violin + overlaid boxplot + stripplot
for metric in sorted(long["metric"].unique()):
    dfm = long[long["metric"] == metric].dropna(subset=["value"])
    if dfm.empty:
        print(f"[INFO] Métrica {metric} sin valores -> salto")
        continue

    plt.figure(figsize=(8,5))
    # violin
    sns.violinplot(data=dfm, x="group", y="value", inner=None, cut=0)
    # narrow boxplot on top
    sns.boxplot(data=dfm, x="group", y="value", width=0.12, showcaps=True,
                boxprops={'zorder':2, 'facecolor':'white'}, showfliers=False)
    # individual points
    sns.stripplot(data=dfm, x="group", y="value", color="k", size=2, alpha=0.35, jitter=True)
    plt.title(f"{metric} — distribución por grupo")
    plt.xlabel("Grupo")
    plt.ylabel(metric)
    plt.tight_layout()
    out_png = OUT_DIR / f"{metric}_violin_boxplot.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved:", out_png)

# 6. (Optional) Create figure with subplots for multiple metrics together (example 3x4)
metrics_to_plot = sorted(long["metric"].unique())
n = len(metrics_to_plot)
cols = 3
rows = int(np.ceil(n/cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
axes = axes.flatten()
for ax, metric in zip(axes, metrics_to_plot):
    dfm = long[long["metric"] == metric].dropna(subset=["value"])
    if dfm.empty:
        ax.axis("off"); continue
    sns.violinplot(data=dfm, x="group", y="value", inner=None, cut=0, ax=ax)
    sns.boxplot(data=dfm, x="group", y="value", width=0.12, showcaps=True,
                boxprops={'zorder':2,'facecolor':'white'}, showfliers=False, ax=ax)
    sns.stripplot(data=dfm, x="group", y="value", color="k", size=1.5, alpha=0.25, jitter=True, ax=ax)
    ax.set_title(metric)
    ax.set_xlabel("")
    ax.set_ylabel("")
plt.tight_layout()
combined_png = OUT_DIR / "combined_metrics_violin_boxplots.png"
plt.savefig(combined_png, dpi=200)
plt.close()
print("Saved combined plot:", combined_png)

print("Done. Plots in:", OUT_DIR.resolve())
