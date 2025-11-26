#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_wide_matrix.py
Generates 'wide' matrices by time_stamp with columns <metric>__<group>
Default output in .\data:
 - metrics_wide_by_timestamp.csv
 - metrics_wide_valid_minonepergroup.csv
 - metrics_wide_strict_allmetrics.csv

Usage:
 python make_wide_matrix.py --input .\metrics_all_groups.csv

Dependencies:
 pip install pandas numpy
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

DEFAULT_INPUT = ".\qe_outputs\metrics_all_groups.csv"
OUT_DIR = ".\qe_outputs"

# Default list of metrics to pivot (adjust if necessary)
DEFAULT_METRICS = [
    "sbert_jap_en","sbert_jap_sp",
    "comet_jap_en","comet_jap_sp",
    "comet_jap_en_norm","sbert_jap_en_norm","qe_mix_en",
    "comet_jap_sp_norm","sbert_jap_sp_norm","qe_mix_sp"
]

def robust_read_csv(fp):
    """Read CSV using engine 'python' and on_bad_lines='warn' to tolerate problematic rows."""
    print(f"[INFO] Leyendo archivo: {fp}")
    df = pd.read_csv(fp, engine="python", on_bad_lines="warn", encoding="utf-8")
    print(f"[INFO] Shape leída: {df.shape}")
    return df

def detect_time_and_group(df):
    cols = df.columns.str.lower().tolist()
    # common candidates
    time_candidates = [c for c in df.columns if c.lower() in ("time_stamp","timestamp","time","timecode","ts")]
    group_candidates = [c for c in df.columns if c.lower() in ("group","group_type","variant","source","system","origin")]
    # heuristic fallback
    if not time_candidates:
        time_candidates = [c for c in df.columns if "time" in c.lower() and "zone" not in c.lower()]
    if not group_candidates:
        group_candidates = [c for c in df.columns if "group" in c.lower() or "variant" in c.lower() or "source" in c.lower()]
    if not time_candidates or not group_candidates:
        raise ValueError("No se pudo detectar automáticamente columnas 'time' o 'group'. Columnas disponibles: " + ", ".join(df.columns.tolist()))
    time_col = time_candidates[0]
    group_col = group_candidates[0]
    print(f"[INFO] time_col -> '{time_col}', group_col -> '{group_col}'")
    return time_col, group_col

def find_metric_columns(df, desired_metrics):
    """
    Map each desired_metric to the actual column in the df.
    If an exact match exists, use it; otherwise look for token matches (relaxed).
    Returns dict: desired_metric -> actual_column_name (or None).
    """
    cols = df.columns.tolist()
    mapping = {}
    for desired in desired_metrics:
        if desired in cols:
            mapping[desired] = desired
            continue
        # tokens: decompose
        tokens = [t for t in desired.split("_") if t]
        # first search: all tokens appear (strict)
        candidates = [c for c in cols if all(tok in c.lower() for tok in tokens)]
        if candidates:
            mapping[desired] = candidates[0]
            continue
        # second search: at least one token matches (loose)
        candidates2 = [c for c in cols if any(tok in c.lower() for tok in tokens)]
        mapping[desired] = candidates2[0] if candidates2 else None
    return mapping

def pivot_to_wide(df, time_col, group_col, metric_map):
    """
    Generate the 'wide' dataframe by concatenating pivots for each metric m:
      pivot: index=time_col, columns=group_col, values=m
    Resulting columns: <metric_actual>__<group>
    """
    pivots = []
    used_metrics = []
    for desired, actual_col in metric_map.items():
        if actual_col is None:
            print(f"[WARN] Métrica {desired} no encontrada -> se omitirá.")
            continue
        if actual_col not in df.columns:
            print(f"[WARN] Columna mapeada {actual_col} no está en el DataFrame -> omitir.")
            continue
        used_metrics.append(actual_col)
        tmp = df[[time_col, group_col, actual_col]].copy()
        # pivot: index=time_col, columns=group, values=actual_col
        pivot = tmp.pivot(index=time_col, columns=group_col, values=actual_col)
        # rename columns: <actual_col>__<group>
        pivot.columns = [f"{actual_col}__{g}" for g in pivot.columns]
        pivots.append(pivot)
        print(f"[INFO] Pivot creada para {actual_col}, columnas: {list(pivot.columns)[:5]}")
    if not pivots:
        raise ValueError("No se generaron pivots: no se encontraron métricas válidas.")
    wide = pd.concat(pivots, axis=1)
    wide = wide.reset_index().rename(columns={wide.index.name or 0: time_col})
    # rename index column to 'time_stamp' if necessary
    wide = wide.reset_index().rename(columns={"index": time_col})
    # ensure that there is a 'time_stamp' column with that explicit name
    if time_col not in wide.columns:
        # if the first column is the time index, force rename
        wide = wide.rename(columns={wide.columns[0]: time_col})
    return wide, used_metrics

def subset_minone_per_group(wide, metrics_actual, groups):
    """
    Keep rows (time_stamp) where, for each group, there exists at least ONE non-null metric among metrics_actual.
    metrics_actual: list of *actual* column names (not desired names) that were used when pivoting.
    """
    # build list of columns per metric x group
    # but we built columns as "<actual_metric>__<group>"
    mask_rows = []
    for idx, row in wide.iterrows():
        ok = True
        for g in groups:
            has_any = False
            for m in metrics_actual:
                col = f"{m}__{g}"
                if col in wide.columns and pd.notna(row[col]):
                    has_any = True
                    break
            if not has_any:
                ok = False
                break
        mask_rows.append(ok)
    return wide.loc[mask_rows].reset_index(drop=True)

def subset_strict_allmetrics(wide, metrics_actual, groups):
    """
    Keep rows where *all* metrics (metrics_actual) are present and non-null for each group.
    """
    required_cols = []
    for g in groups:
        for m in metrics_actual:
            col = f"{m}__{g}"
            if col not in wide.columns:
                raise KeyError(f"Columna requerida no encontrada en wide: {col}")
            required_cols.append(col)
    mask = wide[required_cols].notna().all(axis=1)
    return wide.loc[mask].reset_index(drop=True)

def main(args):
    input_fp = Path(args.input) if args.input else Path(DEFAULT_INPUT)
    if not input_fp.exists():
        raise FileNotFoundError(f"Archivo de entrada no encontrado: {input_fp}")
    df = robust_read_csv(input_fp)
    time_col, group_col = detect_time_and_group(df)
    # detect groups
    groups = sorted(df[group_col].dropna().unique().tolist())
    print(f"[INFO] Grupos detectados: {groups}")

    # determine metrics to use: from CLI or DEFAULT_METRICS
    metrics = args.metrics if args.metrics else DEFAULT_METRICS
    print(f"[INFO] Métricas deseadas: {metrics}")

    metric_map = find_metric_columns(df, metrics)
    print("[INFO] Mapeo de métricas (deseada -> columna encontrada):")
    for k,v in metric_map.items():
        print(f"  {k} -> {v}")

    # pivot
    wide, used_metrics = pivot_to_wide(df, time_col, group_col, metric_map)
    # wide now has index column named time_col as first column
    # ensure time_col is explicitly named "time_stamp" for consistency with downstream
    if time_col != "time_stamp":
        wide = wide.rename(columns={time_col: "time_stamp"})
        time_col = "time_stamp"

    # save wide_all
    out_all = Path(OUT_DIR) / "metrics_wide_by_timestamp.csv"
    out_all.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_all, index=False, encoding="utf-8")
    print(f"[OK] Saved wide all -> {out_all} (rows: {len(wide)})")

    # generate valid minone per group
    wide_valid = subset_minone_per_group(wide, used_metrics, groups)
    out_valid = Path(OUT_DIR) / "metrics_wide_valid_minonepergroup.csv"
    wide_valid.to_csv(out_valid, index=False, encoding="utf-8")
    print(f"[OK] Saved valid (min-one-per-group) -> {out_valid} (rows: {len(wide_valid)})")

    # generate strict allmetrics per group (try; if missing metric columns will raise)
    try:
        wide_strict = subset_strict_allmetrics(wide, used_metrics, groups)
        out_strict = Path(OUT_DIR) / "metrics_wide_strict_allmetrics.csv"
        wide_strict.to_csv(out_strict, index=False, encoding="utf-8")
        print(f"[OK] Saved strict (all-metrics per group) -> {out_strict} (rows: {len(wide_strict)})")
    except KeyError as e:
        print("[WARN] No fue posible generar strict_allmetrics:", e)
        out_strict = None

    # summary
    summary = {
        "input_rows": len(df),
        "distinct_time_stamps": wide["time_stamp"].nunique() if "time_stamp" in wide.columns else wide.iloc[:,0].nunique(),
        "groups_detected": groups,
        "metrics_used_actual": used_metrics,
        "wide_all": str(out_all),
        "wide_valid": str(out_valid),
        "wide_strict": str(out_strict) if out_strict else None
    }
    print("[SUMMARY]")
    for k,v in summary.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Genera matriz wide por time_stamp para comparar métricas por grupo.")
    p.add_argument("--input", type=str, default=DEFAULT_INPUT, help="CSV de entrada (metrics_all_groups.csv por defecto)")
    p.add_argument("--metrics", nargs="+", help="Lista de métricas base a pivotar (ej: sbert_jap_en comet_jap_en ...).")
    args = p.parse_args()
    main(args)

