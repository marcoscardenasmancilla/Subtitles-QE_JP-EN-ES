#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QE_mix + flagging + HTML report per-group
Adapted for local dataset file. Edit DATASET_PATH if needed.

Outputs:
 - ./qe_outputs/group__{group}/metrics_group_{group}.csv
 - ./qe_outputs/group__{group}/report_group_{group}.html
 - ./qe_outputs/metrics_all_groups.csv

Dependencies:
 pip install pandas numpy matplotlib sentence-transformers comet jinja2 scipy
(If you don't want COMET/SBERT set COMET_QE_ENABLE = False / SBERT_ENABLE = False)
"""

import os, math, re, unicodedata, base64, io
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jinja2 import Template
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# --------- CONFIG ----------
DATASET_PATH = ".\dataset.csv"   # <-- LOCAL path to CSV
OUT_ROOT = ".\qe_outputs"
CHAR_N = 3

# QE_mix weights (default)
W_COMET = 0.6
W_SBERT = 0.4

# Feature toggles
SBERT_ENABLE = True
COMET_QE_ENABLE = True

# SBERT / COMET settings
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SBERT_BATCH_SIZE = 64
COMET_QE_MODEL = "Unbabel/wmt20-comet-qe-da"
COMET_QE_BATCH_SIZE = 8

# Flagging thresholds (editable)
TH_QE_MIX_LOW = 0.25
TH_LENGTH_RATIO_LOW = 0.5
TH_LENGTH_RATIO_HIGH = 1.8
TH_PUNC_JACCARD_LOW = 0.3
TH_DIGIT_MATCH = True
TH_CHAR_N_HIGH_SIM = 0.95
TH_DISCREPANCY = 0.5
# ----------------------------

# ---- Helpers ----
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def strip_diacritics(s: str) -> str:
    if not isinstance(s, str): return ""
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def char_ngrams(s: str, n: int = 3):
    if not isinstance(s, str): return []
    s = strip_diacritics(s.lower())
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return [s[i:i+n] for i in range(len(s)-n+1)] if len(s) >= n else []

def cosine_from_ngrams(a: str, b: str, n: int = 3) -> float:
    na, nb = Counter(char_ngrams(a, n)), Counter(char_ngrams(b, n))
    if not na or not nb: return 0.0
    dot = sum(na[g]*nb.get(g,0) for g in na)
    norm_a = math.sqrt(sum(v*v for v in na.values()))
    norm_b = math.sqrt(sum(v*v for v in nb.values()))
    return dot/(norm_a*norm_b) if norm_a and norm_b else 0.0

def punctuation_jaccard(a: str, b: str):
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    pa = set(re.findall(r"[^\w\s]", a))
    pb = set(re.findall(r"[^\w\s]", b))
    if not pa and not pb: return 1.0
    inter = pa.intersection(pb)
    union = pa.union(pb)
    return len(inter) / len(union) if union else 1.0

def digit_match(a: str, b: str):
    da = re.findall(r"\d+", a or "")
    db = re.findall(r"\d+", b or "")
    return sorted(da) == sorted(db)

def length_ratio(a: str, b: str):
    la = len((a or "").split())
    lb = len((b or "").split())
    if la == 0: return np.nan
    return lb / la

# --- SBERT load (optional) ---
_HAS_SBERT = False
_sbert_model = None
if SBERT_ENABLE:
    try:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer(SBERT_MODEL)
        _HAS_SBERT = True
        print("[INFO] SBERT loaded:", SBERT_MODEL)
    except Exception as e:
        print("[WARN] SBERT not available:", e)
        _HAS_SBERT = False

def sbert_batch_cosine(src_texts, hyp_texts, batch_size=SBERT_BATCH_SIZE):
    if not _HAS_SBERT:
        return [np.nan]*len(src_texts)
    import torch
    model = _sbert_model
    model.max_seq_length = 512
    src_embs, hyp_embs = [], []
    for i in range(0, len(src_texts), batch_size):
        chunk = src_texts[i:i+batch_size]
        emb = model.encode(chunk, convert_to_tensor=True, normalize_embeddings=True)
        src_embs.append(emb)
    for i in range(0, len(hyp_texts), batch_size):
        chunk = hyp_texts[i:i+batch_size]
        emb = model.encode(chunk, convert_to_tensor=True, normalize_embeddings=True)
        hyp_embs.append(emb)
    src_all = torch.cat(src_embs, dim=0)
    hyp_all = torch.cat(hyp_embs, dim=0)
    sims = (src_all * hyp_all).sum(dim=1).cpu().numpy()
    return sims.tolist()

# --- COMET-QE load (optional) ---
_HAS_COMET = False
_comet_obj = None
if COMET_QE_ENABLE:
    try:
        from comet import download_model, load_from_checkpoint
        ckpt = download_model(COMET_QE_MODEL)
        _comet_obj = load_from_checkpoint(ckpt)
        _HAS_COMET = True
        print("[INFO] COMET-QE model ready:", COMET_QE_MODEL)
    except Exception as e:
        print("[WARN] COMET-QE not available or download failed:", e)
        _HAS_COMET = False

def comet_qe_score_batch(src_texts, hyp_texts, batch_size=COMET_QE_BATCH_SIZE):
    if not _HAS_COMET: return [np.nan] * len(src_texts)
    data = [{"src": s or "", "mt": t or ""} for s,t in zip(src_texts, hyp_texts)]
    try:
        out = _comet_obj.predict(data, batch_size=batch_size, gpus=0, progress_bar=False)
        if isinstance(out, dict) and "scores" in out:
            return [float(x) for x in out["scores"]]
        if isinstance(out, list):
            return [float(x.get("score", np.nan)) if isinstance(x, dict) else float(np.nan) for x in out]
    except Exception as e:
        print("[WARN] COMET predict error:", e)
    return [np.nan] * len(src_texts)

# --- Normalization helper ---
def minmax_series(s):
    s = np.array(s, dtype=float)
    if np.all(np.isnan(s)):
        return np.array([np.nan]*len(s))
    mn = np.nanmin(s)
    mx = np.nanmax(s)
    if mx == mn:
        return np.array([0.5 if not np.isnan(x) else np.nan for x in s])
    return (s - mn) / (mx - mn)

# --- HTML template ---
HTML_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>QE Report - {{ group }}</title>
<style>body{font-family:Arial;margin:20px}h1,h2{color:#2b6cb0}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:6px;font-size:13px}th{background:#f7fafc}.small{font-size:12px;color:#666}</style></head><body>
<h1>QE Report — group: {{ group }}</h1><p class="small">Generated: {{ generated_at }}</p>
<h2>Summary</h2><table><tr><th>Rows</th><td>{{ n_rows }}</td></tr><tr><th>Mean QE_mix</th><td>{{ mean_qe_mix }}</td></tr><tr><th>Median QE_mix</th><td>{{ median_qe_mix }}</td></tr><tr><th>Low-quality (%)</th><td>{{ pct_low_quality }}</td></tr></table>
<h2>Top flags (most frequent)</h2><table><tr><th>Flag</th><th>Count</th></tr>{% for f,c in flags %}<tr><td>{{ f }}</td><td>{{ c }}</td></tr>{% endfor %}</table>
<h2>Distribution</h2><p><img src="data:image/png;base64,{{ plot_qe }}" alt="qe_dist" /></p>
<h2>Sample rows (first 20)</h2>{{ sample_table | safe }}</body></html>"""

# --- Main group processor ---
def process_group(df_group, group_name, out_root):
    ensure_dir(out_root)
    gdf = df_group.copy().reset_index(drop=True)

    srcs = gdf["sub_jap"].fillna("").astype(str).tolist()
    hyp_en = gdf["sub_trad_en"].fillna("").astype(str).tolist() if "sub_trad_en" in gdf.columns else ["" for _ in srcs]
    hyp_sp = gdf["sub_trad_sp"].fillna("").astype(str).tolist() if "sub_trad_sp" in gdf.columns else ["" for _ in srcs]

    gdf["cosn_jap_en"] = [cosine_from_ngrams(a,b,CHAR_N) for a,b in zip(srcs,hyp_en)]
    gdf["cosn_jap_sp"] = [cosine_from_ngrams(a,b,CHAR_N) for a,b in zip(srcs,hyp_sp)]
    gdf["punc_jaccard_en"] = [punctuation_jaccard(a,b) for a,b in zip(srcs,hyp_en)]
    gdf["punc_jaccard_sp"] = [punctuation_jaccard(a,b) for a,b in zip(srcs,hyp_sp)]
    if TH_DIGIT_MATCH:
        gdf["digits_match_en"] = [int(digit_match(a,b)) for a,b in zip(srcs,hyp_en)]
        gdf["digits_match_sp"] = [int(digit_match(a,b)) for a,b in zip(srcs,hyp_sp)]
    gdf["len_ratio_en"] = [length_ratio(a,b) for a,b in zip(srcs,hyp_en)]
    gdf["len_ratio_sp"] = [length_ratio(a,b) for a,b in zip(srcs,hyp_sp)]

    if _HAS_SBERT:
        try:
            print(f"[INFO] Computing SBERT for group {group_name} (EN)")
            gdf["sbert_jap_en"] = sbert_batch_cosine(srcs, hyp_en)
            print(f"[INFO] Computing SBERT for group {group_name} (SP)")
            gdf["sbert_jap_sp"] = sbert_batch_cosine(srcs, hyp_sp)
        except Exception as e:
            print("[WARN] SBERT failed:", e)
            gdf["sbert_jap_en"] = np.nan; gdf["sbert_jap_sp"] = np.nan
    else:
        gdf["sbert_jap_en"] = np.nan; gdf["sbert_jap_sp"] = np.nan

    if _HAS_COMET:
        try:
            print(f"[INFO] Computing COMET-QE for group {group_name} (EN)")
            gdf["comet_jap_en"] = comet_qe_score_batch(srcs, hyp_en)
            print(f"[INFO] Computing COMET-QE for group {group_name} (SP)")
            gdf["comet_jap_sp"] = comet_qe_score_batch(srcs, hyp_sp)
        except Exception as e:
            print("[WARN] COMET failed:", e)
            gdf["comet_jap_en"] = np.nan; gdf["comet_jap_sp"] = np.nan
    else:
        gdf["comet_jap_en"] = np.nan; gdf["comet_jap_sp"] = np.nan

    for pair in [("comet_jap_en","sbert_jap_en","qe_mix_en"),
                 ("comet_jap_sp","sbert_jap_sp","qe_mix_sp")]:
        comet_col, sbert_col, qe_col = pair
        comet_norm = minmax_series(gdf[comet_col].values) if comet_col in gdf.columns else np.array([np.nan]*len(gdf))
        sbert_norm = minmax_series(gdf[sbert_col].values) if sbert_col in gdf.columns else np.array([np.nan]*len(gdf))
        gdf[comet_col + "_norm"] = comet_norm
        gdf[sbert_col + "_norm"] = sbert_norm
        gdf[qe_col] = W_COMET * comet_norm + W_SBERT * sbert_norm

    def compute_flags_row(row, suffix):
        flags = []
        qe = row.get(f"qe_mix_{suffix}", np.nan)
        if not np.isnan(qe) and qe < TH_QE_MIX_LOW: flags.append("low_qe_mix")
        lr = row.get(f"len_ratio_{suffix}", np.nan)
        if not np.isnan(lr) and (lr < TH_LENGTH_RATIO_LOW or lr > TH_LENGTH_RATIO_HIGH):
            flags.append("length_ratio_extreme")
        pj = row.get(f"punc_jaccard_{suffix}", np.nan)
        if not np.isnan(pj) and pj < TH_PUNC_JACCARD_LOW: flags.append("low_punc_jaccard")
        if TH_DIGIT_MATCH:
            dm = row.get(f"digits_match_{suffix}", 1)
            if dm == 0: flags.append("digits_mismatch")
        cn = row.get(f"cosn_jap_{suffix}", np.nan)
        if not np.isnan(cn) and cn > TH_CHAR_N_HIGH_SIM: flags.append("high_char_ngram_sim")
        c_norm = row.get(f"comet_jap_{suffix}_norm", np.nan)
        s_norm = row.get(f"sbert_jap_{suffix}_norm", np.nan)
        if (not np.isnan(c_norm)) and (not np.isnan(s_norm)) and abs(c_norm - s_norm) > TH_DISCREPANCY:
            flags.append("model_discrepancy")
        return ";".join(flags) if flags else ""

    gdf["flags_en"] = gdf.apply(lambda r: compute_flags_row(r, "en"), axis=1)
    gdf["flags_sp"] = gdf.apply(lambda r: compute_flags_row(r, "sp"), axis=1)
    def consolidate_flags(row):
        flags = []
        if row.get("flags_en"): flags.extend([f"EN:{x}" for x in row["flags_en"].split(";") if x])
        if row.get("flags_sp"): flags.extend([f"SP:{x}" for x in row["flags_sp"].split(";") if x])
        return ";".join(flags) if flags else ""
    gdf["flags_all"] = gdf.apply(consolidate_flags, axis=1)

    out_csv = os.path.join(out_root, f"metrics_group_{group_name}.csv")
    gdf.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved metrics CSV -> {out_csv}")

    # plot distribution
    import matplotlib.pyplot as plt, io, base64
    plt.figure(figsize=(6,3.5))
    if "qe_mix_en" in gdf.columns and gdf["qe_mix_en"].notna().any():
        vals = gdf["qe_mix_en"].dropna(); label = "QE_mix_en"
    elif "qe_mix_sp" in gdf.columns and gdf["qe_mix_sp"].notna().any():
        vals = gdf["qe_mix_sp"].dropna(); label = "QE_mix_sp"
    else:
        vals = pd.Series(gdf.filter(regex="qe_mix").mean(axis=1).dropna()); label = "QE_mix"
    plt.hist(vals, bins=25)
    plt.title(f"Distribution {label} — group {group_name}")
    plt.xlabel("QE_mix (normalized)"); plt.ylabel("Frequency")
    buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format="png"); plt.close(); buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode("utf-8")

    mean_qe = float(np.nanmean(vals)) if len(vals) else float("nan")
    median_qe = float(np.nanmedian(vals)) if len(vals) else float("nan")
    n_rows = len(gdf)
    pct_low = float((vals < TH_QE_MIX_LOW).mean() * 100) if len(vals) else 0.0

    flags_flat = gdf["flags_all"].dropna().astype(str)
    counts = {}
    for fcell in flags_flat:
        if not fcell: continue
        for f in fcell.split(";"):
            counts[f] = counts.get(f,0) + 1
    flags_sorted = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    sample_html = gdf.head(20).to_html(classes="sample", index=False, escape=False)
    tpl = Template(HTML_TEMPLATE)
    html = tpl.render(
        group=group_name,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        n_rows=n_rows,
        mean_qe_mix=f"{mean_qe:.4f}" if not np.isnan(mean_qe) else "NA",
        median_qe_mix=f"{median_qe:.4f}" if not np.isnan(median_qe) else "NA",
        pct_low_quality=f"{pct_low:.2f}%",
        flags=flags_sorted,
        plot_qe=plot_b64,
        sample_table=sample_html
    )
    out_html = os.path.join(out_root, f"report_group_{group_name}.html")
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[OK] Saved HTML report -> {out_html}")
    return out_csv, out_html, gdf

def main():
    ensure_dir(OUT_ROOT)
    print("[INFO] Reading dataset:", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH, engine="python", on_bad_lines="warn")
    if "group" not in df.columns:
        df["group"] = "all"
    groups = sorted(df["group"].dropna().unique())
    all_frames = []
    for g in groups:
        group_dir = os.path.join(OUT_ROOT, f"group__{g}")
        ensure_dir(group_dir)
        subdf = df[df["group"] == g].copy()
        print(f"[INFO] Processing group={g} ({len(subdf)} rows)")
        csv_path, html_path, gdf = process_group(subdf, g, group_dir)
        all_frames.append(gdf)
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_csv = os.path.join(OUT_ROOT, "metrics_all_groups.csv")
        combined.to_csv(combined_csv, index=False, encoding="utf-8")
        print("[OK] Saved combined CSV:", combined_csv)
    else:
        print("[WARN] No data processed.")

if __name__ == "__main__":
    main()

