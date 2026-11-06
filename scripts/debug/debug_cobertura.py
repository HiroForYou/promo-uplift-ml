"""
scripts/debug_cobertura.py — Diagnóstico del merge de cobertura_score

Verifica por qué confidence_score tiene valores casi cero.

USO:
    python scripts/debug_cobertura.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import build_features

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA_PATH = ROOT / "data" / "train-set-features.csv"

print("=" * 68)
print("  Diagnóstico cobertura_score / confidence_score")
print("=" * 68)

# ── 1. Cargar datos ────────────────────────────────────────────────────────
print(f"\n[1] Cargando {TRAIN_DATA_PATH.name}...")
df_raw = pd.read_csv(
    TRAIN_DATA_PATH,
    dtype={"tienda_id": str, "generico_id": str},
    low_memory=False,
)
print(f"    {len(df_raw):,} filas × {len(df_raw.columns)} columnas")

for c in df_raw.select_dtypes(include=["object"]).columns:
    try:
        df_raw[c] = pd.to_numeric(df_raw[c])
    except (ValueError, TypeError):
        pass

print("    Aplicando build_features()...")
df = build_features(df_raw)
if "division_desc" in df.columns:
    df = df[~df["division_desc"].isin({"VENTA DE BOLSAS"})]
df = df.dropna(subset=["unidades_vendidas"])
df["unidades_vendidas"] = df["unidades_vendidas"].clip(lower=0)
print(f"    Post build_features: {len(df):,} filas")

# ── 2. Tipos de datos ──────────────────────────────────────────────────────
print("\n[2] Tipos de datos clave:")
print(f"    tienda_id   dtype: {df['tienda_id'].dtype}")
print(f"    generico_id dtype: {df['generico_id'].dtype}")
print(f"    fecha_venta dtype: {df['fecha_venta'].dtype}")
print(f"    tienda_id   muestra: {df['tienda_id'].iloc[:3].tolist()}")
print(f"    generico_id muestra: {df['generico_id'].iloc[:3].tolist()}")

# ── 3. Rango de fechas ─────────────────────────────────────────────────────
scoring_date = df["fecha_venta"].max()
window_start = scoring_date - pd.Timedelta(days=90)
print(f"\n[3] Rango de fechas en df:")
print(f"    Min fecha_venta  : {df['fecha_venta'].min()}")
print(f"    Max fecha_venta  : {df['fecha_venta'].max()}")
print(f"    scoring_date     : {scoring_date}  (tipo: {type(scoring_date).__name__})")
print(f"    window_start     : {window_start}")

# ── 4. Filas en ventana 90d ────────────────────────────────────────────────
mask = df["fecha_venta"].between(window_start, scoring_date)
df_90d = df[mask]
print(f"\n[4] Datos en ventana 90d ({window_start.date()} → {scoring_date.date()}):")
print(f"    Filas en ventana      : {mask.sum():,}")
print(f"    Combos únicos 90d     : {df_90d.groupby(['tienda_id','generico_id']).ngroups:,}")

# ── 5. counts_90d ──────────────────────────────────────────────────────────
counts_90d = (
    df_90d
    .groupby(["tienda_id", "generico_id"])
    .size()
    .reset_index(name="dias_con_venta_90d")
)
print(f"\n[5] counts_90d (después de reset_index):")
print(f"    Filas               : {len(counts_90d):,}")
print(f"    tienda_id dtype     : {counts_90d['tienda_id'].dtype}")
print(f"    generico_id dtype   : {counts_90d['generico_id'].dtype}")
print(f"    dias_con_venta_90d  media: {counts_90d['dias_con_venta_90d'].mean():.2f}")
print(f"    dias_con_venta_90d  max  : {counts_90d['dias_con_venta_90d'].max()}")
print(f"    Muestra:\n{counts_90d.head(3).to_string(index=False)}")

# ── 6. scoring_df ──────────────────────────────────────────────────────────
scoring_df = (
    df[df["fecha_venta"] <= scoring_date]
    .sort_values("fecha_venta")
    .groupby(["tienda_id", "generico_id"])
    .last()
    .reset_index()
)
print(f"\n[6] scoring_df (última obs por combo):")
print(f"    Filas               : {len(scoring_df):,}")
print(f"    tienda_id dtype     : {scoring_df['tienda_id'].dtype}")
print(f"    generico_id dtype   : {scoring_df['generico_id'].dtype}")
print(f"    tienda_id muestra   : {scoring_df['tienda_id'].iloc[:3].tolist()}")
print(f"    generico_id muestra : {scoring_df['generico_id'].iloc[:3].tolist()}")

# ── 7. Merge ───────────────────────────────────────────────────────────────
merged = scoring_df[["tienda_id", "generico_id"]].merge(
    counts_90d, on=["tienda_id", "generico_id"], how="left"
)
nans = merged["dias_con_venta_90d"].isna().sum()
print(f"\n[7] Merge resultado:")
print(f"    Filas resultado     : {len(merged):,}")
print(f"    NaN dias_90d        : {nans:,}  ({100*nans/len(merged):.1f} %)")
print(f"    Non-NaN dias_90d    : {(~merged['dias_con_venta_90d'].isna()).sum():,}")
print(f"    Media dias_90d (fill 0): {merged['dias_con_venta_90d'].fillna(0).mean():.4f}")

# ── 8. Compatibilidad de llaves ────────────────────────────────────────────
print(f"\n[8] Solapamiento de llaves scoring_df ∩ counts_90d:")
keys_scoring = set(zip(scoring_df["tienda_id"], scoring_df["generico_id"]))
keys_counts  = set(zip(counts_90d["tienda_id"],  counts_90d["generico_id"]))
overlap = keys_scoring & keys_counts
print(f"    Combos en scoring_df  : {len(keys_scoring):,}")
print(f"    Combos en counts_90d  : {len(keys_counts):,}")
print(f"    Combos en intersección: {len(overlap):,}")

# ── 9. Distribución de cobertura ───────────────────────────────────────────
cobertura = np.minimum(1.0, merged["dias_con_venta_90d"].fillna(0) / 90.0)
print(f"\n[9] cobertura_score (MIN(1, dias/90)):")
print(f"    Media    : {cobertura.mean():.4f}")
print(f"    > 0      : {(cobertura > 0).sum():,}  ({100*(cobertura>0).mean():.1f} %)")
print(f"    > 0.1    : {(cobertura > 0.1).sum():,}")
print(f"    > 0.5    : {(cobertura > 0.5).sum():,}")
print(f"    Percentiles [25,50,75,90,99]: "
      f"{np.percentile(cobertura, [25,50,75,90,99]).round(4).tolist()}")

# ── 10. Distribución de fechas en scoring_df ──────────────────────────────
print(f"\n[10] Distribución de fecha_venta en scoring_df (última obs por combo):")
if "fecha_venta" in scoring_df.columns:
    sd_dates = pd.to_datetime(scoring_df["fecha_venta"])
    print(f"    Min: {sd_dates.min().date()}")
    print(f"    Max: {sd_dates.max().date()}")
    buckets = {
        "< 2025-09-01": (sd_dates < "2025-09-01").sum(),
        "2025-09-01 → 2025-11-29": ((sd_dates >= "2025-09-01") & (sd_dates < "2025-11-30")).sum(),
        "≥ 2025-11-30 (dentro ventana 90d)": (sd_dates >= "2025-11-30").sum(),
    }
    for label, n in buckets.items():
        print(f"    {label}: {n:,} combos ({100*n/len(scoring_df):.1f} %)")
else:
    print("    fecha_venta no disponible en scoring_df")

print(f"\n{'='*68}")
print("  Diagnóstico completo.")
print(f"{'='*68}\n")
