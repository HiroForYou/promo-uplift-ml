"""
scripts/profile_score.py — Profiler de tiempos por sección de score.py

Replica el flujo exacto de score.py midiendo el tiempo de cada paso.
No modifica ningún archivo de salida.

USO:
    python scripts/profile_score.py
    python scripts/profile_score.py --active-window-days 180
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import FEATURE_COLS, CAT_COLS

ROOT       = Path(__file__).resolve().parent.parent
TRAIN_DATA_PATH  = ROOT / "data" / "train-set-features.csv"
MODELS_DIR = ROOT / "models"
EXCLUDED_DIVISIONS = {"VENTA DE BOLSAS"}


def _tick(label: str, t0: float) -> float:
    elapsed = time.perf_counter() - t0
    print(f"  {elapsed:>7.2f}s  {label}")
    return time.perf_counter()


def main():
    parser = argparse.ArgumentParser(description="Profiler de score.py")
    parser.add_argument("--data",  type=Path, default=TRAIN_DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "lgbm_tuned.pkl")
    parser.add_argument("--active-window-days", type=int, default=None)
    args = parser.parse_args()

    if not args.model.exists():
        alt = MODELS_DIR / "lgbm_base.pkl"
        print(f"  ⚠️  {args.model.name} no encontrado, usando {alt.name}")
        args.model = alt

    sep = "=" * 68
    print(f"\n{sep}")
    print("  score.py — Profiler de tiempos")
    print(f"  Datos : {args.data.name}")
    print(f"  Modelo: {args.model.name}")
    print(f"{sep}")
    print(f"\n  {'Tiempo':>7s}  Sección")
    print(f"  {'─'*60}")

    wall_start = time.perf_counter()
    t = time.perf_counter()

    # ── 1. CSV read ───────────────────────────────────────────────────────────
    df_raw = pd.read_csv(
        args.data,
        dtype={"tienda_id": str, "generico_id": str},
        low_memory=False,
    )
    t = _tick(f"pd.read_csv  ({len(df_raw):,} filas × {len(df_raw.columns)} cols)", t)

    # ── 2. Coerción de tipos (bucle actual) ───────────────────────────────────
    t0_coerce = time.perf_counter()
    for c in df_raw.select_dtypes(include=["object"]).columns:
        try:
            df_raw[c] = pd.to_numeric(df_raw[c])
        except (ValueError, TypeError):
            pass
    t = _tick("Coerción de tipos (bucle try/except)", t)

    # ── 3. build_features() — desglosado por etapa interna ───────────────────
    from src.features import (
        build_baseline_features, build_promo_history_features,
        build_seasonality_features, build_lag_features,
        _SQL_BASELINE_SENTINEL, _SQL_PROMO_SENTINEL,
        GROUP_COLS,
    )

    df = df_raw.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["fecha_venta"]):
        df["fecha_venta"] = pd.to_datetime(df["fecha_venta"].astype(str), format="%Y%m%d")
    t = _tick("build_features: parse fecha_venta", t)

    df = df.sort_values(GROUP_COLS + ["fecha_venta"]).reset_index(drop=True)
    t = _tick(f"build_features: sort_values ({len(df):,} filas)", t)

    # precio_efectivo / costo_unitario (siempre)
    if "precio_efectivo" not in df.columns:
        df["precio_efectivo"] = df["monto_venta"] / df["unidades_vendidas"].replace(0, np.nan)
    if "costo_unitario" not in df.columns:
        df["costo_unitario"] = df["costo_linea"] / df["unidades_vendidas"].replace(0, np.nan)
    t = _tick("build_features: precio_efectivo / costo_unitario", t)

    # Tarea 3.1 — baseline (solo si no viene del SQL)
    if _SQL_BASELINE_SENTINEL not in df.columns:
        df = build_baseline_features(df)
        t = _tick("build_features: build_baseline_features (groupby rolling — Python lambda)", t)
    else:
        t = _tick("build_features: baseline SKIPPED (viene del SQL)", t)

    # Tarea 3.2 — historial promo (solo si no viene del SQL)
    if _SQL_PROMO_SENTINEL not in df.columns:
        df = build_promo_history_features(df)
        t = _tick("build_features: build_promo_history_features (groupby rolling — Python lambda)", t)
    else:
        t = _tick("build_features: promo history SKIPPED (viene del SQL)", t)

    # Tarea 3.3 — estacionalidad
    df = build_seasonality_features(df)
    t = _tick("build_features: build_seasonality_features", t)

    # Lags y rolling
    df = build_lag_features(df)
    t = _tick("build_features: build_lag_features (7× groupby transform lambda)", t)

    if "division_desc" in df.columns:
        df = df[~df["division_desc"].isin(EXCLUDED_DIVISIONS)]
    df = df.dropna(subset=["unidades_vendidas"])
    df["unidades_vendidas"] = df["unidades_vendidas"].clip(lower=0)
    t = _tick(f"build_features: filter + clip  ({len(df):,} filas)", t)

    # ── 4. Feature encoding (category cast) ──────────────────────────────────
    feat_cols     = [c for c in FEATURE_COLS if c in df.columns]
    cat_cols_used = [c for c in CAT_COLS if c in feat_cols]
    for c in cat_cols_used:
        df[c] = df[c].astype("category")
    t = _tick(f"Category encoding  ({len(cat_cols_used)} cols cat)", t)

    # ── 5. scoring_df construction ────────────────────────────────────────────
    scoring_date = df["fecha_venta"].max()
    scoring_df = (
        df[df["fecha_venta"] <= scoring_date]
        .sort_values("fecha_venta")
        .groupby(["tienda_id", "generico_id"])
        .last()
        .reset_index()
    )
    t = _tick(f"scoring_df (groupby last)  → {len(scoring_df):,} combos", t)

    # ── 5b. Filtro active-window-days (si aplica) ─────────────────────────────
    if args.active_window_days:
        active_cutoff = scoring_date - pd.Timedelta(days=args.active_window_days)
        before = len(scoring_df)
        scoring_df = scoring_df[scoring_df["fecha_venta"] >= active_cutoff]
        t = _tick(f"Filtro active_window_days={args.active_window_days}  "
                  f"({before}→{len(scoring_df)} combos)", t)

    # ── 6. Model load ─────────────────────────────────────────────────────────
    model = joblib.load(args.model)
    t = _tick(f"joblib.load (modelo)", t)

    # ── 7. Predict CON promo ──────────────────────────────────────────────────
    X_con_promo = scoring_df[feat_cols].copy()
    if "tiene_promo_flag" in feat_cols:
        X_con_promo["tiene_promo_flag"] = 1
    if "en_calendario_promo" in feat_cols:
        X_con_promo["en_calendario_promo"] = 1
    for c in cat_cols_used:
        if c in X_con_promo.columns:
            X_con_promo[c] = X_con_promo[c].astype("category")
    pred_con = model.predict(X_con_promo).clip(min=0)
    t = _tick(f"model.predict (CON promo)  → {len(pred_con):,} filas", t)

    # ── 8. Predict SIN promo ──────────────────────────────────────────────────
    X_sin_promo = scoring_df[feat_cols].copy()
    if "tiene_promo_flag" in feat_cols:
        X_sin_promo["tiene_promo_flag"] = 0
    if "en_calendario_promo" in feat_cols:
        X_sin_promo["en_calendario_promo"] = 0
    if "pct_descuento_calc" in feat_cols:
        X_sin_promo["pct_descuento_calc"] = 0.0
    if "descuento_total" in feat_cols:
        X_sin_promo["descuento_total"] = 0.0
    for c in cat_cols_used:
        if c in X_sin_promo.columns:
            X_sin_promo[c] = X_sin_promo[c].astype("category")
    pred_sin = model.predict(X_sin_promo).clip(min=0)
    t = _tick(f"model.predict (SIN promo)  → {len(pred_sin):,} filas", t)

    # ── 9. cobertura_score merge ──────────────────────────────────────────────
    window_start = scoring_date - pd.Timedelta(days=90)
    counts_90d = (
        df[df["fecha_venta"].between(window_start, scoring_date)]
        .groupby(["tienda_id", "generico_id"])
        .size()
        .reset_index(name="dias_con_venta_90d")
    )
    scoring_df = scoring_df.merge(counts_90d, on=["tienda_id", "generico_id"], how="left")
    scoring_df["dias_con_venta_90d"] = scoring_df["dias_con_venta_90d"].fillna(0)
    t = _tick("cobertura_score merge (90d window)", t)

    # ── 10. is_extrapolated merge ─────────────────────────────────────────────
    if "pct_descuento_calc" in df.columns and "pct_descuento_calc" in scoring_df.columns:
        max_desc = (
            df.groupby(["tienda_id", "generico_id"])["pct_descuento_calc"]
            .max()
            .reset_index(name="max_desc_hist")
        )
        scoring_df = scoring_df.merge(max_desc, on=["tienda_id", "generico_id"], how="left")
        scoring_df["max_desc_hist"] = scoring_df["max_desc_hist"].fillna(0)
    t = _tick("is_extrapolated merge (max desc hist)", t)

    # ── 11. SHAP — exact ──────────────────────────────────────────────────────
    try:
        import shap as _shap
        explainer = _shap.TreeExplainer(model)
        t = _tick("shap.TreeExplainer() — instanciar", t)

        explainer.shap_values(X_con_promo)
        t = _tick(f"shap_values EXACT  ({len(X_con_promo):,} filas × {len(feat_cols)} feats)", t)

        # Alternativa: feature_importances_ global (sin SHAP por fila — instantáneo)
        fi = model.feature_importances_
        top3_global = np.argsort(fi)[::-1][:3]
        top3_names  = [feat_cols[i] for i in top3_global]
        t = _tick(f"model.feature_importances_ global  top3={top3_names}", t)

    except ImportError:
        print("  [SHAP] no instalado — omitido")
        t = time.perf_counter()

    # ── Resumen ───────────────────────────────────────────────────────────────
    total = time.perf_counter() - wall_start
    print(f"\n{sep}")
    print(f"  TOTAL wall-clock: {total:.2f}s")
    print(f"{'─'*68}")
    print(f"  Conclusiones:")
    print(f"    • build_lag_features: OPTIMIZADO en features.py (groupby shift+rolling vectorizado)")
    print(f"    • SHAP: ~40s inevitables con LightGBM SHAP exact (approx y threads no soportados)")
    print(f"    • Alternativa SHAP: feature_importances_ global → 0.00s")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
