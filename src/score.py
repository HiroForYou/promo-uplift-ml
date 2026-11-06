# %% [markdown]
# ### promo-uplift-ml · Precisión Promocional
# src/score.py — Pipeline de scoring MVP
# 
# Paso 6 de 8: Genera predicciones + semáforo de recomendación por combo.
# 
# FLUJO:
#     models/<CHAMPION_MODEL>  +  data/train-set-features.csv (o datos frescos)
#         → build_features()
#         → Predicción de unidades vendidas con promo
#         → Semáforo GO/CONSIDER/AVOID basado en umbral de uplift
#         → Top-3 features SHAP por fila (si shap_importance.csv disponible)
#         → reports/scoring/scoring_output.csv
# 
# CAMPOS (BBP v1.4):
#     cobertura_score   — días_únicos_con_venta_90d / esperados_90d  (cobertura histórica calibrada)
#                         esperados_90d = frecuencia_natural_combo × 90
#                         frecuencia_natural = días_únicos_con_venta / días_calendario_historial
#                         Recalibrado Junio 2026: denominador fijo 90 penalizaba retail fashion
#                         con venta irregular (3 ventas/sem → 0.43 con denominador fijo vs ~1.0 real).
#                         Alineado con alcance_original §21.9 componente (a): "volumen histórico".
#     is_extrapolated   — TRUE si el descuento actual supera el máximo histórico del combo
#     confidence_score  — cobertura_score × (1 − fraccion_extrapolada)  [0, 1]
#                         Solo implementa componente (a) del alcance §21.9. Componentes (b) varianza
#                         CV por combo y (c) cercanía a training quedan para Fase 2.
# 
# SEMÁFORO (proxy MVP — alcance §21.9):
#     GO      — uplift_pct ≥ 15 %   → recomendar la promo
#               + is_extrapolated=False  (regla BBP v1.4: extrapolado → baja a CONSIDER)
#     CONSIDER— 5 % ≤ uplift_pct < 15 %, o uplift ≥ 15% pero is_extrapolated=True
#     AVOID   — uplift_pct < 5 %, o flag_baseline_insuficiente=1  (regla BBP v1.4)
# 
#     Semáforo completo del alcance (GO: confidence≥0.7 AND margin_roi>1 AND stock OK)
#     pendiente Fase 2 cuando lleguen datos SAP de costos (A018) e inventario (MARD).
# 
# USO:
#     python src/score.py
#     python src/score.py --data data/train-set-features.csv
#     python src/score.py --go-threshold 15 --consider-threshold 5

# %%
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
except NameError:
    _cwd = Path().resolve()
    sys.path.insert(0, str(_cwd if (_cwd / "config.py").exists() else _cwd.parent))
from src.features import build_features, FEATURE_COLS, CAT_COLS
from src._console import reconfigure_stdout
from config import TRAIN_DATA_PATH, MODELS_DIR, REPORTS_SCORING as REPORTS, CHAMPION_MODEL, UPLIFT_GO, UPLIFT_CONSIDER, TARGET

REPORTS.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# #### Helpers de semáforo y SHAP

# %%

def _semaforo(uplift_pct: float, go_thr: float, consider_thr: float) -> str:
    if uplift_pct >= go_thr:
        return "GO"
    elif uplift_pct >= consider_thr:
        return "CONSIDER"
    return "AVOID"


# Nombres legibles en español para recommendation_reason (alcance_M1 sección 11)
_FEAT_LABELS: dict[str, str] = {
    "pct_descuento_calc":       "% descuento",
    "descuento_total":          "descuento absoluto",
    "precio_efectivo":          "precio transaccional",
    "costo_unitario":           "costo unitario",
    "margin_bruto":             "margen bruto",
    "pct_margen":               "% margen",
    "avg_units_4w":             "demanda baseline 4w",
    "avg_units_8w":             "demanda baseline 8w",
    "avg_units_12w":            "demanda baseline 12w",
    "rolling_7d_mean":          "media ventas 7d",
    "rolling_28d_mean":         "media ventas 28d",
    "dia_semana":               "día de semana",
    "es_fin_semana":            "fin de semana",
    "es_quincena":              "quincena",
    "mes":                      "mes",
    "dias_desde_ultima_promo":  "días desde última promo",
    "frecuencia_promo_90d":     "frecuencia promo 90d",
    "n_tallas_vendidas":        "nro. tallas",
    "marca_desc":               "marca",
    "division_desc":            "división",
    "stock_disponible":         "stock disponible",
    "tiene_promo_flag":         "promo activa",
    "semanas_en_mercado":       "semanas en mercado",
}


def _shap_reason(
    shap_row: np.ndarray,
    feat_cols: list[str],
    uplift_pct: float,
    top_n: int = 3,
) -> str:
    """Genera texto narrativo en español con top-N drivers SHAP (alcance sección 11)."""
    abs_vals  = np.abs(shap_row)
    top_idx   = np.argsort(abs_vals)[::-1][:top_n]
    parts = []
    for i in top_idx:
        feat  = feat_cols[i]
        label = _FEAT_LABELS.get(feat, feat.replace("_", " "))
        sign  = "positivo" if shap_row[i] > 0 else "negativo"
        parts.append(f"{label} ({sign})")
    drivers = ", ".join(parts)
    uplift_str = f"+{uplift_pct:.0f}%" if uplift_pct >= 0 else f"{uplift_pct:.0f}%"
    return f"Uplift estimado {uplift_str} en unidades. Drivers: {drivers}."

# %% [markdown]
# #### Main

# %%
def main() -> None:
    reconfigure_stdout()

    parser = argparse.ArgumentParser(
        description="M1 Demand Response — Pipeline de scoring MVP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data",    type=Path, default=TRAIN_DATA_PATH)
    parser.add_argument("--model",   type=Path, default=MODELS_DIR / CHAMPION_MODEL)
    parser.add_argument(
        "--go-threshold",      type=float, default=UPLIFT_GO,
        help=f"Uplift %% mínimo para semáforo GO (default: {UPLIFT_GO})",
    )
    parser.add_argument(
        "--consider-threshold", type=float, default=UPLIFT_CONSIDER,
        help=f"Uplift %% mínimo para semáforo CONSIDER (default: {UPLIFT_CONSIDER})",
    )
    parser.add_argument(
        "--scoring-date", type=str, default=None,
        help="Fecha de scoring YYYYMMDD (default: última fecha en los datos)",
    )
    parser.add_argument(
        "--active-window-days", type=int, default=None,
        help=(
            "Si se especifica, solo puntúa combos cuya última venta sea dentro de "
            "los últimos N días respecto al scoring_date. Ejemplo: --active-window-days 180 "
            "filtra a combos con al menos una venta en los últimos 6 meses. "
            "En producción esto no es necesario (el dato ya llega fresco); en MVP con "
            "dataset histórico permite simular el universo activo real."
        ),
    )
    args, _ = parser.parse_known_args()

    if not args.model.exists():
        fallbacks = [
            MODELS_DIR / "xgb_base.pkl",
            MODELS_DIR / "lgbm_tuned.pkl",
            MODELS_DIR / "lgbm_base.pkl",
        ]
        alt = next((p for p in fallbacks if p.exists()), None)
        if alt is None:
            raise FileNotFoundError(
                f"No se encontró {args.model.name} ni ningún modelo de respaldo en {MODELS_DIR}"
            )
        print(f"  ⚠️  {args.model.name} no encontrado, usando {alt.name}")
        args.model = alt

    sep = "=" * 68
    print(f"\n{sep}")
    print("  M1 Demand Response — Scoring MVP (Semana 10)")
    print(f"  Modelo: {args.model.name}")
    print(f"  Semáforo: GO≥{args.go_threshold}%  CONSIDER≥{args.consider_threshold}%")
    print(f"{sep}\n")

    # ── Cargar y preparar ─────────────────────────────────────────────────────
    from config import CSV_SEP
    print(f"Cargando: {args.data}")
    df_raw = pd.read_csv(
        args.data,
        sep=CSV_SEP,
        dtype={"tienda_id": str, "generico_id": str},
        low_memory=False,
    )
    print(f"  {len(df_raw):,} filas × {len(df_raw.columns)} columnas")

    for c in df_raw.select_dtypes(include=["object"]).columns:
        try:
            df_raw[c] = pd.to_numeric(df_raw[c])
        except (ValueError, TypeError):
            pass

    print("Aplicando build_features()...")
    df = build_features(df_raw)

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].clip(lower=0)

    feat_cols     = [c for c in FEATURE_COLS if c in df.columns]
    cat_cols_used = [c for c in CAT_COLS if c in feat_cols]
    for c in cat_cols_used:
        df[c] = df[c].astype("category")

    # ── Fecha de scoring: última observación por combo ────────────────────────
    scoring_date = (
        pd.to_datetime(args.scoring_date, format="%Y%m%d")
        if args.scoring_date
        else df["fecha_venta"].max()
    )
    scoring_df = (
        df[df["fecha_venta"] <= scoring_date]
        .sort_values("fecha_venta")
        .groupby(["tienda_id", "generico_id"])
        .last()
        .reset_index()
    )

    # Filtro opcional: solo combos con actividad reciente (simula universo activo en prod)
    if args.active_window_days:
        active_cutoff = scoring_date - pd.Timedelta(days=args.active_window_days)
        before = len(scoring_df)
        scoring_df = scoring_df[scoring_df["fecha_venta"] >= active_cutoff]
        print(f"  Filtro --active-window-days={args.active_window_days}: "
              f"{before - len(scoring_df):,} combos inactivos excluidos "
              f"(última venta < {active_cutoff.date()})")

    print(f"  Fecha de scoring: {scoring_date.date()}")
    print(f"  Combos a puntuar: {len(scoring_df):,}")

    # ── Predicción CON promo (tiene_promo_flag=1) ─────────────────────────────
    model = joblib.load(args.model)

    X_con_promo = scoring_df[feat_cols].copy()
    if "tiene_promo_flag" in feat_cols:
        X_con_promo["tiene_promo_flag"] = 1
    if "en_calendario_promo" in feat_cols:
        X_con_promo["en_calendario_promo"] = 1
    for c in cat_cols_used:
        if c in X_con_promo.columns:
            X_con_promo[c] = X_con_promo[c].astype("category")

    pred_con_promo = model.predict(X_con_promo).clip(min=0)

    # ── Predicción SIN promo (tiene_promo_flag=0) — baseline ─────────────────
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

    pred_sin_promo = model.predict(X_sin_promo).clip(min=0)

    # ── Calcular uplift y semáforo ────────────────────────────────────────────
    uplift_units = pred_con_promo - pred_sin_promo
    uplift_pct   = np.where(
        pred_sin_promo > 0,
        (uplift_units / pred_sin_promo) * 100,
        np.where(pred_con_promo > 0, 100.0, 0.0),
    )
    semaforo = [
        _semaforo(u, args.go_threshold, args.consider_threshold)
        for u in uplift_pct
    ]

    # ── cobertura_score: días únicos con venta en 90d / esperados por frecuencia natural ─────────
    # Denominador recalibrado (Junio 2026): dividir entre 90 días calendario asume venta diaria.
    # En retail fashion un combo con 3 ventas/semana tiene ~38 días únicos en 90d → 0.43 con
    # denominador fijo, cuando debería ser ~1.0. Nuevo denominador = frecuencia_natural × 90,
    # donde frecuencia_natural = días_únicos_con_venta / días_calendario_del_historial_completo.
    if "fecha_venta" in df.columns:
        window_start = scoring_date - pd.Timedelta(days=90)

        # Días únicos con venta en ventana 90d (nunique evita contar duplicados)
        counts_90d = (
            df[df["fecha_venta"].between(window_start, scoring_date)]
            .groupby(["tienda_id", "generico_id"])["fecha_venta"]
            .nunique()
            .reset_index(name="dias_con_venta_90d")
        )

        # Frecuencia natural histórica del combo: días_únicos / días_calendario_total
        freq_hist = (
            df.groupby(["tienda_id", "generico_id"])["fecha_venta"]
            .agg(dias_venta="nunique", first_sale="min", last_sale="max")
            .reset_index()
        )
        freq_hist["dias_cal"] = (freq_hist["last_sale"] - freq_hist["first_sale"]).dt.days + 1
        freq_hist["esperados_90d"] = (
            (freq_hist["dias_venta"] / freq_hist["dias_cal"].clip(lower=1)) * 90
        ).clip(lower=1.0, upper=90.0)

        scoring_df = (
            scoring_df
            .merge(counts_90d, on=["tienda_id", "generico_id"], how="left")
            .merge(
                freq_hist[["tienda_id", "generico_id", "esperados_90d"]],
                on=["tienda_id", "generico_id"], how="left",
            )
        )
        scoring_df["dias_con_venta_90d"] = scoring_df["dias_con_venta_90d"].fillna(0)
        scoring_df["esperados_90d"]      = scoring_df["esperados_90d"].fillna(1.0)
    else:
        scoring_df["dias_con_venta_90d"] = 0
        scoring_df["esperados_90d"]      = 1.0

    cobertura_score = np.minimum(
        1.0,
        scoring_df["dias_con_venta_90d"].values / scoring_df["esperados_90d"].values,
    )

    # ── is_extrapolated: descuento actual > máximo histórico del combo ────────
    if "pct_descuento_calc" in df.columns and "pct_descuento_calc" in scoring_df.columns:
        max_desc_hist = (
            df.groupby(["tienda_id", "generico_id"])["pct_descuento_calc"]
            .max()
            .reset_index(name="max_desc_hist")
        )
        scoring_df = scoring_df.merge(
            max_desc_hist, on=["tienda_id", "generico_id"], how="left"
        )
        scoring_df["max_desc_hist"] = scoring_df["max_desc_hist"].fillna(0)
        is_extrapolated = (
            scoring_df["pct_descuento_calc"].values
            > scoring_df["max_desc_hist"].values
        ).astype(bool)
    else:
        is_extrapolated = np.zeros(len(scoring_df), dtype=bool)

    # fraccion_extrapolada: 0 si dentro del rango, 1 si completamente fuera
    fraccion_extrapolada = is_extrapolated.astype(float)

    # confidence_score = cobertura_historica × (1 − fraccion_extrapolada)
    confidence_score = cobertura_score * (1.0 - fraccion_extrapolada)

    # ── Reglas BBP v1.4: ajuste post-umbral del semáforo ─────────────────────
    # Regla 1 — is_extrapolated → máximo CONSIDER (nunca GO).
    #   El modelo predice fuera del rango histórico de descuentos: confianza insuficiente.
    # Regla 2 — flag_baseline_insuficiente → AVOID.
    #   Sin baseline suficiente (<14 días sin promo en 28d) el uplift no es confiable.
    # Nota: el semáforo por uplift_pct es el proxy válido de MVP mientras no lleguen
    #   datos reales de stock (MARD) y costos (A018). Ver plan_trabajo §Semana 12+.
    semaforo_arr = np.array(semaforo)
    semaforo_arr = np.where(
        is_extrapolated & (semaforo_arr == "GO"), "CONSIDER", semaforo_arr
    )
    if "flag_baseline_insuficiente" in scoring_df.columns:
        flag_bl = scoring_df["flag_baseline_insuficiente"].fillna(0).astype(bool).values
        semaforo_arr = np.where(flag_bl, "AVOID", semaforo_arr)
    semaforo = semaforo_arr.tolist()

    # ── SHAP por fila — recommendation_reason narrativo (alcance sección 11) ──
    print("\n  Calculando SHAP values por fila...")
    reason_list: list[str] = [""] * len(scoring_df)
    try:
        import shap as _shap
        from xgboost import XGBRegressor as _XGBReg
        _X_shap = X_con_promo.copy()
        if isinstance(model, _XGBReg):
            for _c in _X_shap.select_dtypes(include=["category"]).columns:
                _X_shap[_c] = _X_shap[_c].cat.codes.astype("int16")
        explainer   = _shap.TreeExplainer(model)
        shap_values = explainer.shap_values(_X_shap)
        for i in range(len(scoring_df)):
            reason_list[i] = _shap_reason(
                shap_values[i], feat_cols, float(uplift_pct[i])
            )
        print(f"  ✅ SHAP calculado para {len(scoring_df):,} filas")
    except Exception as exc:
        print(f"  ⚠️  SHAP no disponible ({exc}) — usando ranking global")
        reason_list = [
            f"Uplift estimado {'+' if u >= 0 else ''}{u:.0f}% en unidades."
            for u in uplift_pct
        ]

    # ── Armar output ──────────────────────────────────────────────────────────
    output_cols = {
        "tienda_id":               scoring_df["tienda_id"],
        "generico_id":             scoring_df["generico_id"],
        "scoring_date":            scoring_date.date(),
        "pred_units_con_promo":    np.round(pred_con_promo, 3),
        "pred_units_sin_promo":    np.round(pred_sin_promo, 3),
        "uplift_units":            np.round(uplift_units, 3),
        "uplift_pct":              np.round(uplift_pct, 2),
        "cobertura_score":         np.round(cobertura_score, 4),
        "is_extrapolated":         is_extrapolated,
        "confidence_score":        np.round(confidence_score, 4),
        "recommendation_semaforo": semaforo,
        "recommendation_reason":   reason_list,
    }

    # Columnas de contexto — precio_venta y precio_compra al 100% en dataset v2
    for col in [
        "division_desc", "marca_desc", "banner", "region_desc", "deporte_desc",
        "avg_units_4w", "pct_descuento_calc", "precio_efectivo",
        "precio_venta", "precio_compra",
    ]:
        if col in scoring_df.columns:
            output_cols[col] = scoring_df[col].values

    output_df = pd.DataFrame(output_cols)

    # ── Validaciones bloqueantes ──────────────────────────────────────────────
    print("\n  Validaciones bloqueantes:")
    errs = []
    if (output_df["uplift_pct"].isna()).any():
        errs.append("uplift_pct tiene nulos")
    if not output_df["recommendation_semaforo"].isin(["GO","CONSIDER","AVOID"]).all():
        errs.append("semáforo con valores inválidos")
    if (output_df["pred_units_con_promo"] < 0).any():
        errs.append("predicciones negativas")

    if errs:
        for e in errs:
            print(f"  ❌ {e}")
        raise ValueError(f"Validaciones fallidas: {errs}")
    print("  ✅ Todas las validaciones pasadas")

    # ── Resumen del scoring ───────────────────────────────────────────────────
    sem_counts = output_df["recommendation_semaforo"].value_counts()
    total = len(output_df)
    print(f"\n  Distribución de semáforo ({total:,} combos):")
    for label, color in [("GO", "🟢"), ("CONSIDER", "🟡"), ("AVOID", "🔴")]:
        n = sem_counts.get(label, 0)
        print(f"  {color} {label:<10s}: {n:>6,}  ({100*n/total:.1f} %)")

    print(f"\n  Uplift medio global: {output_df['uplift_pct'].mean():+.1f} %")
    if "division_desc" in output_df.columns:
        print(f"\n  Uplift medio por división:")
        div_up = output_df.groupby("division_desc")["uplift_pct"].mean().sort_values(ascending=False)
        for div, up in div_up.items():
            print(f"    {str(div):<35s}  {up:+.1f} %")

    # ── Guardar ───────────────────────────────────────────────────────────────
    out_path = REPORTS / "scoring_output.csv"
    output_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n{'─'*68}")
    print(f"  {out_path}")
    print(f"  {len(output_df):,} filas  |  {output_df.columns.tolist()}")
    print(f"\n{sep}")
    print("  Listo. Próximo paso: escritura a Snowflake staging (Semana 11)")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
