# %% [markdown]
# ### promo-uplift-ml · Precisión Promocional
# src/train.py — Entrenamiento final
# 
# EJECUTAR DESPUÉS DE tune.py. Carga los mejores hiperparámetros de
# models/tune_meta.json (si existe) y entrena LightGBM y XGBoost en su
# mejor versión para hacer la comparación BLS justa y seleccionar el champion.
# Si tune_meta.json no existe, usa params base como fallback.
# 
# ALCANCE DEL MODELO (confirmado con cliente 27/05/2026):
#     Predecir cuántas unidades más venderá un genérico en una tienda durante
#     una promoción vs sin promoción (uplift en volumen dado que hay venta).
#     El modelo NO predice si el producto venderá o no.
#     El dataset solo incluye días con transacciones reales — los días sin venta
#     no están en el dataset y quedan fuera del alcance del MVP.
# 
# FLUJO:
#     tune_meta.json (params óptimos)  +  train-set-features.csv
#         → build_features()
#         → split temporal (train < val_cutoff ≤ val)
#         → LightGBM (params tuneados) + XGBoost (params tuneados)
#         → métricas (WMAPE, MAE, R², Bias, Lift Accuracy, BLS)
#         → feature importance
#         → guardar modelos en models/
# 
# USO:
#     python src/train.py
#     python src/train.py --data data/train-set-features.csv
#     python src/train.py --val-cutoff 20251201

# %%
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Importar módulo de features (funciona desde src/ o desde raíz)
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
except NameError:
    _cwd = Path().resolve()
    sys.path.insert(0, str(_cwd if (_cwd / "config.py").exists() else _cwd.parent))
from src.features import build_features, FEATURE_COLS, CAT_COLS
from src._console import reconfigure_stdout
from config import (
    TRAIN_DATA_PATH, MODELS_DIR, INPUT_MODELS_DIR, N_ESTIMATORS, TARGET,
    WMAPE_OBJETIVO, WMAPE_GREEN,
    LIFT_ACC_MIN, LIFT_ACC_OBJETIVO,
    BIAS_TRIGGER,
    R2_MIN,
)

MODELS_DIR.mkdir(exist_ok=True)

# %% [markdown]
# #### Métricas

# %%

def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE — métrica principal (objetivo ≤ 0.12 deseable / ≤ 0.25 gate — BBP v1.4)."""
    denom = np.sum(np.abs(y_true))
    return float(np.nan) if denom == 0 else float(np.sum(np.abs(y_true - y_pred)) / denom)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² — métrica de aceptación mínima ≥ 0.60 (alcance_M1 sección 7)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float(np.nan)

def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sesgo relativo: > 0 sobreestima, < 0 subestima."""
    denom = np.sum(np.abs(y_true))
    return float(np.nan) if denom == 0 else float(np.sum(y_pred - y_true) / denom)


def lift_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline: np.ndarray | None = None,
    threshold: float = 0.20,
) -> float:
    """% de predicciones con lift dentro de ±20% del lift real (alcance §7.5.2).

    Lift = (units - baseline) / baseline. Usa avg_units_4w como proxy de baseline.
    Mínimo ≥ 70%, deseable ≥ 85%. Retorna NaN si no hay baseline disponible.
    """
    if baseline is None:
        return float("nan")
    mask = (baseline > 0) & ~np.isnan(baseline.astype(float))
    if mask.sum() == 0:
        return float("nan")
    bt = baseline[mask]
    lr = (y_true[mask] - bt) / bt
    lp = (y_pred[mask] - bt) / bt
    denom = np.abs(lr)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(lp - lr) / denom
    within = np.where(denom > 1e-6, ratio <= threshold, np.abs(lp - lr) <= threshold)
    return float(within.mean())


def bls_score(wmape_val: float, lift_acc: float, bias_val: float) -> float:
    """BLS scorecard — selección champion/challenger (alcance §7.5.2).

    BLS = Lift_Accuracy × (1 − |Bias| / 0.15) / WMAPE_norm
    WMAPE_norm = WMAPE / 0.25 (umbral objetivo del alcance).
    Mayor BLS = mejor. Fallback a 1/WMAPE_norm si Lift Accuracy no disponible.
    """
    wmape_norm  = max(wmape_val / WMAPE_GREEN, 1e-6)
    bias_factor = max(0.0, 1.0 - abs(bias_val) / BIAS_TRIGGER)
    if np.isnan(lift_acc):
        return 1.0 / wmape_norm
    return (lift_acc * bias_factor) / wmape_norm


# Restricciones de monotonicidad para features de descuento (alcance_M1 sección 6 y 13)
# Garantiza que curvas de elasticidad sean monotónicas sin post-proceso isotónico.
_MONOTONE_CONSTRAINTS: dict[str, int] = {
    "pct_descuento_calc":  1,   # más descuento % → más unidades
    "descuento_total":     1,   # más descuento absoluto → más unidades
    "tiene_promo_flag":    1,   # promo activa → más unidades
    "en_calendario_promo": 1,
}


def _build_monotone_list(feat_cols: list[str]) -> list[int] | None:
    constraints = [_MONOTONE_CONSTRAINTS.get(f, 0) for f in feat_cols]
    return constraints if any(c != 0 for c in constraints) else None

# %% [markdown]
# #### Carga de params tuneados

# %%

def _load_tuned_params(tune_meta_path: Path) -> tuple[dict, dict]:
    """
    Lee tune_meta.json generado por tune.py.
    Retorna (lgbm_params, xgb_params) — dicts vacíos si el archivo no existe.
    """
    if not tune_meta_path.exists():
        return {}, {}
    meta = json.loads(tune_meta_path.read_text(encoding="utf-8"))
    lgbm = meta.get("lgbm", {}).get("best_params", {})
    xgb  = meta.get("xgb",  {}).get("best_params", {})
    return lgbm, xgb

# %% [markdown]
# #### Carga y preparación

# %%

def load_and_prepare(
    path: Path,
    val_cutoff: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Carga CSV, aplica build_features() y hace split temporal.

    Returns:
        X_train, X_val, y_train, y_val, feat_cols, cat_cols_used
    """
    print(f"Cargando: {path}")
    from config import CSV_SEP
    df_raw = pd.read_csv(
        path,
        sep=CSV_SEP,
        dtype={"tienda_id": str, "generico_id": str},
        low_memory=False,
    )
    print(f"  {len(df_raw):,} filas × {len(df_raw.columns)} columnas")

    # Convertir columnas numéricas almacenadas como string
    for c in df_raw.select_dtypes(include=["object"]).columns:
        try:
            df_raw[c] = pd.to_numeric(df_raw[c])
        except (ValueError, TypeError):
            pass

    print("Aplicando build_features()...")
    df = build_features(df_raw)
    print(f"  → {len(df):,} filas × {len(df.columns)} columnas")

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].clip(lower=0)

    # Features disponibles (intersección FEATURE_COLS ∩ columnas reales)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  ⚠️  Features ausentes del dataset (se omiten): {missing}")

    # Categóricas → dtype category para LightGBM
    cat_cols_used = [c for c in CAT_COLS if c in feat_cols]
    for c in cat_cols_used:
        df[c] = df[c].astype("category")

    # Split temporal — fecha_venta ya es datetime post build_features()
    fecha_corte = pd.to_datetime(str(val_cutoff), format="%Y%m%d")
    train = df[df["fecha_venta"] < fecha_corte]
    val   = df[df["fecha_venta"] >= fecha_corte]

    print(f"\n  Split temporal (corte: {fecha_corte.date()}):")
    print(f"    Train : {len(train):>7,} filas  "
          f"({train['fecha_venta'].min().date()} → {train['fecha_venta'].max().date()})")
    print(f"    Val   : {len(val):>7,} filas  "
          f"({val['fecha_venta'].min().date()} → {val['fecha_venta'].max().date()})")

    if len(val) == 0:
        raise ValueError(
            f"Validación vacía con val_cutoff={val_cutoff}. "
            "Ajusta la fecha con --val-cutoff."
        )

    X_train = train[feat_cols]
    X_val   = val[feat_cols]
    y_train = train[TARGET].values
    y_val   = val[TARGET].values

    return X_train, X_val, y_train, y_val, feat_cols, cat_cols_used

# %% [markdown]
# #### LightGBM

# %%

def train_lgbm(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    cat_cols: list[str],
    feat_cols: list[str] | None = None,
    tuned_params: dict | None = None,
    n_estimators: int = N_ESTIMATORS,
) -> tuple:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    # Winsorize y_train a P99.5 antes de L2: evita que outliers (max=236, P99.5=8)
    # dominen el gradiente cuadrático. No afecta y_val (métricas sobre distribución real).
    _cap = float(np.percentile(y_train, 99.5))
    y_train = np.clip(y_train, None, _cap)

    params = {
        "objective":         "regression",      # L2/MSE — soporta monotone_constraints
        "metric":            "mae",             # early stopping por MAE (más interpretable)
        "n_estimators":      n_estimators,
        "learning_rate":     0.05,
        "num_leaves":        127,
        "min_child_samples": 20,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "random_state":      42,
        "n_jobs":            -1,
        "verbose":           -1,
    }
    if tuned_params:
        params.update(tuned_params)

    # Monotone constraints garantizan curvas de elasticidad monotónicas (alcance sección 6/13)
    mc = _build_monotone_list(feat_cols or list(X_train.columns))
    if mc:
        params["monotone_constraints"]        = mc
        params["monotone_constraints_method"] = "advanced"

    model = LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols if cat_cols else "auto",
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=200),
        ],
    )
    preds = model.predict(X_val).clip(min=0)
    return model, preds

# %% [markdown]
# #### XGBoost

# %%

def train_xgb(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    tuned_params: dict | None = None,
    n_estimators: int = N_ESTIMATORS,
) -> tuple:
    from xgboost import XGBRegressor

    # XGBoost no acepta dtype=category → convertir a int code
    X_tr = X_train.copy()
    X_vl = X_val.copy()
    for c in X_tr.select_dtypes(include=["category"]).columns:
        X_tr[c] = X_tr[c].cat.codes.astype("int16")
        X_vl[c] = X_vl[c].cat.codes.astype("int16")

    params = {
        "objective":             "reg:absoluteerror",
        "eval_metric":           "mae",
        "n_estimators":          n_estimators,
        "learning_rate":         0.05,
        "max_depth":             7,
        "min_child_weight":      20,
        "subsample":             0.8,
        "colsample_bytree":      0.8,
        "reg_alpha":             0.1,
        "reg_lambda":            0.1,
        "random_state":          42,
        "n_jobs":                -1,
        "tree_method":           "hist",
        "enable_categorical":    False,
        "early_stopping_rounds": 50,
    }
    if tuned_params:
        params.update(tuned_params)
    model = XGBRegressor(**params)
    model.fit(
        X_tr, y_train,
        eval_set=[(X_vl, y_val)],
        verbose=200,
    )
    preds = model.predict(X_vl).clip(min=0)
    return model, preds

# %% [markdown]
# #### Reporte de métricas

# %%

def _print_metrics(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline: np.ndarray | None = None,
) -> dict:
    w      = wmape(y_true, y_pred)
    m      = mae(y_true, y_pred)
    rs     = rmse(y_true, y_pred)
    b      = bias(y_true, y_pred)
    r2_val = r2(y_true, y_pred)
    la     = lift_accuracy(y_true, y_pred, baseline)
    bls    = bls_score(w, la, b)
    ok_w  = "✅" if w      <= WMAPE_OBJETIVO else ("⚠️ " if w      <= WMAPE_GREEN else "❌")
    ok_r2 = "✅" if r2_val >= R2_MIN else "⚠️ "
    ok_la = (
        "✅" if la >= LIFT_ACC_OBJETIVO else ("⚠️ " if la >= LIFT_ACC_MIN else "❌")
    ) if not np.isnan(la) else "─ "
    la_str = f"{la:.4f}{ok_la}" if not np.isnan(la) else "n/a (sin avg_units_4w)"
    print(f"  {name:<12s}  WMAPE={w*100:.2f}%{ok_w}  MAE={m:.4f}  "
          f"R²={r2_val:.4f}{ok_r2}  Bias={b:+.4f}  RMSE={rs:.4f}(inf)")
    print(f"  {'':12s}  LiftAcc={la_str}  BLS={bls:.4f}")
    return {"wmape": w, "mae": m, "rmse": rs, "bias": b, "r2": r2_val,
            "lift_accuracy": la, "bls": bls}


def _print_importance(model, feat_cols: list[str], top_n: int = 20, label: str = "") -> None:
    try:
        imp = pd.Series(model.feature_importances_, index=feat_cols).astype(float)
        total = imp.sum()
        imp_pct = (imp / total * 100) if total > 0 else imp
        imp_pct = imp_pct.sort_values(ascending=False).head(top_n)
        print(f"\n  Top {top_n} features por importancia ({label}, escala 0-100%):")
        print(f"  {'Feature':<35s} {'Importancia':>12s}")
        print("  " + "-" * 50)
        for feat, pct in imp_pct.items():
            bar = "█" * max(1, int(pct / imp_pct.max() * 25))
            print(f"  {feat:<35s} {pct:>10.2f}%  {bar}")
    except Exception as e:
        print(f"  (importancia no disponible: {e})")

# %% [markdown]
# #### Main

# %%

def main() -> None:
    reconfigure_stdout()

    parser = argparse.ArgumentParser(
        description="M1 Demand Response — Entrenamiento base LightGBM vs XGBoost",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data", type=Path, default=TRAIN_DATA_PATH,
        help="CSV de entrada (default: data/train-set-features.csv)",
    )
    parser.add_argument(
        "--val-cutoff", type=int, default=20251201,
        help="Fecha de corte train/val YYYYMMDD (default: 20251201)",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=N_ESTIMATORS,
        help=f"Número de árboles máximos para LGBM y XGBoost (default: {N_ESTIMATORS})",
    )
    args, _ = parser.parse_known_args()

    sep = "=" * 68
    print(f"\n{sep}")
    print("  M1 Demand Response — Entrenamiento final (Paso 2 de 2)")
    print("  Alcance: uplift en unidades vendidas con promo vs sin promo")
    print(f"  Métrica objetivo: WMAPE ≤ {WMAPE_OBJETIVO*100:.0f}% (deseable) | ≤ {WMAPE_GREEN*100:.0f}% (gate)")
    print(f"{sep}\n")

    # Cargar params tuneados — generados por tune.py (paso 1)
    lgbm_tuned, xgb_tuned = _load_tuned_params(INPUT_MODELS_DIR / "tune_meta.json")
    if lgbm_tuned or xgb_tuned:
        print("  Hiperparámetros: tune_meta.json encontrado — usando params tuneados")
        if lgbm_tuned:
            print(f"    LGBM: {list(lgbm_tuned.keys())}")
        if xgb_tuned:
            print(f"    XGB:  {list(xgb_tuned.keys())}")
    else:
        print("  ⚠️  tune_meta.json no encontrado — usando params base.")
        print("     Ejecuta tune.py primero para obtener mejores resultados.")

    X_train, X_val, y_train, y_val, feat_cols, cat_cols = load_and_prepare(
        args.data, args.val_cutoff
    )

    results: dict = {}

    # Baseline para Lift Accuracy: avg_units_4w como proxy (alcance §7.5.2)
    baseline_val = X_val["avg_units_4w"].values if "avg_units_4w" in X_val.columns else None

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("  LightGBM")
    print(f"{'─'*68}")
    lgbm_model, lgbm_preds = train_lgbm(
        X_train, X_val, y_train, y_val, cat_cols, feat_cols,
        tuned_params=lgbm_tuned, n_estimators=args.n_estimators,
    )
    results["lgbm"] = _print_metrics("LightGBM", y_val, lgbm_preds, baseline_val)
    _print_importance(lgbm_model, feat_cols, label="LightGBM gain")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("  XGBoost")
    print(f"{'─'*68}")
    xgb_model, xgb_preds = train_xgb(
        X_train, X_val, y_train, y_val,
        tuned_params=xgb_tuned, n_estimators=args.n_estimators,
    )
    results["xgb"] = _print_metrics("XGBoost", y_val, xgb_preds, baseline_val)
    _print_importance(xgb_model, feat_cols, label="XGBoost gain")

    # ── Comparación (BLS scorecard — alcance §7.5.2) ─────────────────────────
    print(f"\n{'─'*68}")
    print("  Comparación final — BLS scorecard")
    print(f"{'─'*68}")
    lgbm_w   = results["lgbm"]["wmape"]
    xgb_w    = results["xgb"]["wmape"]
    lgbm_bls = results["lgbm"]["bls"]
    xgb_bls  = results["xgb"]["bls"]
    winner       = "lgbm" if lgbm_bls >= xgb_bls else "xgb"
    winner_label = "LightGBM" if winner == "lgbm" else "XGBoost"

    print(f"  LightGBM  WMAPE={lgbm_w*100:.2f}%  BLS={lgbm_bls:.4f}")
    print(f"  XGBoost   WMAPE={xgb_w*100:.2f}%  BLS={xgb_bls:.4f}")
    print(f"  Ganador (BLS scorecard): {winner_label}")
    if min(lgbm_w, xgb_w) <= WMAPE_GREEN:
        print(f"  ✅ Objetivo WMAPE ≤ {WMAPE_GREEN*100:.0f}% alcanzado en val set")
    else:
        delta = min(lgbm_w, xgb_w) - WMAPE_GREEN
        print(f"  ⚠️  Objetivo no alcanzado — delta: +{delta*100:.2f}%")
        print("     Próximos pasos: tuning bayesiano (src/tune.py) [2] → CV temporal (src/cv.py) [3]")

    # Segmentar WMAPE por división para diagnóstico
    print(f"\n  WMAPE por segmento (val set, mejor modelo):")
    try:
        val_df = X_val.copy()
        val_df["_y_true"] = y_val
        val_df["_y_pred"] = lgbm_preds if winner == "lgbm" else xgb_preds
        if "division_desc" in val_df.columns:
            seg = val_df.groupby("division_desc").apply(
                lambda g: wmape(g["_y_true"].values, g["_y_pred"].values)
            ).sort_values()
            for div, w in seg.items():
                flag = "✅" if w <= WMAPE_GREEN else "⚠️ "
                print(f"    {flag} {str(div):<35s}  WMAPE={w*100:.2f}%")
    except Exception:
        pass

    # ── Guardar modelos ───────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("  Guardando modelos")
    print(f"{'─'*68}")
    lgbm_path = MODELS_DIR / "lgbm_base.pkl"
    xgb_path  = MODELS_DIR / "xgb_base.pkl"
    meta_path = MODELS_DIR / "train_meta.json"

    joblib.dump(lgbm_model, lgbm_path)
    joblib.dump(xgb_model,  xgb_path)

    meta = {
        "val_cutoff":    args.val_cutoff,
        "n_train":       int(len(y_train)),
        "n_val":         int(len(y_val)),
        "feature_cols":  feat_cols,
        "cat_cols":      cat_cols,
        "results":       results,
        "winner":        winner,
        "source_params": "tuned" if (lgbm_tuned or xgb_tuned) else "base",
        "data_path":     str(args.data),
        "alcance":       (
            "Predecir uplift en unidades vendidas con promo vs sin promo. "
            "Dataset: solo dias con transacciones reales (sale-only). "
            "No predice si el producto vendera o no."
        ),
    }
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"  {lgbm_path}")
    print(f"  {xgb_path}")
    print(f"  {meta_path}")
    print(f"\n{sep}")
    print("  Listo. Próximo paso: [3] CV temporal → python src/cv.py --n-splits 5 --gap-days 28")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
