# %% [markdown]
# ### promo-uplift-ml · Precisión Promocional
# src/cv.py — Validación cruzada temporal (5-fold)
# 
# Paso 3 de 8: Confirmar robustez temporal de LightGBM y XGBoost con los
# hiperparámetros tuneados. El reporte marca al champion y expone métricas
# de ambos algoritmos en la misma ejecución.
# 
# FLUJO:
#     train-set-features.csv
#         → build_features()
#         → TimeSeriesSplit 5-fold con gap=28d
#         → LightGBM + XGBoost (hiperparámetros de tune_meta.json o base si no existe)
#         → WMAPE, MAE, RMSE, Bias, LiftAcc por fold × algoritmo
#         → WMAPE por division_desc (agregado)
#         → Gate de calidad: WMAPE CV champion ≤ 0.35
#         → Guardar reporte en reports/cv/cv_report.json
# 
# USO:
#     python src/cv.py
#     python src/cv.py --n-splits 5 --gap-days 28
#     python src/cv.py --data data/train-set-features.csv

# %%
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

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
from src.train import wmape, mae, rmse, bias, lift_accuracy
from src._console import reconfigure_stdout
from config import (
    TRAIN_DATA_PATH, MODELS_DIR, REPORTS_CV as REPORTS, N_ESTIMATORS, CHAMPION_MODEL,
    WMAPE_GREEN, WMAPE_GATE, LIFT_ACC_MIN, TARGET,
)

REPORTS.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# #### Helpers de carga y configuración

# %%

def _load_full(path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    from config import CSV_SEP
    print(f"Cargando: {path}")
    df_raw = pd.read_csv(
        path,
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
    print(f"  → {len(df):,} filas × {len(df.columns)} columnas")

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].clip(lower=0)

    feat_cols     = [c for c in FEATURE_COLS if c in df.columns]
    cat_cols_used = [c for c in CAT_COLS if c in feat_cols]
    for c in cat_cols_used:
        df[c] = df[c].astype("category")

    df = df.sort_values("fecha_venta").reset_index(drop=True)
    return df, feat_cols, cat_cols_used


def _lgbm_params(tune_meta_path: Path, n_estimators: int = N_ESTIMATORS) -> dict:
    base = {
        "objective":         "regression",
        "metric":            "mae",
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
    if tune_meta_path.exists():
        meta    = json.loads(tune_meta_path.read_text(encoding="utf-8"))
        lgbm_meta = meta.get("lgbm", meta)  # formato nuevo {"lgbm": {...}} con fallback al viejo
        tuned   = lgbm_meta.get("best_params", {})
        base.update(tuned)
        print(f"  LightGBM: tune_meta.json (trial #{lgbm_meta.get('best_trial')})")
    else:
        print("  LightGBM: base (tune_meta.json no encontrado)")
    return base


def _xgb_params(tune_meta_path: Path, n_estimators: int = N_ESTIMATORS) -> dict:
    base = {
        "objective":             "reg:absoluteerror",
        "eval_metric":           "mae",
        "n_estimators":          n_estimators,
        "learning_rate":         0.05,
        "max_depth":             6,
        "min_child_weight":      20,
        "subsample":             0.8,
        "colsample_bytree":      0.8,
        "reg_alpha":             0.1,
        "reg_lambda":            1.0,
        "random_state":          42,
        "n_jobs":                -1,
        "tree_method":           "hist",
        "enable_categorical":    False,
        "early_stopping_rounds": 50,
    }
    if tune_meta_path.exists():
        meta     = json.loads(tune_meta_path.read_text(encoding="utf-8"))
        xgb_meta = meta.get("xgb", {})
        tuned    = xgb_meta.get("best_params", {})
        base.update(tuned)
        print(f"  XGBoost:  tune_meta.json (trial #{xgb_meta.get('best_trial')})")
    else:
        print("  XGBoost:  base (tune_meta.json no encontrado)")
    return base

# %% [markdown]
# #### Ejecución de folds y resumen

# %%

def _run_cv(
    algo_label: str,
    predict_fn,
    df: pd.DataFrame,
    feat_cols: list[str],
    cat_cols: list[str],
    fold_edges: list,
    gap: pd.Timedelta,
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Corre todos los folds para un algoritmo. predict_fn(X_tr, y_tr, X_vl, y_vl, cat_cols) → preds."""
    fold_results: list[dict]      = []
    div_accum:    list[pd.DataFrame] = []
    last_date = df["fecha_venta"].max()

    print(f"\n  {'─'*64}")
    print(f"  [{algo_label}]")
    print(f"  {'Fold':<6s} {'Train':>8s} {'Val':>7s} {'WMAPE':>8s} {'MAE':>8s} "
          f"{'RMSE':>8s} {'Bias':>8s} {'Status'}")
    print(f"  {'─'*64}")

    for i, cut in enumerate(fold_edges):
        val_start = cut + gap
        val_end   = fold_edges[i + 1] if i + 1 < len(fold_edges) else last_date

        tr = df[df["fecha_venta"] < cut]
        vl = df[(df["fecha_venta"] >= val_start) & (df["fecha_venta"] <= val_end)]

        if len(vl) == 0:
            print(f"  Fold {i+1}: val vacío — omitido")
            continue

        X_tr = tr[feat_cols]
        X_vl = vl[feat_cols]
        y_tr = tr[TARGET].values
        y_vl = vl[TARGET].values

        preds = predict_fn(X_tr, y_tr, X_vl, y_vl, cat_cols)

        w  = wmape(y_vl, preds)
        m  = mae(y_vl, preds)
        r  = rmse(y_vl, preds)
        b  = bias(y_vl, preds)
        baseline_vl = vl["avg_units_4w"].values if "avg_units_4w" in vl.columns else None
        la = lift_accuracy(y_vl, preds, baseline_vl)
        status = "✅" if w <= WMAPE_GREEN else ("⚠️ " if w <= WMAPE_GATE else "❌")
        la_str = f"{la:.4f}" if not np.isnan(la) else "  n/a"

        fold_results.append({
            "fold": i + 1, "n_train": len(tr), "n_val": len(vl),
            "wmape": w, "mae": m, "rmse": r, "bias": b,
            "lift_accuracy": float(la) if not np.isnan(la) else None,
            "train_end": str(pd.Timestamp(cut).date()),
            "val_start": str(pd.Timestamp(val_start).date()),
            "val_end":   str(pd.Timestamp(val_end).date()),
        })

        print(f"  {i+1:<6d} {len(tr):>8,} {len(vl):>7,} {w:>8.4f} "
              f"{m:>8.4f} {r:>8.4f} {b:>+8.4f}  LA={la_str}  {status}")

        if "division_desc" in vl.columns:
            tmp = vl[["division_desc"]].copy()
            tmp["_y_true"] = y_vl
            tmp["_y_pred"] = preds
            tmp["_fold"]   = i + 1
            div_accum.append(tmp)

    return fold_results, div_accum


def _summarize(
    algo_label: str,
    fold_results: list[dict],
    div_accum: list[pd.DataFrame],
    n_splits: int,
    is_champion: bool,
) -> dict:
    """Imprime resumen de un algoritmo y retorna dict con métricas agregadas."""
    wmapes = [r["wmape"] for r in fold_results]
    mean_w = float(np.mean(wmapes))
    std_w  = float(np.std(wmapes))
    min_w  = float(np.min(wmapes))
    max_w  = float(np.max(wmapes))
    la_vals = [r["lift_accuracy"] for r in fold_results if r["lift_accuracy"] is not None]
    mean_la = float(np.mean(la_vals)) if la_vals else float("nan")

    champ_tag = "  ★ CHAMPION" if is_champion else ""
    print(f"\n  [{algo_label}]{champ_tag}")
    print(f"  CV WMAPE:   {mean_w*100:.2f}% ± {std_w*100:.2f}%  "
          f"[min={min_w*100:.2f}%, max={max_w*100:.2f}%]")
    print(f"  CV MAE:     {np.mean([r['mae']  for r in fold_results]):.4f}")
    print(f"  CV RMSE:    {np.mean([r['rmse'] for r in fold_results]):.4f}  (informativo)")
    print(f"  CV Bias:    {np.mean([r['bias'] for r in fold_results]):+.4f}")
    if not np.isnan(mean_la):
        la_flag = "✅" if mean_la >= LIFT_ACC_MIN else "⚠️ "
        print(f"  CV LiftAcc: {mean_la:.4f} {la_flag} (mínimo ≥ {LIFT_ACC_MIN:.0%} — alcance §7.5.2)")

    gate_green  = mean_w <= WMAPE_GREEN
    gate_passed = mean_w <= WMAPE_GATE
    if gate_green:
        print(f"  ✅ Gate: WMAPE CV={mean_w*100:.2f}% ≤ {WMAPE_GREEN:.0%}")
    elif gate_passed:
        print(f"  ⚠️  Gate: WMAPE CV={mean_w*100:.2f}% entre {WMAPE_GREEN:.0%} y {WMAPE_GATE:.0%}")
    else:
        print(f"  ❌ Gate: WMAPE CV={mean_w*100:.2f}% supera {WMAPE_GATE:.0%}")

    div_dict: dict[str, float] = {}
    if div_accum:
        div_df = pd.concat(div_accum, ignore_index=True)
        print(f"\n  WMAPE por división ({n_splits} folds):")
        div_summary = (
            div_df.groupby("division_desc")
            .apply(lambda g: wmape(g["_y_true"].values, g["_y_pred"].values))
            .sort_values()
        )
        for div, w in div_summary.items():
            flag = "✅" if w <= WMAPE_GREEN else "⚠️ "
            print(f"    {flag} {str(div):<35s}  WMAPE={w*100:.2f}%")
            div_dict[str(div)] = round(w, 6)

    return {
        "algo":             algo_label,
        "is_champion":      is_champion,
        "cv_wmape_mean":    mean_w,
        "cv_wmape_std":     std_w,
        "cv_wmape_min":     min_w,
        "cv_wmape_max":     max_w,
        "cv_mae_mean":      float(np.mean([r["mae"]  for r in fold_results])),
        "cv_rmse_mean":     float(np.mean([r["rmse"] for r in fold_results])),
        "cv_bias_mean":     float(np.mean([r["bias"] for r in fold_results])),
        "cv_lift_acc_mean": float(mean_la) if not np.isnan(mean_la) else None,
        "gate_passed":      gate_passed,
        "gate_green":       gate_green,
        "folds":            fold_results,
        "wmape_by_division": div_dict,
    }

# %% [markdown]
# #### Main

# %%
def main() -> None:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    from xgboost import XGBRegressor

    reconfigure_stdout()

    parser = argparse.ArgumentParser(
        description="M1 Demand Response — CV temporal 5-fold (LightGBM + XGBoost)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data",      type=Path, default=TRAIN_DATA_PATH)
    parser.add_argument("--n-splits",  type=int,  default=5)
    parser.add_argument("--gap-days",  type=int,  default=28,
                        help="Días de gap entre train y val de cada fold (default: 28)")
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS,
                        help=f"Árboles máximos para LGBM y XGBoost (default: {N_ESTIMATORS})")
    args, _ = parser.parse_known_args()

    sep = "=" * 68
    print(f"\n{sep}")
    print("  M1 Demand Response — CV Temporal (Paso 3 de 8)")
    print(f"  Folds: {args.n_splits}  |  Gap: {args.gap_days} días  |  Algoritmos: LGBM + XGB")
    print(f"  Champion: {CHAMPION_MODEL}")
    print(f"{sep}\n")

    df, feat_cols, cat_cols = _load_full(args.data)

    tune_meta_path = MODELS_DIR / "tune_meta.json"
    print("\n  Cargando hiperparámetros...")
    lgbm_p = _lgbm_params(tune_meta_path, args.n_estimators)
    xgb_p  = _xgb_params(tune_meta_path, args.n_estimators)

    # ── Construir folds temporales ────────────────────────────────────────────
    dates   = df["fecha_venta"].sort_values().unique()
    n_dates = len(dates)
    gap     = pd.Timedelta(days=args.gap_days)

    fold_edges = [
        dates[int(n_dates * (k + 1) / (args.n_splits + 1))]
        for k in range(args.n_splits)
    ]

    print(f"\n  Estructura de folds ({args.n_splits} cortes equidistantes):")
    for i, cut in enumerate(fold_edges):
        val_start = cut + gap
        val_end   = fold_edges[i + 1] if i + 1 < len(fold_edges) else dates[-1]
        tr = df[df["fecha_venta"] < cut]
        vl = df[(df["fecha_venta"] >= val_start) & (df["fecha_venta"] <= val_end)]
        print(f"  Fold {i+1}: train hasta {pd.Timestamp(cut).date()}  "
              f"| val {pd.Timestamp(val_start).date()} → {pd.Timestamp(val_end).date()}  "
              f"| train={len(tr):>7,}  val={len(vl):>6,}")

    # ── Funciones de ajuste por algoritmo ────────────────────────────────────
    def _fit_lgbm(X_tr, y_tr, X_vl, y_vl, cat_cols_):
        model = LGBMRegressor(**lgbm_p)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            categorical_feature=cat_cols_ if cat_cols_ else "auto",
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_evaluation(period=-1),
            ],
        )
        return model.predict(X_vl).clip(min=0)

    def _fit_xgb(X_tr, y_tr, X_vl, y_vl, _cat_cols):
        X_tr_xgb = X_tr.copy()
        X_vl_xgb = X_vl.copy()
        for c in X_tr_xgb.select_dtypes(include=["category"]).columns:
            X_tr_xgb[c] = X_tr_xgb[c].cat.codes.astype("int16")
            X_vl_xgb[c] = X_vl_xgb[c].cat.codes.astype("int16")
        model = XGBRegressor(**xgb_p)
        model.fit(X_tr_xgb, y_tr, eval_set=[(X_vl_xgb, y_vl)], verbose=False)
        return model.predict(X_vl_xgb).clip(min=0)

    # ── Correr CV para ambos ──────────────────────────────────────────────────
    lgbm_folds, lgbm_div = _run_cv(
        "LightGBM", _fit_lgbm, df, feat_cols, cat_cols, fold_edges, gap
    )
    xgb_folds, xgb_div = _run_cv(
        "XGBoost", _fit_xgb, df, feat_cols, cat_cols, fold_edges, gap
    )

    # ── Resumen por algoritmo ─────────────────────────────────────────────────
    is_lgbm_champ = "lgbm" in CHAMPION_MODEL.lower()
    is_xgb_champ  = "xgb"  in CHAMPION_MODEL.lower()

    print(f"\n{sep}")
    print("  Resumen CV por algoritmo")
    lgbm_summary = _summarize("LightGBM", lgbm_folds, lgbm_div, args.n_splits, is_lgbm_champ)
    xgb_summary  = _summarize("XGBoost",  xgb_folds,  xgb_div,  args.n_splits, is_xgb_champ)

    champ_summary = xgb_summary if is_xgb_champ else lgbm_summary

    # ── Guardar reporte ───────────────────────────────────────────────────────
    # Top-level keys = champion para compatibilidad con dryrun.py
    report = {
        "champion_model":   CHAMPION_MODEL,
        "n_splits":         args.n_splits,
        "gap_days":         args.gap_days,
        "cv_wmape_mean":    champ_summary["cv_wmape_mean"],
        "cv_wmape_std":     champ_summary["cv_wmape_std"],
        "cv_wmape_min":     champ_summary["cv_wmape_min"],
        "cv_wmape_max":     champ_summary["cv_wmape_max"],
        "cv_mae_mean":      champ_summary["cv_mae_mean"],
        "cv_rmse_mean":     champ_summary["cv_rmse_mean"],
        "cv_bias_mean":     champ_summary["cv_bias_mean"],
        "cv_lift_acc_mean": champ_summary["cv_lift_acc_mean"],
        "gate_passed":      champ_summary["gate_passed"],
        "gate_green":       champ_summary["gate_green"],
        "by_algo": {
            "lgbm": lgbm_summary,
            "xgb":  xgb_summary,
        },
    }
    out_path = REPORTS / "cv_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'─'*68}")
    print(f"  Reporte guardado: {out_path}")

    if not report["gate_green"]:
        mean_w = report["cv_wmape_mean"]
        print(f"\n{sep}")
        print("  ❌  GATE DE CALIDAD FALLADO — pipeline de entrenamiento detenido.")
        print(f"  WMAPE CV champion = {mean_w*100:.2f}%  >  25%  (BBP §Criterios de Aceptación)")
        print(f"  El modelo no cumple el umbral mínimo de precisión funcional.")
        print(f"  Acción: revisar datos, features o hiperparámetros y re-ejecutar el pipeline de entrenamiento.")
        print(f"{sep}\n")
        sys.exit(1)

    print(f"\n{sep}")
    print("  ✅  Gate de calidad superado. Próximo paso: [4] SHAP analysis → python src/explain.py")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
