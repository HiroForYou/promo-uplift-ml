# %% [markdown]
# ### promo-uplift-ml · Precisión Promocional
# src/tune.py — Búsqueda bayesiana de hiperparámetros
# 
# EJECUTAR ANTES DE train.py. Tunea LightGBM y XGBoost por separado con Optuna
# y guarda los mejores params en models/tune_meta.json. train.py los carga
# automáticamente para entrenar ambos modelos en su mejor versión antes de
# hacer la comparación BLS y seleccionar el champion.
# 
# FLUJO:
#     train-set-features.csv
#         → build_features()
#         → split temporal (mismo corte que train.py)
#         → Optuna TPE — n_trials por algoritmo
#         → mejores params en models/tune_meta.json  (sin modelos — train.py los entrena)
# 
# USO:
#     python src/tune.py
#     python src/tune.py --n-trials 50
#     python src/tune.py --data data/train-set-features.csv --val-cutoff 20251201 --n-trials 50

# %%
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
except NameError:
    _cwd = Path().resolve()
    sys.path.insert(0, str(_cwd if (_cwd / "config.py").exists() else _cwd.parent))
from src.train import load_and_prepare, wmape, _build_monotone_list
from src._console import reconfigure_stdout
from config import TRAIN_DATA_PATH, MODELS_DIR, N_ESTIMATORS

MODELS_DIR.mkdir(exist_ok=True)

# %% [markdown]
# #### Objetivos Optuna

# %%

def _objective_lgbm(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    cat_cols: list[str],
    feat_cols: list[str],
) -> float:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    _cap = float(np.percentile(y_train, 99.5))
    y_tr = np.clip(y_train, None, _cap)

    params = {
        "objective":         "regression",
        "metric":            "mae",
        "n_estimators":      N_ESTIMATORS,
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":      trial.suggest_int("bagging_freq", 1, 10),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "random_state":      42,
        "n_jobs":            -1,
        "verbose":           -1,
    }
    mc = _build_monotone_list(feat_cols)
    if mc:
        params["monotone_constraints"]        = mc
        params["monotone_constraints_method"] = "advanced"

    model = LGBMRegressor(**params)
    model.fit(
        X_train, y_tr,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols if cat_cols else "auto",
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=-1),
        ],
    )
    return wmape(y_val, model.predict(X_val).clip(min=0))


def _objective_xgb(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> float:
    from xgboost import XGBRegressor

    X_tr = X_train.copy()
    X_vl = X_val.copy()
    for c in X_tr.select_dtypes(include=["category"]).columns:
        X_tr[c] = X_tr[c].cat.codes.astype("int16")
        X_vl[c] = X_vl[c].cat.codes.astype("int16")

    params = {
        "objective":             "reg:absoluteerror",
        "eval_metric":           "mae",
        "n_estimators":          N_ESTIMATORS,
        "learning_rate":         trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth":             trial.suggest_int("max_depth", 3, 10),
        "min_child_weight":      trial.suggest_int("min_child_weight", 10, 100),
        "subsample":             trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":      trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":             trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda":            trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "random_state":          42,
        "n_jobs":                -1,
        "tree_method":           "hist",
        "enable_categorical":    False,
        "early_stopping_rounds": 50,
    }
    model = XGBRegressor(**params)
    model.fit(X_tr, y_train, eval_set=[(X_vl, y_val)], verbose=False)
    return wmape(y_val, model.predict(X_vl).clip(min=0))

# %% [markdown]
# #### Helper: ejecutar estudio

# %%

def _run_study(label: str, objective_fn, n_trials: int) -> optuna.Study:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    def _cb(s: optuna.Study, t: optuna.Trial) -> None:
        if t.number % 10 == 0 or t.number == 0:
            print(f"    Trial {t.number:>3d}  WMAPE={t.value*100:.2f}%  "
                  f"best={s.best_value*100:.2f}%")

    print(f"\n  [{label}] {n_trials} trials (TPE sampler)...")
    study.optimize(objective_fn, n_trials=n_trials, callbacks=[_cb], show_progress_bar=True)
    return study

# %% [markdown]
# #### Main

# %%
def main() -> None:
    reconfigure_stdout()

    parser = argparse.ArgumentParser(
        description="M1 Demand Response — Búsqueda bayesiana (LightGBM + XGBoost)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data",       type=Path, default=TRAIN_DATA_PATH)
    parser.add_argument("--val-cutoff", type=int,  default=20251201)
    parser.add_argument("--n-trials",   type=int,  default=50,
                        help="Trials por algoritmo (default: 50; total: 100)")
    args, _ = parser.parse_known_args()

    sep = "=" * 68
    print(f"\n{sep}")
    print("  M1 Demand Response — Búsqueda bayesiana (Paso 1 de 2)")
    print(f"  {args.n_trials} trials × 2 algoritmos = {args.n_trials*2} trials totales")
    print(f"  Resultado: tune_meta.json  →  train.py lo cargará en el paso 2")
    print(f"{sep}\n")

    X_train, X_val, y_train, y_val, feat_cols, cat_cols = load_and_prepare(
        args.data, args.val_cutoff
    )

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("  LightGBM")
    print(f"{'─'*68}")
    lgbm_study = _run_study(
        "LGBM",
        lambda t: _objective_lgbm(t, X_train, X_val, y_train, y_val, cat_cols, feat_cols),
        args.n_trials,
    )
    lgbm_best = lgbm_study.best_trial
    print(f"\n  Mejor WMAPE: {lgbm_best.value*100:.2f}%  (trial #{lgbm_best.number})")
    for k, v in lgbm_best.params.items():
        print(f"    {k:<25s} = {v}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("  XGBoost")
    print(f"{'─'*68}")
    xgb_study = _run_study(
        "XGB",
        lambda t: _objective_xgb(t, X_train, X_val, y_train, y_val),
        args.n_trials,
    )
    xgb_best = xgb_study.best_trial
    print(f"\n  Mejor WMAPE: {xgb_best.value*100:.2f}%  (trial #{xgb_best.number})")
    for k, v in xgb_best.params.items():
        print(f"    {k:<25s} = {v}")

    # ── Comparativa tuning ────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("  Mejor WMAPE por algoritmo en tuning:")
    print(f"  LightGBM: {lgbm_best.value*100:.2f}%")
    print(f"  XGBoost:  {xgb_best.value*100:.2f}%")
    print(f"  (train.py entrenará ambos con estos params y hará el BLS definitivo)")

    # ── Guardar tune_meta.json ────────────────────────────────────────────────
    meta_path = MODELS_DIR / "tune_meta.json"
    meta = {
        "val_cutoff": args.val_cutoff,
        "n_trials":   args.n_trials,
        "data_path":  str(args.data),
        "lgbm": {
            "best_trial":  lgbm_best.number,
            "best_wmape":  lgbm_best.value,
            "best_params": lgbm_best.params,
        },
        "xgb": {
            "best_trial":  xgb_best.number,
            "best_wmape":  xgb_best.value,
            "best_params": xgb_best.params,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'─'*68}")
    print(f"  {meta_path}")
    print(f"\n{sep}")
    print("  Listo. Próximo paso: [2] entrenamiento final → python src/train.py")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
