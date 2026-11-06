# %% [markdown]
# ### promo-uplift-ml · Precisión Promocional
# src/explain.py — SHAP Analysis
# 
# Interpretabilidad del modelo champion (config.CHAMPION_MODEL).
# 
# FLUJO:
#     models/<CHAMPION_MODEL>  (xgb_base.pkl por defecto — config.py)
#         → SHAP TreeExplainer sobre val set
#         → Summary plot  (reports/shap/figures/shap_summary.png)
#         → Bar plot      (reports/shap/figures/shap_bar.png)
#         → Dependence plots top-5 features (reports/shap/figures/shap_dep_*.png)
#         → Waterfall de peor predicción (reports/shap/figures/shap_waterfall_worst.png)
#         → CSV con SHAP values medios (reports/shap/shap_importance.csv)
# 
# USO:
#     python src/explain.py
#     python src/explain.py --model models/xgb_base.pkl
#     python src/explain.py --max-display 25

# %%
from __future__ import annotations

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
from src.train import load_and_prepare
from src._console import reconfigure_stdout
from config import TRAIN_DATA_PATH, MODELS_DIR, REPORTS_SHAP as REPORTS, CHAMPION_MODEL

FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

try:
    get_ipython()
    _IN_NOTEBOOK = True
except NameError:
    _IN_NOTEBOOK = False
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

# %% [markdown]
# #### Análisis SHAP

# %%
def main() -> None:
    import argparse
    import shap

    reconfigure_stdout()

    parser = argparse.ArgumentParser(
        description="M1 Demand Response — SHAP Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=Path,
        default=MODELS_DIR / CHAMPION_MODEL,
        help=f"Modelo a analizar (default: models/{CHAMPION_MODEL} — config.py)",
    )
    parser.add_argument(
        "--data", type=Path,
        default=TRAIN_DATA_PATH,
    )
    parser.add_argument("--val-cutoff", type=int, default=20251201)
    parser.add_argument(
        "--max-display", type=int, default=20,
        help="Número de features en los plots (default: 20)",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Muestra aleatoria del val set para SHAP (default: todo el val set)",
    )
    args, _ = parser.parse_known_args()

    # Fallback a base si no existe tuned
    if not args.model.exists():
        alt = MODELS_DIR / "lgbm_base.pkl"
        if alt.exists():
            print(f"  ⚠️  {args.model.name} no encontrado, usando {alt.name}")
            args.model = alt
        else:
            raise FileNotFoundError(
                f"Modelo no encontrado: {args.model}\n"
                "Corre primero: python src/train.py"
            )

    sep = "=" * 68
    print(f"\n{sep}")
    print("  M1 Demand Response — SHAP Analysis (Semana 6)")
    print(f"  Modelo: {args.model.name}")
    print(f"{sep}\n")

    # ── Cargar datos y modelo ─────────────────────────────────────────────────
    X_train, X_val, y_train, y_val, feat_cols, cat_cols = load_and_prepare(
        args.data, args.val_cutoff
    )
    model = joblib.load(args.model)
    print(f"  Modelo cargado: {args.model.name}")

    if args.sample and args.sample < len(X_val):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_val), size=args.sample, replace=False)
        X_shap = X_val.iloc[idx]
        y_shap = y_val[idx]
        print(f"  Val sample: {len(X_shap):,} filas (de {len(X_val):,})")
    else:
        X_shap = X_val
        y_shap = y_val
        print(f"  Val set completo: {len(X_shap):,} filas")

    # ── Encode categoricals para XGBoost (entrenado con int codes, no category dtype) ──
    try:
        from xgboost import XGBRegressor as _XGBReg
        if isinstance(model, _XGBReg):
            X_shap = X_shap.copy()
            for c in X_shap.select_dtypes(include=["category"]).columns:
                X_shap[c] = X_shap[c].cat.codes.astype("int16")
    except ImportError:
        pass

    # ── SHAP TreeExplainer ────────────────────────────────────────────────────
    print("\n  Calculando SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    expected_val = float(explainer.expected_value)
    print(f"  expected_value (E[f(x)]): {expected_val:.4f}")

    # ── Importancia SHAP media ────────────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_signed   = shap_values.mean(axis=0)
    shap_df = pd.DataFrame({
        "feature":          feat_cols,
        "mean_abs_shap":    mean_abs_shap,
        "mean_shap":        mean_signed,
        "direction":        np.where(mean_signed > 0, "POSITIVO", "NEGATIVO"),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df["rank"] = shap_df.index + 1

    print(f"\n  Top {args.max_display} features por |SHAP| medio:")
    print(f"  {'Rank':<5s} {'Feature':<35s} {'|SHAP|':>8s}  {'Dirección'}")
    print("  " + "-" * 60)
    for _, row in shap_df.head(args.max_display).iterrows():
        arrow = "▲" if row["direction"] == "POSITIVO" else "▼"
        print(f"  {int(row['rank']):<5d} {row['feature']:<35s} "
              f"{row['mean_abs_shap']:>8.4f}  {arrow} {row['direction']}")

    csv_path = REPORTS / "shap_importance.csv"
    shap_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  Guardado: {csv_path}")

    # ── Summary plot (beeswarm) ───────────────────────────────────────────────
    print("\n  Generando plots SHAP...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_shap,
        feature_names=feat_cols,
        max_display=args.max_display,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Summary — M1 Demand Response", pad=12)
    plt.tight_layout()
    out = FIGURES / "shap_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    if _IN_NOTEBOOK:
        from IPython.display import display, Image
        display(Image(filename=str(out)))
    print(f"  {out}")

    # ── Bar plot (|SHAP| medio) ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(
        shap_values, X_shap,
        feature_names=feat_cols,
        max_display=args.max_display,
        plot_type="bar",
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Feature Importance — M1 Demand Response", pad=12)
    plt.tight_layout()
    out = FIGURES / "shap_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    if _IN_NOTEBOOK:
        from IPython.display import display, Image
        display(Image(filename=str(out)))
    print(f"  {out}")

    # ── Dependence plots — top 5 ──────────────────────────────────────────────
    top5 = shap_df.head(5)["feature"].tolist()
    feat_idx = {f: i for i, f in enumerate(feat_cols)}

    for feat in top5:
        fi = feat_idx[feat]
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            fi, shap_values, X_shap,
            feature_names=feat_cols,
            ax=ax, show=False,
        )
        ax.set_title(f"SHAP Dependence — {feat}", pad=10)
        plt.tight_layout()
        safe_name = feat.replace("/", "_").replace(" ", "_")
        out = FIGURES / f"shap_dep_{safe_name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        if _IN_NOTEBOOK:
            from IPython.display import display, Image
            display(Image(filename=str(out)))
        print(f"  {out}")

    # ── Waterfall — peor predicción ───────────────────────────────────────────
    preds     = model.predict(X_shap).clip(min=0)
    abs_error = np.abs(preds - y_shap)
    worst_idx = int(np.argmax(abs_error))

    expl_obj = shap.Explanation(
        values=shap_values[worst_idx],
        base_values=expected_val,
        data=X_shap.iloc[worst_idx].values,
        feature_names=feat_cols,
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(expl_obj, max_display=15, show=False)
    plt.title(
        f"SHAP Waterfall — peor predicción\n"
        f"y_true={y_shap[worst_idx]:.1f}  y_pred={preds[worst_idx]:.1f}  "
        f"error={abs_error[worst_idx]:.2f}",
        pad=10,
    )
    plt.tight_layout()
    out = FIGURES / "shap_waterfall_worst.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    if _IN_NOTEBOOK:
        from IPython.display import display, Image
        display(Image(filename=str(out)))
    print(f"  {out}")

    # ── Resumen de hallazgos ──────────────────────────────────────────────────
    top3 = shap_df.head(3)
    print(f"\n{'─'*68}")
    print("  Hallazgos clave SHAP:")
    print(f"  1. Feature más influyente: '{top3.iloc[0]['feature']}' "
          f"(|SHAP|={top3.iloc[0]['mean_abs_shap']:.4f}, "
          f"{top3.iloc[0]['direction']})")
    print(f"  2. '{top3.iloc[1]['feature']}' "
          f"(|SHAP|={top3.iloc[1]['mean_abs_shap']:.4f}, "
          f"{top3.iloc[1]['direction']})")
    print(f"  3. '{top3.iloc[2]['feature']}' "
          f"(|SHAP|={top3.iloc[2]['mean_abs_shap']:.4f}, "
          f"{top3.iloc[2]['direction']})")

    # Features con dirección negativa sorpresiva (pueden indicar leakage o efecto no intuitivo)
    neg_top10 = shap_df.head(10)[shap_df.head(10)["direction"] == "NEGATIVO"]
    if not neg_top10.empty:
        print(f"\n  Features top-10 con efecto NEGATIVO (revisar intuición):")
        for _, row in neg_top10.iterrows():
            print(f"    ▼ {row['feature']} (rank #{int(row['rank'])})")

    print(f"\n{sep}")
    print("  Listo. Plots en reports/shap/figures/  |  CSV en reports/shap/shap_importance.csv")
    print("  Próximo paso: [5] curvas de elasticidad → python src/elasticity.py --min-events 3")
    print(f"{sep}\n")

    from src._aws_triggers import trigger_pipeline
    trigger_pipeline(
        pipeline_name="promo-uplift-scoring",
        env_var="AUTO_TRIGGER_SCORING",
        execution_display_name="auto-trigger-post-training",
        reason="Training completado exitosamente",
    )


if __name__ == "__main__":
    main()
