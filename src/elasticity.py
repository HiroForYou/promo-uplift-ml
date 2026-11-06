# %% [markdown]
# ### promo-uplift-ml · Precisión Promocional
# src/elasticity.py — Curvas de elasticidad
# 
# Semana 9: Generar curvas de uplift por tienda × genérico × mecánica.
# 
# FLUJO:
#     models/<CHAMPION_MODEL> + train-set-features.csv
#         → Última observación por combo (tienda × genérico)
#         → Simular 9 niveles de descuento × mecánicas válidas (batch predict)
#         → Verificar monotonicidad (más descuento → más uplift)
#         → Plots: curva media global, top-5 mecánicas, heatmap por división
#         → reports/elasticity/promo_elasticity_curves.parquet  +  reports/elasticity/elasticity_summary.csv
#         → reports/elasticity/figures/elasticity_*.png
# 
# DECISIONES DE ALCANCE:
#     - Solo combos con ≥ 3 eventos históricos en la mecánica (filtro del BBP)
#     - Descuentos fuera del rango histórico marcados como is_extrapolated=True
#     - Predicción en batch (un solo model.predict) — no fila por fila
# 
# USO:
#     python src/elasticity.py
#     python src/elasticity.py --model models/xgb_base.pkl
#     python src/elasticity.py --min-events 3 --discount-levels 0,10,20,30,40,50

# %%
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
except NameError:
    _cwd = Path().resolve()
    sys.path.insert(0, str(_cwd if (_cwd / "config.py").exists() else _cwd.parent))
from src.features import build_features, FEATURE_COLS, CAT_COLS, derive_tipo_mecanica
from src._console import reconfigure_stdout
from config import TRAIN_DATA_PATH, MODELS_DIR, REPORTS_ELASTICITY as REPORTS, CHAMPION_MODEL, ELASTICITY_MIN_EVENTS, ELAST_MONO_MIN, TARGET

FIGURES = REPORTS / "figures"
REPORTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

DEFAULT_DISCOUNTS = [0, 5, 10, 15, 20, 25, 30, 40, 50]

try:
    get_ipython()
    _IN_NOTEBOOK = True
except NameError:
    _IN_NOTEBOOK = False
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

# %% [markdown]
# #### Carga de datos

# %%

def _load_data(path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    from config import CSV_SEP
    df_raw = pd.read_csv(
        path,
        sep=CSV_SEP,
        dtype={"tienda_id": str, "generico_id": str},
        low_memory=False,
    )
    for c in df_raw.select_dtypes(include=["object"]).columns:
        try:
            df_raw[c] = pd.to_numeric(df_raw[c])
        except (ValueError, TypeError):
            pass

    df = build_features(df_raw)
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].clip(lower=0)

    feat_cols     = [c for c in FEATURE_COLS if c in df.columns]
    cat_cols_used = [c for c in CAT_COLS if c in feat_cols]
    for c in cat_cols_used:
        df[c] = df[c].astype("category")

    return df, feat_cols, cat_cols_used

# %% [markdown]
# #### Monotonicidad

# %%

def _check_monotonicity(group: pd.DataFrame) -> bool:
    units = group.sort_values("discount_pct")["predicted_units"].values
    return bool(np.all(np.diff(units) >= -1e-6))


def _enforce_monotonicity(elasticity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-proceso estándar en uplift modeling: isotonic regression por combo×mecánica.
    Fuerza que a mayor descuento → mayor uplift predicho.
    Los valores ajustados se marcan con is_monotone_adjusted=True.
    Alternativa más limpia pero que requiere reentrenamiento: monotone_constraints en LightGBM.
    """
    from sklearn.isotonic import IsotonicRegression

    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    rows = []
    for _, grp in elasticity_df.groupby(
        ["tienda_id", "generico_id", "nombre_promo"]
    ):
        grp = grp.sort_values("discount_pct").copy()
        raw = grp["predicted_units"].values
        smoothed = ir.fit_transform(grp["discount_pct"].values.astype(float), raw)
        grp["predicted_units"]      = np.round(smoothed, 4)
        grp["is_monotone_adjusted"] = ~np.isclose(raw, smoothed, atol=1e-4)
        rows.append(grp)

    return pd.concat(rows, ignore_index=True)

# %% [markdown]
# #### Construcción del batch de simulaciones

# %%

def _build_sim_batch(
    latest: pd.DataFrame,
    mecanicas_validas: pd.DataFrame,
    feat_cols: list[str],
    discount_levels: list[int],
) -> pd.DataFrame:
    """Construye el DataFrame de todas las simulaciones de una vez."""
    rows = []
    for _, mv_row in tqdm(
        mecanicas_validas.iterrows(),
        total=len(mecanicas_validas),
        desc="  Armando simulaciones",
        unit="combo×mec",
    ):
        tid      = mv_row["tienda_id"]
        gid      = mv_row["generico_id"]
        mecanica = mv_row["nombre_promo"]
        max_hist = float(mv_row.get("max_disc_historico", 0))

        # tipo_mecanica de esta mecánica simulada (derivado una vez por mecánica)
        tipo_mec = str(derive_tipo_mecanica(pd.Series([mecanica])).iloc[0])

        base_mask = (latest["tienda_id"] == tid) & (latest["generico_id"] == gid)
        if base_mask.sum() == 0:
            continue
        base_row = latest[base_mask].iloc[0]

        # Precio base y unidades baseline para calcular descuento_total consistente
        # pct = desc / (monto + desc)  →  desc = pct * monto / (1 - pct)
        precio_base = float(
            base_row.get("precio_venta_bruto")
            or base_row.get("precio_venta")
            or base_row.get("precio_efectivo")
            or 1
        )
        units_base = max(float(base_row.get("avg_units_4w") or 1), 1)
        monto_base = precio_base * units_base  # monto sin descuento (referencia)

        for disc in discount_levels:
            sim = {f: base_row[f] for f in feat_cols if f in base_row.index}

            disc_frac = disc / 100
            # Actualizar pct y descuento_total de forma consistente entre sí
            if "pct_descuento_calc" in feat_cols:
                sim["pct_descuento_calc"] = disc_frac
            if "descuento_total" in feat_cols:
                # desc = pct * monto_base / (1 - pct); si disc=0 → desc=0
                sim["descuento_total"] = (disc_frac * monto_base / (1 - disc_frac)
                                          if disc_frac < 1.0 else monto_base * 10)
            if "tiene_promo_flag" in feat_cols:
                sim["tiene_promo_flag"] = 1 if disc > 0 else 0
            if "en_calendario_promo" in feat_cols and disc > 0:
                sim["en_calendario_promo"] = 1
            # Inyectar tipo_mecanica de la mecánica simulada (no la del base_row)
            if "tipo_mecanica" in feat_cols:
                sim["tipo_mecanica"] = tipo_mec

            sim["_tienda_id"]      = tid
            sim["_generico_id"]    = gid
            sim["_nombre_promo"]   = mecanica
            sim["_discount_pct"]   = disc
            sim["_is_extrapolated"]= bool(disc / 100 > max_hist * 1.05)
            rows.append(sim)

    return pd.DataFrame(rows)

# %% [markdown]
# #### Plots

# %%

def _plot_global_curve(summary: pd.DataFrame, out: Path) -> None:
    agg = (
        summary.groupby("discount_pct")["predicted_units_mean"]
        .mean()
        .reset_index()
    )
    _, ax = plt.subplots(figsize=(9, 5))
    ax.plot(agg["discount_pct"], agg["predicted_units_mean"],
            marker="o", linewidth=2, color="#1f77b4")
    ax.fill_between(agg["discount_pct"], agg["predicted_units_mean"],
                    alpha=0.12, color="#1f77b4")
    ax.set_xlabel("Descuento (%)", fontsize=12)
    ax.set_ylabel("Unidades predichas (media)", fontsize=12)
    ax.set_title("Curva de elasticidad global — M1 Demand Response\n"
                 "(media de todos los combos y mecánicas)", fontsize=13)
    ax.grid(axis="y", alpha=0.4)
    ax.set_xticks(agg["discount_pct"])
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    if _IN_NOTEBOOK:
        from IPython.display import display, Image
        display(Image(filename=str(out)))


def _plot_by_mecanica(summary: pd.DataFrame, top_n: int, out: Path) -> None:
    top_mec = (
        summary.groupby("nombre_promo")["n_combos"]
        .sum()
        .nlargest(top_n)
        .index.tolist()
    )
    _, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    for i, mec in enumerate(top_mec):
        sub = summary[summary["nombre_promo"] == mec].sort_values("discount_pct")
        ax.plot(sub["discount_pct"], sub["predicted_units_mean"],
                marker="o", label=mec[:40], color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel("Descuento (%)", fontsize=12)
    ax.set_ylabel("Unidades predichas (media)", fontsize=12)
    ax.set_title(f"Curvas de elasticidad — Top {top_n} mecánicas por volumen", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.4)
    ax.set_xticks(summary["discount_pct"].unique())
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    if _IN_NOTEBOOK:
        from IPython.display import display, Image
        display(Image(filename=str(out)))


def _plot_heatmap(elasticity_df: pd.DataFrame, out: Path) -> None:
    if "division_desc" not in elasticity_df.columns:
        return

    pivot = (
        elasticity_df
        .groupby(["division_desc", "discount_pct"])["predicted_units"]
        .mean()
        .unstack("discount_pct")
    )
    _, ax = plt.subplots(figsize=(11, max(4, len(pivot) * 0.7 + 1)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{d}%" for d in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xlabel("Descuento (%)", fontsize=11)
    ax.set_title("Heatmap de uplift medio por división × descuento", fontsize=13)
    plt.colorbar(im, ax=ax, label="Unidades predichas (media)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    if _IN_NOTEBOOK:
        from IPython.display import display, Image
        display(Image(filename=str(out)))

# %% [markdown]
# #### Main

# %%
def main() -> None:
    reconfigure_stdout()

    parser = argparse.ArgumentParser(
        description="M1 Demand Response — Curvas de elasticidad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",   type=Path, default=MODELS_DIR / CHAMPION_MODEL)
    parser.add_argument("--data",    type=Path, default=TRAIN_DATA_PATH)
    parser.add_argument("--min-events", type=int, default=ELASTICITY_MIN_EVENTS)
    parser.add_argument(
        "--discount-levels", type=str,
        default=",".join(map(str, DEFAULT_DISCOUNTS)),
    )
    parser.add_argument("--top-mecanicas", type=int, default=5,
                        help="Top N mecánicas en el plot (default: 5)")
    args, _ = parser.parse_known_args()

    discount_levels = [int(x) for x in args.discount_levels.split(",")]

    if not args.model.exists():
        alt = MODELS_DIR / "lgbm_base.pkl"
        print(f"  ⚠️  {args.model.name} no encontrado, usando {alt.name}")
        args.model = alt

    sep = "=" * 68
    print(f"\n{sep}")
    print("  M1 Demand Response — Curvas de elasticidad (Semana 9)")
    print(f"  Descuentos: {discount_levels}")
    print(f"  Min eventos/mecánica/combo: {args.min_events}")
    print(f"{sep}\n")

    # ── Cargar ────────────────────────────────────────────────────────────────
    print("Cargando datos y modelo...")
    df, feat_cols, cat_cols = _load_data(args.data)
    model = joblib.load(args.model)
    print(f"  Modelo: {args.model.name}")

    # ── Mecánicas válidas ─────────────────────────────────────────────────────
    if "nombre_promo" not in df.columns:
        df["nombre_promo"] = df["tiene_promo_flag"].map({1: "PROMO", 0: "SIN_PROMO"})

    eventos = (
        df[df["tiene_promo_flag"] == 1]
        .groupby(["tienda_id", "generico_id", "nombre_promo"])
        .size()
        .reset_index(name="n_eventos")
    )
    mecanicas_validas = eventos[eventos["n_eventos"] >= args.min_events].copy()

    n_combos_total   = df.groupby(["tienda_id", "generico_id"]).ngroups
    n_combos_validos = mecanicas_validas.groupby(["tienda_id", "generico_id"]).ngroups

    print(f"\n  Combos totales            : {n_combos_total:,}")
    print(f"  Combos con mecánica válida: {n_combos_validos:,}  "
          f"({100*n_combos_validos/n_combos_total:.1f} %)")
    print(f"  Mecánicas distintas       : {mecanicas_validas['nombre_promo'].nunique()}")
    print(f"  Combinaciones combo×mec.  : {len(mecanicas_validas):,}")

    # Advertencia de desbalanceo por tipo_mecanica (BBP §13.tipo_mecanica)
    if "tipo_mecanica" in df.columns:
        dist = (
            df[df["tiene_promo_flag"] == 1]
            .groupby("tipo_mecanica")
            .size()
            .sort_values(ascending=False)
        )
        total_promo = dist.sum()
        print(f"\n  Distribución tipo_mecanica en datos de entrenamiento:")
        for mec, cnt in dist.items():
            pct = 100 * cnt / total_promo if total_promo > 0 else 0
            flag = "⚠️ " if pct < 5 else "   "
            print(f"  {flag} {str(mec):<18s} {cnt:>8,}  ({pct:.1f}%)")
        low_coverage = [str(m) for m, c in dist.items()
                        if (100 * c / total_promo if total_promo > 0 else 0) < 5]
        if low_coverage:
            print(f"\n  ⚠️  Mecánicas con < 5% de registros: {low_coverage}")
            print(f"     Las curvas de elasticidad para estas mecánicas son")
            print(f"     orientativas — verificar n_combos en elasticity_summary.csv")

    # ── Última observación por combo ──────────────────────────────────────────
    latest = (
        df.sort_values("fecha_venta")
        .groupby(["tienda_id", "generico_id"])
        .last()
        .reset_index()
    )

    hist_max = (
        df[df["tiene_promo_flag"] == 1]
        .groupby(["tienda_id", "generico_id"])["pct_descuento_calc"]
        .max().fillna(0).reset_index(name="max_disc_historico")
    )
    latest = latest.merge(hist_max, on=["tienda_id", "generico_id"], how="left")
    latest["max_disc_historico"] = latest["max_disc_historico"].fillna(0)

    mecanicas_validas = mecanicas_validas.merge(
        latest[["tienda_id", "generico_id", "max_disc_historico"]],
        on=["tienda_id", "generico_id"], how="left",
    )

    # ── Construir batch de simulaciones ──────────────────────────────────────
    print()
    sim_df = _build_sim_batch(latest, mecanicas_validas, feat_cols, discount_levels)

    X_sim = sim_df[feat_cols].copy()
    for c in cat_cols:
        if c in X_sim.columns:
            X_sim[c] = X_sim[c].astype("category")

    # ── Batch predict ─────────────────────────────────────────────────────────
    print(f"\n  Predicting {len(X_sim):,} filas en batch...")
    preds = model.predict(X_sim).clip(min=0)

    elasticity_df = pd.DataFrame({
        "tienda_id":       sim_df["_tienda_id"].values,
        "generico_id":     sim_df["_generico_id"].values,
        "nombre_promo":    sim_df["_nombre_promo"].values,
        "discount_pct":    sim_df["_discount_pct"].values,
        "predicted_units": preds.round(4),
        "is_extrapolated": sim_df["_is_extrapolated"].values,
    })

    # Agregar division_desc para heatmap
    div_map = (
        latest[["tienda_id", "generico_id", "division_desc"]]
        if "division_desc" in latest.columns else None
    )
    if div_map is not None:
        elasticity_df = elasticity_df.merge(
            div_map, on=["tienda_id", "generico_id"], how="left"
        )

    print(f"  Filas generadas: {len(elasticity_df):,}  "
          f"({elasticity_df['is_extrapolated'].sum():,} extrapoladas "
          f"[{100*elasticity_df['is_extrapolated'].mean():.1f} %])")

    # ── Monotonicidad — verificar y corregir con isotonic regression ──────────
    print("\n  Verificando monotonicidad (antes de corrección)...")
    mono_raw = (
        elasticity_df
        .groupby(["tienda_id", "generico_id", "nombre_promo"])
        .apply(_check_monotonicity)
    )
    pct_mono_raw = mono_raw.mean()
    print(f"  Monótonas antes : {pct_mono_raw:.1%}  ({int((~mono_raw).sum())} no monótonas)")

    print("  Aplicando isotonic regression (post-proceso MVP)...")
    elasticity_df = _enforce_monotonicity(elasticity_df)
    n_adjusted = int(elasticity_df["is_monotone_adjusted"].sum())
    print(f"  Filas ajustadas : {n_adjusted:,}  ({100*n_adjusted/len(elasticity_df):.1f} %)")

    mono = (
        elasticity_df
        .groupby(["tienda_id", "generico_id", "nombre_promo"])
        .apply(_check_monotonicity)
    )
    pct_mono = mono.mean()
    n_non_mono = int((~mono).sum())
    status = "✅" if pct_mono >= ELAST_MONO_MIN else "⚠️ "
    print(f"  {status} Monótonas después: {pct_mono:.1%}  ({n_non_mono} no monótonas)")

    # ── Resumen uplift por descuento ──────────────────────────────────────────
    print(f"\n  Uplift medio por nivel de descuento:")
    print(f"  {'Desc %':>7s}  {'Unidades (media)':>18s}  {'vs 0%':>8s}")
    print("  " + "-" * 38)
    base_units = elasticity_df[elasticity_df["discount_pct"] == 0]["predicted_units"].mean()
    for disc, grp in elasticity_df.groupby("discount_pct"):
        mean_u = grp["predicted_units"].mean()
        uplift = (mean_u / base_units - 1) * 100 if base_units > 0 else 0
        print(f"  {disc:>6d}%  {mean_u:>18.3f}  {uplift:>+7.1f}%")

    # ── Guardar datos ─────────────────────────────────────────────────────────
    parquet_path = REPORTS / "promo_elasticity_curves.parquet"
    csv_path     = REPORTS / "elasticity_summary.csv"

    elasticity_df.to_parquet(parquet_path, index=False)

    summary = (
        elasticity_df
        .groupby(["nombre_promo", "discount_pct"])
        .agg(
            n_combos=("tienda_id", "nunique"),
            predicted_units_mean=("predicted_units", "mean"),
            predicted_units_median=("predicted_units", "median"),
            pct_extrapolated=("is_extrapolated", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(csv_path, index=False, encoding="utf-8")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n  Generando gráficas...")

    out1 = FIGURES / "elasticity_global.png"
    _plot_global_curve(summary, out1)
    print(f"  {out1}")

    out2 = FIGURES / "elasticity_by_mecanica.png"
    _plot_by_mecanica(summary, args.top_mecanicas, out2)
    print(f"  {out2}")

    if "division_desc" in elasticity_df.columns:
        out3 = FIGURES / "elasticity_heatmap_division.png"
        _plot_heatmap(elasticity_df, out3)
        print(f"  {out3}")

    # ── Resumen final ─────────────────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print(f"  {parquet_path}")
    print(f"  {csv_path}")
    print(f"  Monotonicidad: {pct_mono:.1%}  (objetivo ≥ 90 %)")

    print(f"\n{sep}")
    print("  Listo. Próximo paso: pipeline de scoring (src/score.py) — Semana 10")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
