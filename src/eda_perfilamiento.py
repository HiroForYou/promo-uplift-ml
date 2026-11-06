# %% [markdown]
# ### promo-uplift-ml · Precisión Promocional
# src/eda_perfilamiento.py — EDA de perfilamiento de datos
#
# Semana 2: Análisis exploratorio de datos para M1 Demand Response
# Granularidad: tienda × genérico × día
#
# TAREAS:
#     2.1 Volumen y cobertura
#     2.2 Análisis de baseline insuficiente
#     2.3 Distribución del target y outliers
#     2.4 Cobertura de precios
#     2.5 Distribución de mecánicas históricas
#
# USO:
#     python src/eda_perfilamiento.py
#     python src/eda_perfilamiento.py --data data/otro_archivo.csv
#     python src/eda_perfilamiento.py --data data/train-set-depured.csv

# %% [markdown]
# #### Configuración e imports

# %%
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import seaborn as sns

try:
    get_ipython()
    _IN_NOTEBOOK = True
except NameError:
    _IN_NOTEBOOK = False
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # En notebook: buscar el marcador config.py subiendo desde el CWD
    _cwd = Path().resolve()
    ROOT = _cwd if (_cwd / "config.py").exists() else _cwd.parent

TRAIN_DATA_PATH = ROOT / "data" / "train-set-features.csv"
REPORTS_DIR     = ROOT / "reports" / "eda"
FIGS_DIR        = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
PALETTE = sns.color_palette("muted")

# %% [markdown]
# #### Helpers y carga

# %%
_report_lines: list[str] = []

def _log(msg: str = "") -> None:
    print(msg)
    _report_lines.append(msg)

def _section(title: str) -> None:
    sep = "=" * 70
    _log(f"\n{sep}")
    _log(f"  {title}")
    _log(sep)

def _save_fig(name: str) -> None:
    path = FIGS_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    if _IN_NOTEBOOK:
        from IPython.display import display, Image
        display(Image(filename=str(path)))
    _log(f"  [fig] {path.relative_to(ROOT)}")

def _max_consecutive_zeros(series: pd.Series) -> int:
    max_streak = current = 0
    for v in series:
        if v == 0:
            current += 1
            if current > max_streak:
                max_streak = current
        else:
            current = 0
    return max_streak


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"tienda_id": str, "generico_id": str},
        low_memory=False,
    )
    df["fecha_venta"] = pd.to_datetime(df["fecha_venta"].astype(str), format="%Y%m%d")
    num_cols = [
        "unidades_vendidas", "monto_venta", "costo_linea", "uom_base_qty",
        "tiene_promo_flag", "descuento_total",
        "precio_venta", "precio_venta_bruto", "precio_compra",
        "stock_disponible", "en_calendario_promo", "n_promos_activas_tienda",
        "tienda_promo_sku_sin_descuento",
        "n_tallas_vendidas", "n_colores_vendidos",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# %% [markdown]
# #### Tarea 2.1 — Volumen y cobertura

# %%
def tarea_2_1(df: pd.DataFrame) -> None:
    _section("TAREA 2.1 — Volumen y cobertura")

    n_filas       = len(df)
    n_tiendas     = df["tienda_id"].nunique()
    n_genericos   = df["generico_id"].nunique()
    n_combos      = df.groupby(["tienda_id", "generico_id"]).ngroups
    rango_fechas  = f"{df['fecha_venta'].min().date()} → {df['fecha_venta'].max().date()}"
    dias_totales  = (df["fecha_venta"].max() - df["fecha_venta"].min()).days + 1

    _log(f"  Filas totales          : {n_filas:,}")
    _log(f"  Tiendas únicas         : {n_tiendas:,}")
    _log(f"  Genéricos únicos       : {n_genericos:,}")
    _log(f"  Combinaciones t×g      : {n_combos:,}")
    _log(f"  Rango de fechas        : {rango_fechas}  ({dias_totales} días calendario)")

    dias_historial = (
        df.groupby(["tienda_id", "generico_id"])["fecha_venta"]
        .nunique()
        .rename("dias_historial")
        .reset_index()
    )
    con_30d = (dias_historial["dias_historial"] >= 30).sum()
    con_90d = (dias_historial["dias_historial"] >= 90).sum()

    _log(f"\n  Historial por combinación tienda × genérico:")
    _log(f"    Con >= 30 días : {con_30d:,}  ({100*con_30d/n_combos:.1f}%)")
    _log(f"    Con >= 90 días : {con_90d:,}  ({100*con_90d/n_combos:.1f}%)")
    _log(f"    Mediana días   : {dias_historial['dias_historial'].median():.0f}")
    _log(f"    P90 días       : {dias_historial['dias_historial'].quantile(0.90):.0f}")

    if con_30d / n_combos < 0.5:
        _log("  ⚠️  ALERTA: Menos del 50 % de combinaciones tienen ≥ 30 días.")
        if n_filas < 5_000:
            _log("     Con muestra pequeña esto es esperado; validar sobre el dataset completo.")
        else:
            _log("     Productos con historial insuficiente predominan — revisar filtro de entrenamiento.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(dias_historial["dias_historial"], bins=30, color=PALETTE[0], edgecolor="white")
    axes[0].axvline(30, color="tomato", linestyle="--", label="30 d (mín lags)")
    axes[0].axvline(90, color="goldenrod", linestyle="--", label="90 d")
    axes[0].set_title("Días de historial por combinación tienda × genérico")
    axes[0].set_xlabel("Días de historial")
    axes[0].set_ylabel("# combinaciones")
    axes[0].legend()

    cobertura_diaria = df.groupby("fecha_venta")["generico_id"].nunique().reset_index()
    cobertura_diaria.columns = ["fecha_venta", "n_genericos"]
    axes[1].plot(cobertura_diaria["fecha_venta"], cobertura_diaria["n_genericos"],
                 color=PALETTE[1], linewidth=1.2)
    axes[1].set_title("Genéricos con venta por día")
    axes[1].set_xlabel("Fecha")
    axes[1].set_ylabel("# genéricos únicos")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    _save_fig("2_1_cobertura")

# %% [markdown]
# #### Tarea 2.2 — Baseline insuficiente

# %%
def tarea_2_2(df: pd.DataFrame) -> None:
    _section("TAREA 2.2 — Baseline insuficiente (28 días consecutivos sin promo)")

    grupos = df.sort_values("fecha_venta").groupby(["tienda_id", "generico_id"])
    max_streak_sin_promo = grupos["tiene_promo_flag"].apply(_max_consecutive_zeros)

    con_baseline = (max_streak_sin_promo >= 28).sum()
    total_combos = len(max_streak_sin_promo)
    pct_baseline = 100 * con_baseline / total_combos

    _log(f"  Combinaciones analizadas  : {total_combos:,}")
    _log(f"  Con baseline válido (≥28d): {con_baseline:,}  ({pct_baseline:.1f}%)")
    _log(f"  Sin baseline              : {total_combos - con_baseline:,}  ({100-pct_baseline:.1f}%)")
    _log(f"\n  Distribución de racha máxima sin promo:")
    for p in [25, 50, 75, 90]:
        _log(f"    P{p:2d}: {max_streak_sin_promo.quantile(p/100):.0f} días")

    if pct_baseline < 40:
        _log(f"\n  ⚠️  Solo el {pct_baseline:.1f}% tiene baseline válido.")
        _log("     Revisar si la muestra tiene sesgo hacia periodos con alta actividad promo.")
        _log("     Si el dato completo confirma esto, considerar reducir el umbral a 14 días.")

    fig, ax = plt.subplots(figsize=(9, 4))
    bins = list(range(0, int(max_streak_sin_promo.max()) + 5, 5))
    ax.hist(max_streak_sin_promo, bins=bins, color=PALETTE[2], edgecolor="white")
    ax.axvline(28, color="tomato", linestyle="--", linewidth=1.5, label="Umbral 28d")
    ax.set_title("Racha máxima de días sin promo por combinación tienda × genérico")
    ax.set_xlabel("Días consecutivos sin promo (máximo histórico)")
    ax.set_ylabel("# combinaciones")
    ax.legend()
    plt.tight_layout()
    _save_fig("2_2_baseline")

# %% [markdown]
# #### Tarea 2.3 — Distribución del target y outliers

# %%
def tarea_2_3(df: pd.DataFrame) -> None:
    _section("TAREA 2.3 — Distribución del target (unidades_vendidas) y outliers")

    uv = df["unidades_vendidas"].dropna()

    _log("  Estadísticas de unidades_vendidas:")
    desc = uv.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    for stat, val in desc.items():
        _log(f"    {stat:6s}: {val:>10.2f}")

    pct_cero  = (uv == 0).mean()
    p99       = uv.quantile(0.99)
    p995      = uv.quantile(0.995)
    n_outlier = (uv > p99).sum()

    _log(f"\n  Días con 0 unidades vendidas : {pct_cero:.1%}")
    _log(f"  P99  unidades                : {p99:.1f}")
    _log(f"  P99.5 unidades               : {p995:.1f}")
    _log(f"  Filas outlier (> P99)        : {n_outlier}  ({100*n_outlier/len(uv):.2f}%)")

    if pct_cero > 0.30:
        _log(f"\n  ⚠️  Alta sparsidad: {pct_cero:.1%} de filas con 0 unidades.")
        _log("     Considerar Tweedie/Poisson o separar modelo de 'vende vs no-vende'.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cap = uv.clip(upper=p995)
    axes[0].hist(cap, bins=50, color=PALETTE[0], edgecolor="white")
    axes[0].axvline(p99, color="tomato", linestyle="--", label=f"P99 = {p99:.0f}")
    axes[0].set_title("Distribución de unidades_vendidas (cap P99.5)")
    axes[0].set_xlabel("Unidades")
    axes[0].set_ylabel("Frecuencia")
    axes[0].legend()

    ts = df.groupby("fecha_venta")["unidades_vendidas"].sum().reset_index()
    axes[1].plot(ts["fecha_venta"], ts["unidades_vendidas"], color=PALETTE[1], linewidth=1)
    axes[1].set_title("Unidades vendidas totales por día")
    axes[1].set_xlabel("Fecha")
    axes[1].set_ylabel("Unidades (suma diaria)")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    _save_fig("2_3_target")

    if "division_desc" in df.columns:
        _log("\n  Mediana de unidades por division_desc:")
        seg = (
            df.groupby("division_desc")["unidades_vendidas"]
            .agg(["median", "count"])
            .sort_values("median", ascending=False)
        )
        for div, row in seg.iterrows():
            _log(f"    {div:<35s}  mediana={row['median']:>6.1f}  n={row['count']:>5.0f}")

# %% [markdown]
# #### Tarea 2.4 — Cobertura de precios

# %%
def tarea_2_4(df: pd.DataFrame) -> None:
    _section("TAREA 2.4 — Cobertura de precios")

    total = len(df)
    for col, label in [
        ("precio_efectivo",    "precio_efectivo (transacc.)"),
        ("costo_unitario",     "costo_unitario  (transacc.)"),
        ("precio_venta",       "precio_venta (catálogo a155+ecc)"),
        ("precio_venta_bruto", "precio_venta_bruto (catálogo)"),
        ("precio_compra",      "precio_compra   (catálogo a018+ecc)"),
    ]:
        if col not in df.columns:
            continue
        n_con = df[col].notna().sum()
        pct   = 100 * n_con / total
        flag  = "⚠️ " if pct < 80 else "✅ "
        _log(f"  {flag}{label}: {n_con:,}/{total:,}  ({pct:.1f}%)")

    _log("")
    sin_precio = df[["precio_venta", "precio_compra"]].isna().all(axis=1).sum()
    _log(f"  Filas sin precio_venta NI precio_compra: {sin_precio:,}  ({100*sin_precio/total:.1f}%)")
    if sin_precio / total > 0.20:
        _log("  ⚠️  Más del 20 % sin ningún precio — revisar join con a155+ecc / a018+ecc.")

    if "division_desc" not in df.columns:
        return

    price_axes_cfg = [
        ("precio_venta",  "precio_venta (a155+ecc)",  PALETTE[0]),
        ("precio_compra", "precio_compra (a018+ecc)", PALETTE[2]),
    ]
    price_axes_cfg = [(col, lbl, clr) for col, lbl, clr in price_axes_cfg if col in df.columns]
    if not price_axes_cfg:
        return

    sort_col = price_axes_cfg[0][0]
    division_order = (
        df.groupby("division_desc")[sort_col]
        .apply(lambda s: 100 * s.notna().sum() / len(s))
        .sort_values()
        .index.tolist()
    )
    n_div = len(division_order)
    fig_h = max(3, n_div * 0.4 + 1)

    fig, axes = plt.subplots(1, len(price_axes_cfg),
                              figsize=(9 * len(price_axes_cfg), fig_h), sharey=True)
    if len(price_axes_cfg) == 1:
        axes = [axes]

    for ax, (col, lbl, clr) in zip(axes, price_axes_cfg):
        cob = (
            df.groupby("division_desc")[col]
            .apply(lambda s: 100 * s.notna().sum() / len(s))
            .reindex(division_order)
            .reset_index()
        )
        cob.columns = ["division_desc", "pct"]
        _log(f"\n  Cobertura {col} por division_desc:")
        for _, row in cob.sort_values("pct").iterrows():
            flag = "⚠️ " if row["pct"] < 80 else "   "
            _log(f"  {flag}  {row['division_desc']:<35s}: {row['pct']:>5.1f}%")
        bar_colors = ["tomato" if v < 80 else clr for v in cob["pct"]]
        ax.barh(cob["division_desc"], cob["pct"], color=bar_colors)
        ax.axvline(80, color="tomato", linestyle="--", linewidth=1.2, label="Umbral 80%")
        ax.set_xlim(0, 105)
        ax.set_title(f"Cobertura de {col} por división")
        ax.set_xlabel(f"% filas con {col}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save_fig("2_4_cobertura_precios")

# %% [markdown]
# #### Tarea 2.5 — Mecánicas históricas

# %%
def tarea_2_5(df: pd.DataFrame) -> None:
    _section("TAREA 2.5 — Mecánicas históricas (promos)")

    total = len(df)
    _log(f"  Filas con tiene_promo_flag = 1 : {df['tiene_promo_flag'].sum():,.0f}  ({100*df['tiene_promo_flag'].mean():.1f}%)")
    _log(f"  Filas con en_calendario_promo  : {df['en_calendario_promo'].sum():,.0f}  ({100*df['en_calendario_promo'].mean():.1f}%)")

    n_tienda_sin_desc = df["tienda_promo_sku_sin_descuento"].sum()
    _log(f"  Tienda en promo pero SKU sin descuento: {n_tienda_sin_desc:,.0f}  ({100*n_tienda_sin_desc/total:.1f}%)")

    promo_df = df[df["tiene_promo_flag"] == 1].copy()
    promo_df["pct_descuento"] = promo_df["descuento_total"] / (
        promo_df["monto_venta"] + promo_df["descuento_total"]
    ).replace(0, np.nan)

    if "nombre_promo" in promo_df.columns:
        promo_df["nombre_promo"] = promo_df["nombre_promo"].str.strip().fillna("(sin nombre)")

        mecanicas = (
            promo_df.groupby("nombre_promo")
            .agg(
                combinaciones=("generico_id", lambda s: (
                    promo_df.loc[s.index].groupby(["tienda_id", "generico_id"]).ngroups
                )),
                dias_en_promo=("fecha_venta", "count"),
                descuento_promedio=("pct_descuento", "mean"),
                monto_total_promo=("monto_venta", "sum"),
            )
            .sort_values("dias_en_promo", ascending=False)
            .reset_index()
        )

        _log(f"\n  Top mecánicas por días en promo (tiene_promo_flag = 1):\n")
        header = f"  {'Mecánica':<45s} {'Combos':>7s} {'Días':>7s} {'Dto avg':>8s}"
        _log(header)
        _log("  " + "-" * (len(header) - 2))
        for _, row in mecanicas.head(20).iterrows():
            dto = f"{row['descuento_promedio']:.1%}" if pd.notna(row["descuento_promedio"]) else "  N/A "
            _log(
                f"  {row['nombre_promo'][:45]:<45s} "
                f"{row['combinaciones']:>7.0f} "
                f"{row['dias_en_promo']:>7.0f} "
                f"{dto:>8s}"
            )

        insuf = mecanicas[mecanicas["combinaciones"] < 3]
        _log(f"\n  Mecánicas con < 3 eventos/combinación (NO aptas para curvas): {len(insuf)}")
        if len(insuf) > 0:
            for _, row in insuf.iterrows():
                _log(f"    - {row['nombre_promo']}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    tipo_counts = df["tipo_descuento"].value_counts()
    axes[0].bar(tipo_counts.index, tipo_counts.values,
                color=PALETTE[:len(tipo_counts)], edgecolor="white")
    axes[0].set_title("Distribución tipo de descuento")
    axes[0].set_xlabel("Tipo de descuento")
    axes[0].set_ylabel("# filas")
    axes[0].tick_params(axis="x", rotation=20)

    pct_desc = promo_df["pct_descuento"].dropna()
    if len(pct_desc) > 0:
        axes[1].hist(pct_desc * 100, bins=30, color=PALETTE[1], edgecolor="white")
        axes[1].set_title("Distribución del % de descuento (filas con promo)")
        axes[1].set_xlabel("% descuento")
        axes[1].set_ylabel("Frecuencia")

    plt.tight_layout()
    _save_fig("2_5_mecanicas")

    if "n_promos_activas_tienda" in df.columns:
        _log(f"\n  n_promos_activas_tienda — distribución:")
        vc = df["n_promos_activas_tienda"].value_counts().sort_index()
        for val, cnt in vc.items():
            _log(f"    {int(val):>3d} promos activas: {cnt:>6,} filas  ({100*cnt/total:.1f}%)")

# %% [markdown]
# #### Resumen ejecutivo

# %%
def resumen_ejecutivo(df: pd.DataFrame) -> None:
    _section("RESUMEN EJECUTIVO — Criterios de exclusión recomendados")

    total    = len(df)
    n_combos = df.groupby(["tienda_id", "generico_id"]).ngroups

    dias_hist = (
        df.groupby(["tienda_id", "generico_id"])["fecha_venta"].nunique()
    )
    grupos    = df.sort_values("fecha_venta").groupby(["tienda_id", "generico_id"])
    max_streak = grupos["tiene_promo_flag"].apply(_max_consecutive_zeros)

    excl_historial  = (dias_hist < 30).sum()
    excl_baseline   = (max_streak < 28).sum()
    excl_sin_precio = df[["precio_venta", "precio_compra"]].isna().all(axis=1).sum()

    uv   = df["unidades_vendidas"].dropna()
    p995 = uv.quantile(0.995)
    excl_outliers = (uv > p995).sum()

    _log(f"  Total filas en dataset     : {total:,}")
    _log(f"  Total combinaciones t×g    : {n_combos:,}")
    _log("")
    _log(f"  [CRITERIO 1] Combos con < 30 días historial  → excluir del entrenamiento: {excl_historial:,}  ({100*excl_historial/n_combos:.1f}%)")
    _log(f"  [CRITERIO 2] Combos sin baseline 28d         → flag_baseline_insuficiente: {excl_baseline:,}  ({100*excl_baseline/n_combos:.1f}%)")
    _log(f"  [CRITERIO 3] Filas sin ningún precio catálogo→ revisar join a155/a018: {excl_sin_precio:,}  ({100*excl_sin_precio/total:.1f}%)")
    _log(f"  [CRITERIO 4] Filas outlier unidades (>P99.5) → winsorizar a {p995:.0f}: {excl_outliers:,}  ({100*excl_outliers/total:.2f}%)")
    _log("")
    if total < 5_000:
        _log("  NOTA: Análisis sobre muestra pequeña — confirmar proporciones sobre")
        _log("  el dataset completo (train-set-features.csv) antes de fijar umbrales.")
    else:
        _log("  NOTA: Análisis sobre dataset de entrenamiento (flag_baseline_insuficiente=0).")
        _log("  Los criterios 1 y 2 ya fueron aplicados por el filtro SQL.")

# %% [markdown]
# #### Main

# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="EDA de perfilamiento M1 — Semana 2")
    parser.add_argument("--data", type=Path, default=TRAIN_DATA_PATH,
                        help="Ruta al CSV de entrada (default: data/train-set-features.csv)")
    args, _ = parser.parse_known_args()

    _log(f"Cargando datos desde: {args.data}")
    df = load_data(args.data)
    _log(f"Dataset cargado: {len(df):,} filas × {len(df.columns)} columnas\n")

    tarea_2_1(df)
    tarea_2_2(df)
    tarea_2_3(df)
    tarea_2_4(df)
    tarea_2_5(df)
    resumen_ejecutivo(df)

    report_path = REPORTS_DIR / "eda_summary.txt"
    report_path.write_text("\n".join(_report_lines), encoding="utf-8")
    print(f"\nReporte guardado en: {report_path.relative_to(ROOT)}")
    print(f"Figuras guardadas en: {FIGS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
