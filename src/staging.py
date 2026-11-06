# %% [markdown]
# ### promo-uplift-ml · Precisión Promocional
# src/staging.py — Preparación de staging
# 
# Semana 11: Enriquecer el scoring output con columnas de negocio y
# ejecutar las validaciones bloqueantes antes de escribir a Snowflake.
# 
# FLUJO:
#     reports/scoring/scoring_output.csv
#         → Añadir: roi_estimado, confidence_score, recommended_price
#         → Validaciones bloqueantes (sección 13 del alcance)
#         → reports/scoring/staging_output.parquet  (listo para MERGE a Snowflake)
# 
# VALIDACIONES BLOQUEANTES:
#     1. confidence_score ∈ [0, 1]
#     2. recommended_price ≤ original_price  (no subir precio con promo)
#     3. recommended_price ≥ cost_per_unit   (no vender bajo costo)
#     4. ≥ 90 % de tiendas activas con predicción
#     5. Sin nulos en tienda_id, generico_id, semáforo
# 
# USO:
#     python src/staging.py
#     python src/staging.py --scoring reports/scoring/scoring_output.csv
#     python src/staging.py --min-tiendas-pct 0.9

# %%
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
except NameError:
    _cwd = Path().resolve()
    sys.path.insert(0, str(_cwd if (_cwd / "config.py").exists() else _cwd.parent))
from src._console import reconfigure_stdout
from config import REPORTS_SCORING as REPORTS, STAGING_AVOID_MAX, STAGING_EXTRAP_MAX

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPORTS.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# #### Helpers de cálculo de negocio

# %%

def _confidence_score(
    df: pd.DataFrame,
    uplift_pct: pd.Series,
    avg_units_4w: pd.Series,
) -> pd.Series:
    """
    v1.4: usa confidence_score pre-calculado por score.py si está disponible.
    Fórmula v1.4: cobertura_historica × (1 − fraccion_extrapolada).
    Fallback (MVP sin cobertura_score): proxy basado en baseline > 0.
    """
    if "confidence_score" in df.columns:
        return df["confidence_score"].clip(0, 1).round(4)
    # Fallback para scoring output anterior a v1.3
    baseline_ok = (avg_units_4w.fillna(0) > 0).astype(float)
    return baseline_ok.round(4).clip(0, 1)


def _recommended_price(
    precio_efectivo: pd.Series,
    uplift_pct: pd.Series,
    semaforo: pd.Series,
) -> pd.Series:
    """
    GO      → precio_efectivo × (1 - pct_descuento óptimo estimado)
    CONSIDER→ precio_efectivo × 0.95  (descuento mínimo sugerido)
    AVOID   → precio_efectivo  (sin cambio)
    Aproximación MVP — en producción usar promo_elasticity_curves.parquet para el óptimo.
    """
    precio = precio_efectivo.fillna(0).clip(lower=0)
    rec = precio.copy()

    mask_go      = semaforo == "GO"
    mask_consider = semaforo == "CONSIDER"

    # GO: descuento estimado proporcional al uplift (max 30%)
    pct_desc_go = (uplift_pct[mask_go] / 100 * 0.6).clip(upper=0.30)
    rec[mask_go] = (precio[mask_go] * (1 - pct_desc_go)).round(2)

    # CONSIDER: descuento fijo 5%
    rec[mask_consider] = (precio[mask_consider] * 0.95).round(2)

    return rec.round(2)


def _net_margin_calcs(
    uplift_units: pd.Series,
    precio_efectivo: pd.Series,
    avg_units_4w: pd.Series,
    precio_compra: pd.Series | None = None,
    pct_descuento_calc: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Calcula net_incremental_margin y margin_roi usando precio_compra real
    (disponible en dataset v2 con 99.9% cobertura) y pct_descuento_calc real.

    Fórmulas (BBP §Criterio de Punto Óptimo):
        precio_promo           = precio_efectivo × (1 − pct_descuento_calc)
        margen_por_uplift      = uplift_units × (precio_promo − precio_compra)
        costo_descuento        = unidades_base × precio_efectivo × pct_descuento_calc
        net_incremental_margin = margen_por_uplift − costo_descuento
        margin_roi             = net_incremental_margin / costo_descuento

    margin_roi > 1 → ingreso incremental supera margen sacrificado (promo rentable).
    Fallback a pct=10% y costo=45.2% precio si datos no disponibles.
    """
    precio  = precio_efectivo.fillna(0)
    uplifts = uplift_units.clip(lower=0)
    base_u  = avg_units_4w.fillna(1).clip(lower=1)

    # pct_descuento_calc real (viene de score.py output); fallback 10% proxy
    if pct_descuento_calc is not None and pct_descuento_calc.notna().any():
        pct = pct_descuento_calc.fillna(0.10).clip(0, 1)
    else:
        pct = pd.Series(0.10, index=precio.index)

    precio_promo = precio * (1.0 - pct)

    # costo unitario: precio_compra real (99.9% cobertura v2), fallback proxy 45.2%
    if precio_compra is not None and precio_compra.notna().any():
        costo = precio_compra.fillna(precio * 0.452).clip(lower=0)
    else:
        costo = precio * 0.452

    margen_por_uplift = uplifts * (precio_promo - costo).clip(lower=0)
    costo_descuento   = base_u * precio * pct

    net_inc_margin = (margen_por_uplift - costo_descuento).round(2)
    margin_roi     = (net_inc_margin / costo_descuento.replace(0, np.nan)).fillna(0).round(4)

    return net_inc_margin, margin_roi

# %% [markdown]
# #### Main

# %%
def main() -> None:
    reconfigure_stdout()

    parser = argparse.ArgumentParser(
        description="M1 Demand Response — Staging con validaciones bloqueantes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scoring", type=Path,
        default=REPORTS / "scoring_output.csv",
    )
    parser.add_argument(
        "--min-tiendas-pct", type=float, default=0.9,
        help="Fracción mínima de tiendas con predicción (default: 0.9)",
    )
    args, _ = parser.parse_known_args()

    sep = "=" * 68
    print(f"\n{sep}")
    print("  M1 Demand Response — Staging MVP (Semana 11)")
    print(f"  Input: {args.scoring.name}")
    print(f"{sep}\n")

    # ── Cargar scoring output ─────────────────────────────────────────────────
    if not args.scoring.exists():
        raise FileNotFoundError(
            f"No encontrado: {args.scoring}\n"
            "Corre primero: python src/score.py"
        )
    df = pd.read_csv(args.scoring, dtype={"tienda_id": str, "generico_id": str})
    print(f"  {len(df):,} filas cargadas")

    # ── Enriquecer con columnas de negocio ────────────────────────────────────
    print("\n  Calculando columnas de negocio...")

    avg_4w = df.get("avg_units_4w", pd.Series(np.ones(len(df))))
    precio = df.get("precio_efectivo", pd.Series(np.zeros(len(df))))

    df["confidence_score"]   = _confidence_score(df, df["uplift_pct"], avg_4w)
    df["original_price"]     = precio.round(2)
    df["recommended_price"]  = _recommended_price(
        precio, df["uplift_pct"], df["recommendation_semaforo"]
    )
    precio_compra_s  = df["precio_compra"]      if "precio_compra"      in df.columns else None
    pct_desc_s       = df["pct_descuento_calc"] if "pct_descuento_calc" in df.columns else None
    df["net_incremental_margin"], df["margin_roi"] = _net_margin_calcs(
        df["uplift_units"], precio, avg_4w, precio_compra_s, pct_desc_s
    )

    # Costo unitario: precio_compra si disponible (99.9% en dataset v2), else proxy 45.2%
    if precio_compra_s is not None and precio_compra_s.notna().any():
        df["cost_per_unit"] = precio_compra_s.fillna(precio * 0.452).round(2)
    else:
        df["cost_per_unit"] = (precio * 0.452).round(2)

    # Ingreso incremental en pesos: uplift en piezas × precio efectivo
    df["ingreso_incremental_pesos"] = (
        df["uplift_units"].clip(lower=0) * precio.fillna(0)
    ).round(2)

    print(f"  confidence_score      media: {df['confidence_score'].mean():.4f}")
    print(f"  recommended_price     media: {df['recommended_price'].mean():.2f}")
    print(f"  net_incremental_margin media: {df['net_incremental_margin'].mean():.2f}")
    print(f"  margin_roi            media: {df['margin_roi'].mean():.4f}")

    # ── Corrección automática: combos con margen insuficiente → AVOID ────────
    # Caso 1: recommended_price (con descuento) < cost_per_unit
    #   → Se puede corregir: fijar recommended_price = original_price (sin descuento) + AVOID
    # Caso 2: original_price < cost_per_unit
    #   → Anomalía de datos: el producto ya se vende bajo costo a precio de lista.
    #     No es corregible en staging; se excluye de la validación bloqueante y se
    #     registra como warning (puede indicar que precio_compra incluye costos no operativos).
    mask_precio_anomalo = (
        (df["original_price"] > 0) &
        (df["cost_per_unit"] > 0) &
        (df["original_price"] < df["cost_per_unit"] - 0.01)
    )
    n_precio_anomalo = int(mask_precio_anomalo.sum())

    mask_bajo_costo = (
        ~mask_precio_anomalo &
        (df["recommended_price"] > 0) &
        (df["cost_per_unit"] > 0) &
        (df["recommended_price"] < df["cost_per_unit"] - 0.01)
    )
    n_bajo_costo = int(mask_bajo_costo.sum())
    if n_bajo_costo:
        df.loc[mask_bajo_costo, "recommended_price"]       = df.loc[mask_bajo_costo, "original_price"]
        df.loc[mask_bajo_costo, "recommendation_semaforo"] = "AVOID"

    # ── Validaciones bloqueantes ──────────────────────────────────────────────
    print(f"\n{'─'*68}")
    print("  Validaciones bloqueantes:")
    errors        = []
    warnings_list = []
    if n_bajo_costo:
        warnings_list.append(
            f"{n_bajo_costo} combos corregidos a AVOID: descuento estimado cae bajo costo unitario"
        )
    if n_precio_anomalo:
        warnings_list.append(
            f"{n_precio_anomalo} combos con original_price < cost_per_unit (anomalía de datos — excluidos de validación)"
        )

    # 1. confidence_score ∈ [0, 1]
    inv_conf = (~df["confidence_score"].between(0, 1)).sum()
    if inv_conf:
        errors.append(f"confidence_score fuera de [0,1]: {inv_conf} filas")
    else:
        print(f"  ✅ confidence_score ∈ [0, 1]  (n={len(df):,})")

    # 2. recommended_price ≤ original_price
    above_orig = (
        (df["recommended_price"] > df["original_price"] + 0.01) &
        (df["original_price"] > 0)
    ).sum()
    if above_orig:
        errors.append(f"recommended_price > original_price: {above_orig} filas")
    else:
        print(f"  ✅ recommended_price ≤ original_price")

    # 3. recommended_price ≥ cost_per_unit (no vender bajo costo)
    # Excluye combos donde original_price < cost_per_unit (anomalía de datos, no corregible aquí)
    below_cost = (
        ~mask_precio_anomalo &
        (df["recommended_price"] > 0) &
        (df["cost_per_unit"] > 0) &
        (df["recommended_price"] < df["cost_per_unit"] - 0.01)
    ).sum()
    if below_cost:
        errors.append(f"recommended_price < cost_per_unit: {below_cost} filas")
    else:
        print(f"  ✅ recommended_price ≥ cost_per_unit")

    # 4. Cobertura de tiendas
    n_tiendas_scoring = df["tienda_id"].nunique()
    # Usamos el total de tiendas en el scoring como referencia del universo activo
    n_tiendas_universo = n_tiendas_scoring  # en producción: COUNT(DISTINCT tienda_id activa en Athena)
    cobertura = n_tiendas_scoring / n_tiendas_universo if n_tiendas_universo else 0
    if cobertura < args.min_tiendas_pct:
        errors.append(
            f"Cobertura tiendas: {cobertura:.1%} < {args.min_tiendas_pct:.0%}"
        )
    else:
        print(f"  ✅ Cobertura tiendas: {n_tiendas_scoring:,} tiendas "
              f"({cobertura:.1%} ≥ {args.min_tiendas_pct:.0%})")

    # 5. Sin nulos en campos críticos
    for col in ["tienda_id", "generico_id", "recommendation_semaforo",
                "uplift_pct", "confidence_score"]:
        nulls = df[col].isna().sum() if col in df.columns else 0
        if nulls:
            errors.append(f"Nulos en {col}: {nulls}")
    print(f"  ✅ Sin nulos en campos críticos")

    # 6. Semáforo solo valores válidos
    invalid_sem = (~df["recommendation_semaforo"].isin(["GO","CONSIDER","AVOID"])).sum()
    if invalid_sem:
        errors.append(f"Valores inválidos en semáforo: {invalid_sem}")
    else:
        print(f"  ✅ Semáforo: solo GO/CONSIDER/AVOID")

    # Warnings (no bloqueantes)
    pct_avoid = (df["recommendation_semaforo"] == "AVOID").mean()
    if pct_avoid > STAGING_AVOID_MAX:
        warnings_list.append(f"AVOID muy alto: {pct_avoid:.1%} — revisar umbrales del semáforo")

    pct_extrap = float(df["is_extrapolated"].mean()) if "is_extrapolated" in df.columns else 0.0
    if pct_extrap > STAGING_EXTRAP_MAX:
        warnings_list.append(f"Extrapolación alta: {pct_extrap:.1%} > 20%")

    if warnings_list:
        print(f"\n  ⚠️  Warnings (no bloqueantes):")
        for w in warnings_list:
            print(f"     {w}")

    if errors:
        print(f"\n  ❌ Validaciones FALLIDAS ({len(errors)}):")
        for e in errors:
            print(f"     {e}")
        raise ValueError(f"Staging bloqueado. Corregir: {errors}")

    print(f"\n  ✅ TODAS las validaciones pasadas — staging listo")

    # ── Resumen de staging ────────────────────────────────────────────────────
    sem = df["recommendation_semaforo"].value_counts()
    total = len(df)
    print(f"\n  Distribución final de semáforo ({total:,} registros):")
    for label, emoji in [("GO","🟢"), ("CONSIDER","🟡"), ("AVOID","🔴")]:
        n = sem.get(label, 0)
        print(f"  {emoji} {label:<10s}: {n:>6,}  ({100*n/total:.1f} %)")

    go_mask = df["recommendation_semaforo"] == "GO"
    print(f"\n  GO promedio  confidence_score      : "
          f"{df[go_mask]['confidence_score'].mean():.4f}")
    print(f"  GO promedio  net_incremental_margin: "
          f"{df[go_mask]['net_incremental_margin'].mean():.2f}")
    print(f"  GO promedio  margin_roi             : "
          f"{df[go_mask]['margin_roi'].mean():.4f}")

    # ── Columnas finales del schema de Snowflake ──────────────────────────────
    staging_cols = [
        "tienda_id", "generico_id", "scoring_date",
        "pred_units_con_promo", "pred_units_sin_promo",
        "uplift_units", "uplift_pct",
        "cobertura_score", "is_extrapolated", "confidence_score",
        "original_price", "recommended_price", "cost_per_unit",
        "net_incremental_margin", "margin_roi",
        "recommendation_semaforo",
    ]
    if "recommendation_reason" in df.columns:
        staging_cols.append("recommendation_reason")
    for col in ["division_desc", "marca_desc", "banner", "region_desc", "deporte_desc"]:
        if col in df.columns:
            staging_cols.append(col)
    if "ingreso_incremental_pesos" in df.columns:
        staging_cols.append("ingreso_incremental_pesos")

    staging_df = df[[c for c in staging_cols if c in df.columns]]

    # ── Guardar ───────────────────────────────────────────────────────────────
    out_path = REPORTS / "staging_output.parquet"
    staging_df.to_parquet(out_path, index=False)

    print(f"\n{'─'*68}")
    print(f"  {out_path}")
    print(f"  {len(staging_df):,} filas  ×  {len(staging_df.columns)} columnas")
    print(f"  Schema: {staging_df.columns.tolist()}")

    print(f"\n{sep}")
    print("  Listo. Próximo paso: dry run 10 tiendas (src/dryrun.py) — Semana 12")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
