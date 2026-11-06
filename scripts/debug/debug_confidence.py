"""
scripts/debug_confidence.py — Diagnóstico de confidence_score bajo

Investiga por qué confidence_score promedio es ~0.013.
Hipótesis a verificar:
  A. Granularidad: ¿df tiene múltiples filas por (combo, fecha)?
     → groupby().size() contaría transacciones, no días únicos
  B. is_extrapolated muy alto: confidence = cobertura × 0 para esos combos
  C. Ventana 90d vs ventana activa: combos "activos" (180d) pero sin ventas en 90d
  D. El denominador 90 es incorrecto: quizás días reales < 90

USO:
    python scripts/debug_confidence.py
    python scripts/debug_confidence.py --active-window-days 180
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import build_features, GROUP_COLS

ROOT      = Path(__file__).resolve().parent.parent
TRAIN_DATA_PATH = ROOT / "data" / "train-set-features.csv"
SEP = "=" * 68


def section(title: str) -> None:
    print(f"\n{'─'*68}")
    print(f"  {title}")
    print(f"{'─'*68}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=TRAIN_DATA_PATH)
    parser.add_argument("--active-window-days", type=int, default=None)
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("  Debug confidence_score — diagnóstico completo")
    print(f"{SEP}")

    # ── Cargar ────────────────────────────────────────────────────────────────
    print(f"\nCargando {args.data.name}...")
    df_raw = pd.read_csv(
        args.data,
        dtype={"tienda_id": str, "generico_id": str},
        low_memory=False,
    )
    for c in df_raw.select_dtypes(include=["object"]).columns:
        try:
            df_raw[c] = pd.to_numeric(df_raw[c])
        except (ValueError, TypeError):
            pass

    df = build_features(df_raw)
    if "division_desc" in df.columns:
        df = df[~df["division_desc"].isin({"VENTA DE BOLSAS"})]
    df = df.dropna(subset=["unidades_vendidas"])
    df["unidades_vendidas"] = df["unidades_vendidas"].clip(lower=0)

    scoring_date = df["fecha_venta"].max()
    window_90_start = scoring_date - pd.Timedelta(days=90)
    print(f"  scoring_date  : {scoring_date.date()}")
    print(f"  ventana 90d   : {window_90_start.date()} → {scoring_date.date()}")

    # scoring_df
    scoring_df = (
        df[df["fecha_venta"] <= scoring_date]
        .sort_values("fecha_venta")
        .groupby(GROUP_COLS)
        .last()
        .reset_index()
    )
    if args.active_window_days:
        cutoff = scoring_date - pd.Timedelta(days=args.active_window_days)
        scoring_df = scoring_df[scoring_df["fecha_venta"] >= cutoff]
        print(f"  active_window : {args.active_window_days}d → {len(scoring_df):,} combos")
    else:
        print(f"  combos totales: {len(scoring_df):,}")

    # ── A. GRANULARIDAD ───────────────────────────────────────────────────────
    section("A. Granularidad: ¿hay múltiples filas por (combo, fecha)?")

    df_90 = df[df["fecha_venta"].between(window_90_start, scoring_date)]
    filas_90 = len(df_90)
    combos_90 = df_90.groupby(GROUP_COLS).ngroups
    fechas_unicas_90 = df_90.groupby(GROUP_COLS + ["fecha_venta"]).ngroups

    print(f"  Filas en ventana 90d          : {filas_90:,}")
    print(f"  Combos únicos en ventana 90d  : {combos_90:,}")
    print(f"  (combo, fecha) únicos en 90d  : {fechas_unicas_90:,}")
    ratio = filas_90 / max(1, fechas_unicas_90)
    print(f"  Ratio filas / (combo,fecha)   : {ratio:.4f}")
    if abs(ratio - 1.0) < 0.01:
        print("  ✅ Granularidad OK — 1 fila por (combo, fecha)")
    else:
        print(f"  ⚠️  Hay {ratio:.2f}× filas por (combo, fecha) — groupby.size() cuenta transacciones, no días")
        print("     FIX: usar .groupby().nunique('fecha_venta') en vez de .size()")

    # ── B. IS_EXTRAPOLATED ────────────────────────────────────────────────────
    section("B. Tasa de is_extrapolated y su impacto en confidence_score")

    if "pct_descuento_calc" in df.columns and "pct_descuento_calc" in scoring_df.columns:
        max_desc = (
            df.groupby(GROUP_COLS)["pct_descuento_calc"]
            .max()
            .reset_index(name="max_desc_hist")
        )
        sdf = scoring_df.merge(max_desc, on=GROUP_COLS, how="left")
        sdf["max_desc_hist"] = sdf["max_desc_hist"].fillna(0)
        is_extrap = sdf["pct_descuento_calc"].values > sdf["max_desc_hist"].values
        n_extrap = is_extrap.sum()
        print(f"  Combos con is_extrapolated=True  : {n_extrap:,} / {len(sdf):,} ({100*n_extrap/len(sdf):.1f}%)")
        print(f"  Combos con is_extrapolated=False : {(~is_extrap).sum():,}")
        if n_extrap > len(sdf) * 0.5:
            print("  ⚠️  Más del 50% están extrapolados → confidence_score = 0 para esos combos")
            print("     Posible causa: scoring_df usa la última observación, que puede tener")
            print("     un pct_descuento_calc = 0, mientras max_desc_hist > 0 (paradoja inversa)")
        # Mostrar stats de descuento en scoring_df vs histórico
        print(f"\n  pct_descuento_calc en scoring_df:")
        print(f"    media  : {sdf['pct_descuento_calc'].mean():.4f}")
        print(f"    mediana: {sdf['pct_descuento_calc'].median():.4f}")
        print(f"    = 0    : {(sdf['pct_descuento_calc'] == 0).sum():,} combos ({100*(sdf['pct_descuento_calc']==0).mean():.1f}%)")
        print(f"\n  max_desc_hist (máximo histórico por combo):")
        print(f"    media  : {sdf['max_desc_hist'].mean():.4f}")
        print(f"    mediana: {sdf['max_desc_hist'].median():.4f}")
        print(f"    = 0    : {(sdf['max_desc_hist'] == 0).sum():,} combos ({100*(sdf['max_desc_hist']==0).mean():.1f}%)")
    else:
        print("  pct_descuento_calc no disponible")

    # ── C. DISTRIBUCIÓN TEMPORAL DE COMBOS ACTIVOS ───────────────────────────
    section("C. ¿Dónde están las últimas ventas de los combos activos?")

    if "fecha_venta" in scoring_df.columns:
        last_sale = pd.to_datetime(scoring_df["fecha_venta"])
        buckets = [
            ("< ventana 90d (sin cobertura)", last_sale < window_90_start),
            ("dentro ventana 90d",            last_sale >= window_90_start),
        ]
        for label, mask in buckets:
            n = mask.sum()
            print(f"  {label:<40s}: {n:,} combos ({100*n/len(scoring_df):.1f}%)")

        print(f"\n  Distribución mensual de última venta:")
        monthly = last_sale.dt.to_period("M").value_counts().sort_index()
        for period, count in monthly.items():
            in_window = "← en ventana 90d" if pd.Timestamp(period.end_time) >= window_90_start else ""
            print(f"    {period}: {count:>5,}  {in_window}")

    # ── D. DISTRIBUCIÓN DE COBERTURA (días únicos vs rows) ────────────────────
    section("D. Distribución de dias_con_venta_90d y cobertura_score")

    # Método actual: .size() (filas)
    counts_rows = (
        df_90
        .groupby(GROUP_COLS)
        .size()
        .reset_index(name="dias_rows")
    )
    # Método correcto: fechas únicas
    counts_dates = (
        df_90
        .groupby(GROUP_COLS)["fecha_venta"]
        .nunique()
        .reset_index(name="dias_uniq")
    )

    merged_check = counts_rows.merge(counts_dates, on=GROUP_COLS, how="left")
    diff = (merged_check["dias_rows"] - merged_check["dias_uniq"]).abs()
    print(f"  Combos con dias_rows ≠ dias_uniq: {(diff > 0).sum():,}")
    if (diff > 0).sum() == 0:
        print("  ✅ .size() y .nunique() dan el mismo resultado → no hay duplicados de fecha")
    else:
        print(f"  ⚠️  Discrepancia en {(diff>0).sum()} combos — hay múltiples filas por día")

    # Filtrar a los combos del scoring_df
    merged_scoring = scoring_df[GROUP_COLS].merge(
        counts_dates.rename(columns={"dias_uniq": "dias_90d"}),
        on=GROUP_COLS, how="left"
    )
    merged_scoring["dias_90d"] = merged_scoring["dias_90d"].fillna(0)
    cob = np.minimum(1.0, merged_scoring["dias_90d"].values / 90.0)

    print(f"\n  cobertura_score (usando fechas únicas) sobre combos activos:")
    print(f"    Media    : {cob.mean():.4f}")
    print(f"    Mediana  : {np.median(cob):.4f}")
    print(f"    > 0      : {(cob > 0).sum():,} ({100*(cob>0).mean():.1f}%)")
    print(f"    > 0.1    : {(cob > 0.1).sum():,} ({100*(cob>0.1).mean():.1f}%)")
    print(f"    > 0.3    : {(cob > 0.3).sum():,} ({100*(cob>0.3).mean():.1f}%)")
    print(f"    Percentiles [10,25,50,75,90,99]: "
          f"{np.percentile(cob,[10,25,50,75,90,99]).round(4).tolist()}")

    # ── E. ¿QUÉ PASA SI USAMOS VENTANA RELATIVA POR COMBO? ──────────────────
    section("E. Alternativa: ventana relativa al último scoring — 90d desde la última venta")

    # En vez de 90d desde scoring_date global, usar 90d desde la última venta del combo
    if "fecha_venta" in scoring_df.columns:
        last_sale_map = scoring_df.set_index(GROUP_COLS)["fecha_venta"].to_dict()

        def _days_in_window_relative(combo_key, last_sale_date, df_90_full):
            combo_start = last_sale_date - pd.Timedelta(days=90)
            mask = (
                (df_90_full["tienda_id"] == combo_key[0]) &
                (df_90_full["generico_id"] == combo_key[1]) &
                (df_90_full["fecha_venta"] >= combo_start) &
                (df_90_full["fecha_venta"] <= last_sale_date)
            )
            return df_90_full[mask]["fecha_venta"].nunique()

        # Para los primeros 50 combos (muestra rápida)
        sample_keys = list(
            zip(scoring_df["tienda_id"].head(50), scoring_df["generico_id"].head(50))
        )
        df_sample_range = df[
            df["tienda_id"].isin(scoring_df["tienda_id"].head(50)) &
            df["generico_id"].isin(scoring_df["generico_id"].head(50))
        ]
        rel_scores = []
        for tk, gk in sample_keys:
            last = last_sale_map.get((tk, gk))
            if last is None:
                rel_scores.append(0)
                continue
            combo_start = last - pd.Timedelta(days=90)
            n = df_sample_range[
                (df_sample_range["tienda_id"] == tk) &
                (df_sample_range["generico_id"] == gk) &
                (df_sample_range["fecha_venta"] >= combo_start) &
                (df_sample_range["fecha_venta"] <= last)
            ]["fecha_venta"].nunique()
            rel_scores.append(min(1.0, n / 90.0))

        rel_arr = np.array(rel_scores)
        print(f"  (Muestra rápida: {len(rel_arr)} combos)")
        print(f"  cobertura_score con ventana RELATIVA (90d desde última venta):")
        print(f"    Media   : {rel_arr.mean():.4f}")
        print(f"    Mediana : {np.median(rel_arr):.4f}")
        print(f"    > 0     : {(rel_arr > 0).sum()} ({100*(rel_arr>0).mean():.1f}%)")
        print(f"    > 0.1   : {(rel_arr > 0.1).sum()} ({100*(rel_arr>0.1).mean():.1f}%)")
        if rel_arr.mean() > cob[:50].mean() * 2:
            print("  ✅ Ventana relativa da cobertura significativamente mayor")
            print("     → Considerar cambiar a ventana relativa al último scoring del combo")
        else:
            print("  → Ventana relativa no mejora significativamente la cobertura")

    # ── F. LECTURA DEL scoring_output.csv REAL ───────────────────────────────
    section("F. Análisis del scoring_output.csv producido por score.py")

    scoring_out = ROOT / "reports" / "scoring" / "scoring_output.csv"
    if scoring_out.exists():
        so = pd.read_csv(scoring_out)
        print(f"  Filas: {len(so):,}")
        print(f"\n  confidence_score:")
        print(f"    Media    : {so['confidence_score'].mean():.4f}")
        print(f"    Mediana  : {so['confidence_score'].median():.4f}")
        print(f"    = 0      : {(so['confidence_score'] == 0).sum():,} ({100*(so['confidence_score']==0).mean():.1f}%)")
        print(f"    > 0      : {(so['confidence_score'] > 0).sum():,} ({100*(so['confidence_score']>0).mean():.1f}%)")
        print(f"    > 0.1    : {(so['confidence_score'] > 0.1).sum():,}")
        print(f"    > 0.3    : {(so['confidence_score'] > 0.3).sum():,}")
        print(f"    Percentiles [10,25,50,75,90,99]: "
              f"{so['confidence_score'].quantile([.1,.25,.5,.75,.9,.99]).round(4).tolist()}")

        print(f"\n  cobertura_score:")
        print(f"    Media    : {so['cobertura_score'].mean():.4f}")
        print(f"    = 0      : {(so['cobertura_score'] == 0).sum():,} ({100*(so['cobertura_score']==0).mean():.1f}%)")
        print(f"    > 0      : {(so['cobertura_score'] > 0).sum():,} ({100*(so['cobertura_score']>0).mean():.1f}%)")

        print(f"\n  is_extrapolated:")
        print(f"    True     : {so['is_extrapolated'].sum():,} ({100*so['is_extrapolated'].mean():.1f}%)")
        print(f"    False    : {(~so['is_extrapolated'].astype(bool)).sum():,}")

        print(f"\n  Combos con cobertura_score > 0 pero confidence_score = 0:")
        paradox = so[(so["cobertura_score"] > 0) & (so["confidence_score"] == 0)]
        print(f"    {len(paradox):,} combos  → todos tienen is_extrapolated=True")
    else:
        print("  scoring_output.csv no encontrado — corre score.py primero")

    print(f"\n{SEP}")
    print("  Diagnóstico completo.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
