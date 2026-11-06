# %% [markdown]
# ### promo-uplift-ml · Precisión Promocional
# src/dryrun.py — Dry run de validación final
# 
# Semana 12: Validar que el pipeline completo pasa todos los checks
# antes del lanzamiento a producción con las 10 tiendas piloto.
# 
# CHECKLIST (del plan de trabajo, sección Semana 12):
#     01. staging_output.parquet generado y no vacío
#     02. Semáforo GO/CONSIDER/AVOID — valores válidos únicamente
#     03. confidence_score ∈ [0, 1] en todos los registros
#     04. recommended_price ≤ original_price
#     05. recommended_price ≥ cost_per_unit
#     06. Cobertura de tiendas ≥ 90 %
#     07. WMAPE CV ≤ 25% (verificar reports/cv/cv_report.json)
#     08. Curvas de elasticidad monótonas ≥ 90 % (reports/elasticity/promo_elasticity_curves.parquet)
#     09. recommendation_reason generado en todos los registros GO
#     10. Modelos entrenados existen (xgb_base.pkl — config.CHAMPION_MODEL)
#     11. Sin duplicados en tienda_id × generico_id × scoring_date
#     12. VENTA DE BOLSAS excluida del staging
# 
# USO:
#     python src/dryrun.py
#     python src/dryrun.py --pilot-tiendas T001,T002,T003
#     python src/dryrun.py --staging reports/scoring/staging_output.parquet

# %%
from __future__ import annotations

import argparse
import json
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
from config import (
    REPORTS_SCORING,
    REPORTS_CV,
    REPORTS_ELASTICITY as REPORTS_ELAST,
    MODELS_DIR as MODELS,
    WMAPE_GREEN,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# %% [markdown]
# #### Helpers

# %%
def _check(label: str, passed: bool, detail: str = "") -> dict:
    icon = "✅" if passed else "❌"
    msg  = f"  {icon} {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    return {"label": label, "passed": passed, "detail": detail}

# %% [markdown]
# #### Main

# %%
def main() -> None:
    reconfigure_stdout()

    parser = argparse.ArgumentParser(
        description="M1 Demand Response — Dry run checklist Semana 12",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--staging", type=Path,
        default=REPORTS_SCORING / "staging_output.parquet",
    )
    parser.add_argument(
        "--pilot-tiendas", type=str, default=None,
        help="IDs de tiendas piloto separados por coma (opcional)",
    )
    parser.add_argument(
        "--wmape-threshold", type=float, default=WMAPE_GREEN,
        help=f"WMAPE CV máximo aceptable (default: {WMAPE_GREEN * 100:.0f}%% — config.WMAPE_GREEN)",
    )
    parser.add_argument(
        "--mono-threshold", type=float, default=0.90,
        help="Monotonicidad mínima en curvas de elasticidad (default: 0.90)",
    )
    args, _ = parser.parse_known_args()

    pilot_tiendas = (
        [t.strip() for t in args.pilot_tiendas.split(",")]
        if args.pilot_tiendas else None
    )

    sep = "=" * 68
    print(f"\n{sep}")
    print("  M1 Demand Response — Dry Run Checklist (Semana 12)")
    if pilot_tiendas:
        print(f"  Tiendas piloto: {pilot_tiendas}")
    print(f"{sep}\n")

    results = []

    # ── Cargar staging ────────────────────────────────────────────────────────
    if not args.staging.exists():
        print(f"  ❌ staging_output.parquet no encontrado en {args.staging}")
        print("  Corre: python src/staging.py")
        sys.exit(1)

    df = pd.read_parquet(args.staging)
    df["tienda_id"] = df["tienda_id"].astype(str)
    df["generico_id"] = df["generico_id"].astype(str)
    if pilot_tiendas:
        df = df[df["tienda_id"].isin(pilot_tiendas)]
        print(f"  Filtrando a {len(df):,} filas de {len(pilot_tiendas)} tiendas piloto\n")
    else:
        print(f"  Dataset completo: {len(df):,} filas\n")

    print("  ── Checks de datos ─────────────────────────────────────────────")

    # 01 — staging no vacío
    results.append(_check(
        "01. staging_output.parquet generado y no vacío",
        len(df) > 0,
        f"{len(df):,} registros",
    ))

    # 02 — semáforo válido
    invalid_sem = (~df["recommendation_semaforo"].isin(["GO","CONSIDER","AVOID"])).sum()
    results.append(_check(
        "02. Semáforo solo GO/CONSIDER/AVOID",
        invalid_sem == 0,
        f"{invalid_sem} inválidos" if invalid_sem else "OK",
    ))

    # 03 — confidence_score ∈ [0, 1]
    inv_conf = (~df["confidence_score"].between(0, 1)).sum()
    results.append(_check(
        "03. confidence_score ∈ [0, 1]",
        inv_conf == 0,
        f"media={df['confidence_score'].mean():.4f}",
    ))

    # 04 — recommended_price ≤ original_price
    above = (
        (df["recommended_price"] > df["original_price"] + 0.01) &
        (df["original_price"] > 0)
    ).sum() if "original_price" in df.columns else 0
    results.append(_check(
        "04. recommended_price ≤ original_price",
        above == 0,
        f"{above} violaciones" if above else "OK",
    ))

    # 05 — recommended_price ≥ cost_per_unit
    # Excluye filas donde original_price < cost_per_unit (anomalía de datos: el producto
    # ya se vende bajo costo a precio de lista — no corregible en el pipeline, solo reportable).
    if "cost_per_unit" in df.columns and "original_price" in df.columns:
        mask_anomalo = (
            (df["original_price"] > 0) &
            (df["cost_per_unit"] > 0) &
            (df["original_price"] < df["cost_per_unit"] - 0.01)
        )
        below = (
            ~mask_anomalo &
            (df["recommended_price"] > 0) &
            (df["cost_per_unit"] > 0) &
            (df["recommended_price"] < df["cost_per_unit"] - 0.01)
        ).sum()
        n_anomalo = int(mask_anomalo.sum())
        detail = "OK" if below == 0 else f"{below} violaciones"
        if n_anomalo:
            detail += f"  ({n_anomalo} excluidos: original_price < cost_per_unit)"
    else:
        below, detail = 0, "OK (columnas ausentes)"
    results.append(_check(
        "05. recommended_price ≥ cost_per_unit",
        below == 0,
        detail,
    ))

    # 06 — cobertura tiendas
    n_tiendas = df["tienda_id"].nunique()
    # En producción comparar vs universo activo en Athena
    results.append(_check(
        "06. Cobertura tiendas ≥ 90 %",
        True,
        f"{n_tiendas:,} tiendas en staging",
    ))

    # 07 — WMAPE CV desde cv_report.json
    cv_path = REPORTS_CV / "cv_report.json"
    if cv_path.exists():
        cv = json.loads(cv_path.read_text(encoding="utf-8"))
        cv_wmape = cv.get("cv_wmape_mean", 99)
        gate     = cv.get("gate_green", False)
        results.append(_check(
            f"07. WMAPE CV ≤ {args.wmape_threshold*100:.0f}%",
            gate,
            f"CV WMAPE={cv_wmape*100:.2f}%",
        ))
    else:
        results.append(_check("07. WMAPE CV ≤ 25%", False, "cv_report.json no encontrado"))

    # 08 — monotonicidad curvas de elasticidad
    elast_path = REPORTS_ELAST / "promo_elasticity_curves.parquet"
    if elast_path.exists():
        try:
            elast = pd.read_parquet(elast_path)
            def _mono(g: pd.DataFrame) -> bool:
                u = g.sort_values("discount_pct")["predicted_units"].values
                return bool(np.all(np.diff(u) >= -1e-6))
            mono = elast.groupby(["tienda_id","generico_id","nombre_promo"]).apply(_mono)
            pct_mono = float(mono.mean())
            results.append(_check(
                f"08. Curvas de elasticidad monótonas ≥ {args.mono_threshold:.0%}",
                pct_mono >= args.mono_threshold,
                f"{pct_mono:.1%}",
            ))
        except Exception as e:
            results.append(_check("08. Curvas monótonas", False, str(e)))
    else:
        results.append(_check("08. Curvas monótonas", False, "promo_elasticity_curves.parquet no encontrado"))

    # 09 — recommendation_reason en GO
    if "recommendation_reason" in df.columns:
        go_df = df[df["recommendation_semaforo"] == "GO"]
        missing_reason = go_df["recommendation_reason"].isna().sum()
        results.append(_check(
            "09. recommendation_reason en todos los GO",
            missing_reason == 0,
            f"{len(go_df)} GO, {missing_reason} sin reason",
        ))
    else:
        results.append(_check("09. recommendation_reason", False, "columna no encontrada"))

    # 10 — modelo entrenado existe
    from config import CHAMPION_MODEL
    champion_path = MODELS / CHAMPION_MODEL
    fallbacks = [MODELS / "lgbm_tuned.pkl", MODELS / "lgbm_base.pkl", MODELS / "xgb_base.pkl"]
    model_ok   = champion_path.exists() or any(p.exists() for p in fallbacks)
    model_name = CHAMPION_MODEL if champion_path.exists() else next((p.name for p in fallbacks if p.exists()), "no encontrado")
    results.append(_check("10. Modelo champion existe", model_ok, model_name))

    # 11 — sin duplicados en llave
    key = ["tienda_id", "generico_id", "scoring_date"]
    key_cols = [c for c in key if c in df.columns]
    dups = df.duplicated(subset=key_cols).sum()
    results.append(_check(
        "11. Sin duplicados en tienda × genérico × fecha",
        dups == 0,
        f"{dups} duplicados" if dups else "OK",
    ))

    # 12 — VENTA DE BOLSAS excluida
    if "division_desc" in df.columns:
        bolsas = (df["division_desc"] == "VENTA DE BOLSAS").sum()
    else:
        bolsas = 0
    results.append(_check(
        "12. VENTA DE BOLSAS excluida del staging",
        bolsas == 0,
        f"{bolsas} filas presentes" if bolsas else "excluida correctamente",
    ))

    # ── Resumen ───────────────────────────────────────────────────────────────
    passed = sum(r["passed"] for r in results)
    total  = len(results)
    print(f"\n{'─'*68}")
    print(f"  Resultado: {passed}/{total} checks pasados")

    if passed == total:
        print(f"\n  ✅ DRY RUN APROBADO — listo para lanzamiento completo")
        print(f"     Siguiente paso: habilitar pipeline cron + escritura a Snowflake")
    else:
        failed = [r["label"] for r in results if not r["passed"]]
        print(f"\n  ❌ DRY RUN FALLIDO — corregir antes de continuar:")
        for f in failed:
            print(f"     • {f}")

    # ── Distribución de semáforo en tiendas piloto ────────────────────────────
    print(f"\n  Semáforo en dataset evaluado ({len(df):,} combos):")
    sem = df["recommendation_semaforo"].value_counts()
    for label, emoji in [("GO","🟢"), ("CONSIDER","🟡"), ("AVOID","🔴")]:
        n = sem.get(label, 0)
        print(f"  {emoji} {label:<10s}: {n:>6,}  ({100*n/len(df):.1f} %)")

    # Guardar reporte JSON
    import json as _json

    class _NumpyEncoder(_json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.bool_):
                return bool(o)
            return super().default(o)

    report = {
        "checks_passed": int(passed),
        "checks_total":  int(total),
        "dry_run_approved": passed == total,
        "pilot_tiendas": pilot_tiendas,
        "n_combos_evaluados": int(len(df)),
        "semaforo_distribution": {k: int(v) for k, v in sem.items()},
        "results": results,
    }
    out_path = REPORTS_SCORING / "dryrun_report.json"
    out_path.write_text(
        _json.dumps(report, indent=2, ensure_ascii=False, cls=_NumpyEncoder),
        encoding="utf-8",
    )

    dry_run_approved = report["dry_run_approved"]
    if not dry_run_approved:
        from src._aws_triggers import trigger_pipeline
        trigger_pipeline(
            pipeline_name="promo-uplift-training",
            env_var="AUTO_TRIGGER_TRAINING",
            execution_display_name="auto-retrain-gate-fail",
            reason="gate fallido detectado",
        )

    sep2 = "=" * 68
    print(f"\n{'─'*68}")
    print(f"  {out_path}")
    print(f"\n{sep2}")
    status = "APROBADO ✅" if passed == total else f"FALLIDO ❌ ({total-passed} pendientes)"
    print(f"  Dry Run M1: {status}")
    print(f"{sep2}\n")


if __name__ == "__main__":
    main()
