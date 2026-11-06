# %% [markdown]
# ### promo-uplift-ml · Precisión Promocional
# src/publish.py — Paso 9 de 9: Publica artefactos aprobados a S3.
# 
# ÚNICO GATE DEL PIPELINE DE SCORING:
#     Lee dryrun_report.json → dry_run_approved
#     Si False → RuntimeError: pipeline se detiene, nada llega a S3/publish/.
#     Si True  → copia staging_output.parquet y promo_elasticity_curves.parquet
#                al directorio de publicación.
#
# SNOWFLAKE:
#     El proceso DataOps externo lee los archivos del directorio de publicación
#     (S3 o local) y ejecuta el MERGE automáticamente:
#         PROMO_PREDICTIONS        ← staging_output.parquet
#         PROMO_ELASTICITY_CURVES  ← promo_elasticity_curves.parquet
# 
#     En SageMaker, PUBLISH_DIR (SM_PUBLISH_DIR) se mapea a
#     s3://<PROMO_S3_BUCKET>/m1/publish/ vía ProcessingOutput.
# 
# USO:
#     python src/publish.py
#     python src/publish.py --dry-run    # verifica gate, no copia archivos

# %%
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
except NameError:
    _cwd = Path().resolve()
    sys.path.insert(0, str(_cwd if (_cwd / "config.py").exists() else _cwd.parent))
from src._console import reconfigure_stdout
from config import REPORTS_SCORING, REPORTS_ELASTICITY, PUBLISH_DIR

PREDICTIONS_DIR = PUBLISH_DIR / "PROMO_PREDICTIONS"
ELASTICITY_DIR  = PUBLISH_DIR / "PROMO_ELASTICITY_CURVES"

_STAGING_FILE   = REPORTS_SCORING    / "staging_output.parquet"
_ELASTICITY_FILE = REPORTS_ELASTICITY / "promo_elasticity_curves.parquet"
_DRYRUN_FILE    = REPORTS_SCORING    / "dryrun_report.json"

# %% [markdown]
# #### Main

# %%
def main() -> None:
    reconfigure_stdout()

    parser = argparse.ArgumentParser(
        description="M1 Demand Response — Publicación a S3 (paso 9)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Verificar gate y archivos sin copiar nada",
    )
    parser.add_argument(
        "--staging",    type=Path, default=_STAGING_FILE,
        help="staging_output.parquet (default: REPORTS_SCORING/staging_output.parquet)",
    )
    parser.add_argument(
        "--elasticity", type=Path, default=_ELASTICITY_FILE,
        help="promo_elasticity_curves.parquet (default: REPORTS_ELASTICITY/…parquet)",
    )
    parser.add_argument(
        "--dryrun-report", type=Path, default=_DRYRUN_FILE,
        help="dryrun_report.json (default: REPORTS_SCORING/dryrun_report.json)",
    )
    args, _ = parser.parse_known_args()

    sep = "=" * 68
    print(f"\n{sep}")
    print("  M1 Demand Response — Publish (paso 9 de 9)")
    mode = "DRY-RUN — solo verificación, sin copia" if args.dry_run else "PUBLICACIÓN"
    print(f"  Modo: {mode}")
    print(f"{sep}\n")

    # ── Gate: dry_run_approved ────────────────────────────────────────────────
    if not args.dryrun_report.exists():
        raise FileNotFoundError(
            f"No encontrado: {args.dryrun_report}\n"
            "Corre primero: python src/dryrun.py"
        )

    report = json.loads(args.dryrun_report.read_text(encoding="utf-8"))
    approved   = report.get("dry_run_approved", False)
    n_passed   = report.get("checks_passed", "?")
    n_total    = report.get("checks_total",  "?")
    wmape_info = ""
    for r in report.get("results", []):
        if "07." in r.get("label", ""):
            wmape_info = f"  ({r.get('detail', '')})"
            break

    print(f"  dryrun_report.json → {n_passed}/{n_total} checks pasados{wmape_info}")

    if not approved:
        failed = [r["label"] for r in report.get("results", []) if not r.get("passed")]
        details = "\n".join(f"     • {f}" for f in failed)
        raise RuntimeError(
            f"❌ DRY RUN NO APROBADO — publicación bloqueada.\n"
            f"   Checks fallidos:\n{details}\n"
            f"   Corregir y re-correr el pipeline antes de publicar."
        )

    print(f"  ✅ dry_run_approved = True — procediendo a publicación\n")

    # ── Verificar archivos fuente ─────────────────────────────────────────────
    for label, path in [
        ("staging_output.parquet",          args.staging),
        ("promo_elasticity_curves.parquet", args.elasticity),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}  ({label})")
        size_kb = path.stat().st_size / 1024
        print(f"  ✅ {label}  ({size_kb:.0f} KB)")

    if args.dry_run:
        print(f"\n  Modo dry-run — archivos no copiados.")
        print(f"  Destino habría sido: {PUBLISH_DIR}")
        print(f"\n{sep}\n")
        return

    # ── Copiar a directorio de publicación ────────────────────────────────────
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ELASTICITY_DIR.mkdir(parents=True, exist_ok=True)

    dest_staging    = PREDICTIONS_DIR / args.staging.name
    dest_elasticity = ELASTICITY_DIR  / args.elasticity.name

    shutil.copy2(args.staging,    dest_staging)
    shutil.copy2(args.elasticity, dest_elasticity)

    print(f"\n  Publicado en:")
    print(f"    {dest_staging}")
    print(f"    {dest_elasticity}")
    print(f"\n  En SageMaker este directorio se mapea a:")
    print(f"    s3://<PROMO_S3_BUCKET>/m1/publish/")

    print(f"\n{sep}")
    print("  ✅ Publicación completada — DataOps recoge los archivos para Snowflake")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
