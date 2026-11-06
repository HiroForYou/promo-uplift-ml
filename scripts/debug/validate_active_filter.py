"""
scripts/validate_active_filter.py — Valida el filtro --active-window-days y la salida del pipeline

Ejecuta score.py con y sin el filtro, luego staging.py + dryrun.py con el output filtrado.

USO:
    python scripts/validate_active_filter.py
    python scripts/validate_active_filter.py --window 90
    python scripts/validate_active_filter.py --window 365

SALIDA:
    Resumen de combos filtrados, distribución de confidence_score, y resultados de dryrun.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORE_PY   = ROOT / "src" / "score.py"
STAGING_PY = ROOT / "src" / "staging.py"
DRYRUN_PY  = ROOT / "src" / "dryrun.py"


def run(cmd: list[str], label: str) -> tuple[bool, str]:
    print(f"\n{'─'*68}")
    print(f"  [{label}]")
    print(f"  > {' '.join(str(c) for c in cmd)}")
    print(f"{'─'*68}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    ok = result.returncode == 0
    status = "OK" if ok else f"ERROR (exit {result.returncode})"
    print(f"\n  [{label}] → {status}")
    return ok, status


def main():
    parser = argparse.ArgumentParser(description="Validación pipeline con filtro activo")
    parser.add_argument("--window", type=int, default=180,
                        help="Días de ventana activa (default: 180)")
    args = parser.parse_args()

    sep = "=" * 68
    print(f"\n{sep}")
    print("  Validación pipeline v1.3 — filtro --active-window-days")
    print(f"  Ventana: {args.window} días")
    print(f"{sep}")

    python = sys.executable
    results = {}

    # 1. Score SIN filtro (baseline)
    ok, status = run(
        [python, str(SCORE_PY)],
        "score.py (sin filtro)"
    )
    results["score_base"] = ok

    # 2. Score CON filtro
    ok, status = run(
        [python, str(SCORE_PY), f"--active-window-days={args.window}"],
        f"score.py (--active-window-days={args.window})"
    )
    results["score_filtered"] = ok

    # 3. Staging (lee scoring_output.csv generado por score.py)
    if ok:
        ok2, _ = run(
            [python, str(STAGING_PY)],
            "staging.py"
        )
        results["staging"] = ok2
    else:
        print("\n  [staging.py] OMITIDO — score.py filtrado falló")
        results["staging"] = False

    # 4. Dryrun
    if results.get("staging"):
        ok3, _ = run(
            [python, str(DRYRUN_PY)],
            "dryrun.py"
        )
        results["dryrun"] = ok3
    else:
        print("\n  [dryrun.py] OMITIDO — staging.py falló")
        results["dryrun"] = False

    # Resumen
    print(f"\n{sep}")
    print("  RESUMEN")
    print(f"{'─'*68}")
    all_ok = True
    for step, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {step}")
        all_ok = all_ok and ok

    print(f"{'─'*68}")
    if all_ok:
        print("  ✅ Pipeline completo OK con filtro activo")
    else:
        print("  ❌ Algún paso falló — revisar output arriba")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
