"""
deploy/run_local.py — Simula el pipeline M1 en local (sin SageMaker).

Ejecuta cada paso como subprocess en el orden correcto. Lee y escribe en
ROOT/data, ROOT/models, ROOT/reports/ tal como config.py lo define cuando
no hay variables SM_* presentes.

PIPELINES:
    training  — Pasos 1–4: tune → train → cv → explain
    scoring   — Pasos 5–9: elasticity → score → staging → dryrun → publish
    all       — Pasos 1–9 (training primero, luego scoring)

USO:
    python deploy/run_local.py                              # pipeline completo (1–9)
    python deploy/run_local.py --pipeline scoring          # solo pasos 5–9
    python deploy/run_local.py --pipeline training         # solo pasos 1–4
    python deploy/run_local.py --from-step 6               # desde score.py en adelante
    python deploy/run_local.py --pipeline scoring --dry-run # muestra pasos sin ejecutar
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_DRYRUN_REPORT = ROOT / "reports" / "scoring" / "dryrun_report.json"

TRAINING_STEPS: list[tuple[int, str, list[str]]] = [
    (1, "tune",        ["src/tune.py", "--n-trials", "1"]),
    (2, "train",       ["src/train.py", "--n-estimators", "2000"]),
    (3, "cv",          ["src/cv.py", "--n-splits", "1", "--gap-days", "28", "--n-estimators", "20"]),
    (4, "explain",     ["src/explain.py"]),
]

SCORING_STEPS: list[tuple[int, str, list[str]]] = [
    (5, "elasticity",  ["src/elasticity.py", "--min-events", "3"]),
    (6, "score",       ["src/score.py", "--active-window-days", "180"]),
    (7, "staging",     ["src/staging.py"]),
    (8, "dryrun",      ["src/dryrun.py"]),
    (9, "publish",     ["src/publish.py"]),
]

ALL_STEPS = TRAINING_STEPS + SCORING_STEPS


def _run_step(
    step_num: int, name: str, cmd_args: list[str], dry_run: bool,
    timings: list[tuple[str, float]],
) -> bool:
    """Ejecuta un paso. Retorna True si tuvo éxito, False si falló."""
    cmd = [sys.executable] + cmd_args
    label = f"[{step_num}] {name}"
    print(f"\n{'─'*68}")
    print(f"  {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'─'*68}")

    if dry_run:
        print("  (dry-run — no ejecutado)")
        return True

    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.monotonic() - t0
    timings.append((label, elapsed))

    if result.returncode != 0:
        print(f"\n{'='*68}")
        print(f"  ❌  {label} FALLADO (exit {result.returncode})")
        print(f"{'='*68}\n")
        return False

    print(f"\n  ✅  {label} completado en {elapsed:.1f}s")
    return True


def _run_steps(
    steps: list[tuple[int, str, list[str]]], dry_run: bool,
    timings: list[tuple[str, float]],
) -> bool:
    """Ejecuta una lista de pasos en orden. Retorna False si alguno falla."""
    for step_num, name, cmd_args in steps:
        if not _run_step(step_num, name, cmd_args, dry_run, timings):
            return False
    return True


def _print_timings(timings: list[tuple[str, float]]) -> None:
    """Imprime un resumen final de duración por step, en minutos."""
    if not timings:
        return
    sep = "=" * 68
    print(f"\n{sep}")
    print("  Resumen de tiempos por step")
    print(f"{sep}")
    for label, elapsed in timings:
        print(f"  {label:<20s} {elapsed/60:>7.2f} min")
    total = sum(elapsed for _, elapsed in timings)
    print(f"  {'─'*66}")
    print(f"  {'TOTAL':<20s} {total/60:>7.2f} min")
    print(f"{sep}")


def _dryrun_approved() -> bool:
    """Lee dry_run_approved del reporte generado por dryrun.py."""
    if not _DRYRUN_REPORT.exists():
        return False
    try:
        report = json.loads(_DRYRUN_REPORT.read_text(encoding="utf-8"))
        return bool(report.get("dry_run_approved", False))
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline M1 en local (sin SageMaker)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pipeline", choices=["training", "scoring", "all"], default="all",
        help="Subconjunto de pasos a ejecutar (default: all)",
    )
    parser.add_argument(
        "--from-step", type=int, default=1, metavar="N",
        help="Número de paso desde el que iniciar (1–9). Útil para reanudar tras un fallo.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Muestra los pasos que se ejecutarían sin correrlos.",
    )
    parser.add_argument(
        "--retrain-on-dryrun-fail", action="store_true",
        help=(
            "Si publish falla por dry_run_approved=False, re-ejecuta el pipeline de "
            "training (pasos 1–4) y reintenta scoring (pasos 5–9) una vez."
        ),
    )
    args = parser.parse_args()

    if args.pipeline == "training":
        steps = TRAINING_STEPS
    elif args.pipeline == "scoring":
        steps = SCORING_STEPS
    else:
        steps = ALL_STEPS

    steps = [(n, name, cmd) for n, name, cmd in steps if n >= args.from_step]

    if not steps:
        print(f"No hay pasos ≥ {args.from_step} en el pipeline '{args.pipeline}'.")
        sys.exit(0)

    sep = "=" * 68
    mode = "DRY-RUN" if args.dry_run else "EJECUCIÓN LOCAL"
    print(f"\n{sep}")
    print(f"  M1 Demand Response — {mode}")
    print(f"  Pipeline : {args.pipeline}  |  Pasos: {steps[0][0]}–{steps[-1][0]}")
    print(f"  ROOT     : {ROOT}")
    print(f"{sep}")

    timings: list[tuple[str, float]] = []
    t_total = time.monotonic()
    ok = _run_steps(steps, args.dry_run, timings)

    if not ok and args.retrain_on_dryrun_fail and not _dryrun_approved():
        print(f"\n{sep}")
        print("  🔄  dry_run_approved=False detectado — reentrenando (pasos 1–4)...")
        print(f"{sep}")
        retrain_ok = _run_steps(TRAINING_STEPS, args.dry_run, timings)
        if not retrain_ok:
            _print_timings(timings)
            print(f"\n{sep}")
            print("  ❌  Reentrenamiento fallado — pipeline detenido.")
            print(f"{sep}\n")
            sys.exit(1)
        print(f"\n{sep}")
        print("  🔄  Reentrenamiento completado — reintentando scoring (pasos 5–9)...")
        print(f"{sep}")
        ok = _run_steps(SCORING_STEPS, args.dry_run, timings)

    if not ok:
        _print_timings(timings)
        print(f"\n{sep}")
        print("  ❌  Pipeline detenido. Revisa el output de arriba para el motivo.")
        print(f"{sep}\n")
        sys.exit(1)

    elapsed_total = time.monotonic() - t_total
    _print_timings(timings)
    print(f"\n{sep}")
    status = "Simulación completa" if args.dry_run else f"Pipeline completado en {elapsed_total/60:.2f} min"
    print(f"  ✅  {status}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
