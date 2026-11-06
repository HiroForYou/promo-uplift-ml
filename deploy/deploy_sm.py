"""
deploy/deploy_sm.py — Publica cambios de src/ a los pipelines de SageMaker.

PIPELINES:
    training  — Pasos 1–4: tune → train → cv → explain
                Frecuencia: mensual o bajo demanda
                Nombre SM:  promo-uplift-training

    scoring   — Pasos 5–9: elasticity → score → staging → dryrun → publish
                Frecuencia: semanal (cron lunes 02:00 CST)
                Nombre SM:  promo-uplift-scoring

    all       — Lanza training primero, luego scoring (útil para bootstrap inicial)

USO:
    python deploy/deploy_sm.py                              # solo sube código
    python deploy/deploy_sm.py --run                        # sube + lanza scoring (default)
    python deploy/deploy_sm.py --run --pipeline training    # sube + lanza entrenamiento
    python deploy/deploy_sm.py --run --pipeline all         # sube + lanza ambos en secuencia
    python deploy/deploy_sm.py --run --pipeline scoring --wait
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BUCKET  = os.environ["PROMO_S3_BUCKET"]
REGION  = "us-west-2"
S3_KEY  = "m1/code/sourcedir.tar.gz"
S3_BASE = f"s3://{BUCKET}/m1"

PIPELINE_NAMES = {
    "training": "promo-uplift-training",
    "scoring":  "promo-uplift-scoring",
}


def package_source() -> Path:
    """Empaqueta src/ + config.py + requirements.txt en un tar.gz temporal."""
    tmp = Path(tempfile.mkdtemp()) / "sourcedir.tar.gz"
    with tarfile.open(tmp, "w:gz") as tar:
        tar.add(ROOT / "src",       arcname="src")
        tar.add(ROOT / "config.py", arcname="config.py")
        req = ROOT / "requirements.txt"
        if req.exists():
            tar.add(req, arcname="requirements.txt")
    size_kb = tmp.stat().st_size / 1024
    print(f"  Empaquetado: {tmp}  ({size_kb:.0f} KB)")
    return tmp


def upload(source_tar: Path, sess) -> str:
    """Sube el tar.gz a S3 y retorna el URI."""
    s3_uri = sess.upload_data(str(source_tar), bucket=BUCKET, key_prefix="m1/code")
    print(f"  Subido a:    {s3_uri}")
    return s3_uri


def run_pipeline(pipeline_key: str, sess, wait: bool) -> None:
    """Lanza un pipeline por nombre y opcionalmente espera a que termine."""
    from sagemaker.workflow.pipeline import Pipeline

    name = PIPELINE_NAMES[pipeline_key]
    pipeline = Pipeline(name=name, sagemaker_session=sess)

    execution = pipeline.start(
        execution_display_name="deploy-manual",
        parallelism_config=None,
    )
    print(f"  [{pipeline_key}] Ejecución: {execution.arn}")

    if wait:
        print(f"  [{pipeline_key}] Esperando a que termine (Ctrl+C para salir sin cancelar)...")
        execution.wait()
        status = execution.describe()["PipelineExecutionStatus"]
        print(f"  [{pipeline_key}] Estado final: {status}")
        if status != "Succeeded":
            sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy de src/ a SageMaker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Lanzar ejecución tras subir código",
    )
    parser.add_argument(
        "--pipeline",
        choices=["training", "scoring", "all"],
        default="scoring",
        help=(
            "Pipeline a lanzar con --run  "
            "(training=pasos 1-4 mensual, scoring=pasos 5-9 semanal, all=ambos). "
            "Default: scoring"
        ),
    )
    parser.add_argument(
        "--wait", action="store_true",
        help="Esperar a que termine la ejecución",
    )
    args = parser.parse_args()

    import os
    import sagemaker
    os.environ.setdefault("AWS_DEFAULT_REGION", REGION)
    sess = sagemaker.Session()

    print("\n── Deploy M1 Demand Response ─────────────────────────────")
    source_tar = package_source()
    upload(source_tar, sess)

    if args.run:
        pipelines_to_run = (
            ["training", "scoring"] if args.pipeline == "all"
            else [args.pipeline]
        )
        for key in pipelines_to_run:
            run_pipeline(key, sess, wait=args.wait)

    print("── Listo ──────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
