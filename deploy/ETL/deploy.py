"""
deploy.py — Empaqueta, sube y ejecuta el pipeline promo-uplift-etl.

FLUJO:
    [1] Empaqueta etl.py + config.py + ctes.py + requirements.txt
    [2] Sube el tar.gz a S3
    [3] Registra/actualiza el pipeline en SageMaker
    [4] (opcional) Lanza ejecución

REQUISITOS:
    SAGEMAKER_ROLE_ARN   env var — ARN del rol de ejecución

USO:
    python deploy_etl/deploy.py                                         # registra
    python deploy_etl/deploy.py --run                                   # + ejecuta
    python deploy_etl/deploy.py --run --wait                            # + espera
    python deploy_etl/deploy.py --run --fecha-inicio 20260101 --fecha-fin 20260301
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    REGION, ROLE_ARN, BUCKET_OUT,
    S3_OUTPUT_DATA, C_OUTPUT,
    IT_ETL, PIPELINE_NAME, OUTPUT_FILE,
    FECHA_INICIO, FECHA_FIN,
)


S3_CODE_KEY = "m1/code-etl/sourcedir.tar.gz"
S3_CODE_URI = f"s3://{BUCKET_OUT}/{S3_CODE_KEY}"


def package_and_upload(s3_client) -> None:
    tmp = Path(tempfile.mkdtemp()) / "sourcedir.tar.gz"
    with tarfile.open(tmp, "w:gz") as tar:
        for fname in ("etl.py", "config.py", "ctes.py", "requirements.txt"):
            tar.add(ROOT / fname, arcname=fname)
    size_kb = tmp.stat().st_size / 1024
    s3_client.upload_file(str(tmp), BUCKET_OUT, S3_CODE_KEY)
    print(f"  Empaquetado y subido : {S3_CODE_URI}  ({size_kb:.0f} KB)")


def build_and_register(sess, fecha_inicio: str, fecha_fin: str):
    from sagemaker.processing import FrameworkProcessor, ProcessingOutput
    from sagemaker.sklearn.estimator import SKLearn
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.steps import ProcessingStep

    processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version="1.4-2",
        instance_type=IT_ETL,
        instance_count=1,
        role=ROLE_ARN,
        sagemaker_session=sess,
        env={
            "SM_OUTPUT_DATA_DIR": C_OUTPUT,
            "ETL_FECHA_INICIO":   fecha_inicio,
            "ETL_FECHA_FIN":      fecha_fin,
        },
    )

    step_etl = ProcessingStep(
        name="ETL",
        step_args=processor.run(
            code="etl.py",
            source_dir=S3_CODE_URI,
            inputs=[],
            outputs=[
                ProcessingOutput(
                    source=C_OUTPUT,
                    destination=S3_OUTPUT_DATA,
                    output_name="data",
                )
            ],
        ),
    )

    pipeline = Pipeline(name=PIPELINE_NAME, steps=[step_etl], sagemaker_session=sess)
    pipeline.upsert(role_arn=ROLE_ARN)
    print(f"  ✅ {PIPELINE_NAME}  registrado/actualizado")
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Deploy del pipeline {PIPELINE_NAME}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--run",  action="store_true", help="Lanzar ejecución tras registrar")
    parser.add_argument("--wait", action="store_true", help="Esperar a que termine la ejecución")
    parser.add_argument("--fecha-inicio", default=FECHA_INICIO, metavar="YYYYMMDD",
                        help=f"Inicio del período (default: {FECHA_INICIO})")
    parser.add_argument("--fecha-fin",    default=FECHA_FIN,    metavar="YYYYMMDD",
                        help=f"Fin del período (default: {FECHA_FIN})")
    args = parser.parse_args()

    if not ROLE_ARN:
        print("ERROR: env var SAGEMAKER_ROLE_ARN no definida.")
        sys.exit(1)

    import boto3
    from sagemaker.workflow.pipeline_context import PipelineSession

    os.environ.setdefault("AWS_DEFAULT_REGION", REGION)
    boto_sess = boto3.Session(region_name=REGION)
    sess = PipelineSession(boto_session=boto_sess)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Deploy — {PIPELINE_NAME}")
    print(f"  Período  : {args.fecha_inicio} → {args.fecha_fin}")
    print(f"  Instancia: {IT_ETL}")
    print(f"  Salida   : {S3_OUTPUT_DATA}/{OUTPUT_FILE}")
    print(f"{sep}\n")

    package_and_upload(boto_sess.client("s3"))
    pipeline = build_and_register(sess, args.fecha_inicio, args.fecha_fin)

    if args.run:
        execution = pipeline.start(execution_display_name="etl-manual")
        print(f"\n  Ejecución iniciada: {execution.arn}")
        if args.wait:
            print("  Esperando a que termine (Ctrl+C para salir sin cancelar)...")
            execution.wait()
            status = execution.describe()["PipelineExecutionStatus"]
            print(f"  Estado final: {status}")
            if status != "Succeeded":
                sys.exit(1)

    print(f"\n{sep}")
    print(f"  Listo. CSV en: {S3_OUTPUT_DATA}/{OUTPUT_FILE}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
