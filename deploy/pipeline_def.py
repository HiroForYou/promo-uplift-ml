"""
deploy/pipeline_def.py — Registra los dos pipelines de SageMaker.

Ejecutar UNA vez (y cada vez que cambie la estructura de pasos).
El código fuente lo provee el tar.gz que sube deploy_sm.py.

    python deploy/pipeline_def.py                  # upsert ambos pipelines
    python deploy/pipeline_def.py --only training
    python deploy/pipeline_def.py --only scoring

FLUJO RECOMENDADO:
    python deploy/deploy_sm.py           # sube código + requirements.txt a S3
    python deploy/pipeline_def.py        # registra/actualiza definición en SageMaker
    python deploy/deploy_sm.py --run     # ejecuta

REQUISITOS:
    SAGEMAKER_ROLE_ARN   env var — ARN del rol de ejecución SageMaker

PIPELINES:
    promo-uplift-training   Pasos 1–4: tune → train → cv + explain (paralelos)   mensual
    promo-uplift-scoring    Pasos 5–9: elasticity + score (paralelos) → staging → dryrun → publish
                         Cron: lunes 02:00 CST
"""

from __future__ import annotations

import argparse
import os

import boto3
import sagemaker
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep

# ── AWS ───────────────────────────────────────────────────────────────────────
BUCKET   = os.environ["PROMO_S3_BUCKET"]
REGION   = "us-west-2"
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]

# Si True, el step Dryrun dispara promo-uplift-training automáticamente cuando gate_green=False.
AUTO_TRIGGER_TRAINING: bool = True

# Si True, el step Explain dispara promo-uplift-scoring automáticamente al terminar Training.
AUTO_TRIGGER_SCORING: bool = True

# Si True, agrega a cada step los mismos flags de corrida rápida que usa
# deploy/run_local.py (1 trial, 1 fold, etc.) — útil para probar el pipeline
# completo en SageMaker sin gastar tiempo/cómputo real. Poner en False para producción.
TEST_MODE: bool = True

# Mismos flags que TRAINING_STEPS/SCORING_STEPS en deploy/run_local.py.
_TEST_ARGS: dict[str, list[str]] = {
    "tune":       ["--n-trials", "1"],
    "train":      ["--n-estimators", "2000"],
    "cv":         ["--n-splits", "1", "--gap-days", "28", "--n-estimators", "20"],
    "elasticity": ["--min-events", "3"],
    "score":      ["--active-window-days", "180"],
}


def _args(step_name: str, base: list[str] | None = None) -> list[str]:
    """Argumentos fijos del step + flags de TEST_MODE (si está activo)."""
    extra = _TEST_ARGS.get(step_name, []) if TEST_MODE else []
    return (base or []) + extra

# ── Tipos de instancia SageMaker ──────────────────────────────────────────────
IT_TUNE    = "ml.t3.medium"   # tune — Optuna multi-trial, compute intensivo
IT_TRAIN   = "ml.t3.medium"   # train — ajuste final LGBM + XGBoost
IT_CV      = "ml.t3.medium"   # cv — CV 5-fold, memoria moderada
IT_EXPLAIN = "ml.t3.medium"   # explain — SHAP sobre dataset completo
IT_SCORE   = "ml.t3.medium"    # elasticity / score — lectura + predicción
IT_LIGHT   = "ml.t3.medium"     # staging / dryrun / publish — operaciones ligeras

# ── Rutas S3 ──────────────────────────────────────────────────────────────────
_BASE      = f"s3://{BUCKET}/m1"
S3_CODE    = f"{_BASE}/code/sourcedir.tar.gz"
S3_DATA    = f"{_BASE}/data-train"
S3_MODELS  = f"{_BASE}/models"
S3_REPORTS = f"{_BASE}/reports"
S3_PUBLISH = f"{_BASE}/publish"          # DataOps/Snowflake lee de aquí

# ── Rutas en el contenedor ────────────────────────────────────────────────────
C_DATA      = "/opt/ml/processing/input/data"     # SM_CHANNEL_TRAIN
C_MODELS_RW = "/opt/ml/processing/models"         # SM_MODEL_DIR (tune/train: r+w)
C_MODELS_IN = "/opt/ml/processing/input/models"   # SM_MODEL_DIR (inferencia: solo lectura)
C_REPORTS   = "/opt/ml/processing/output/reports" # SM_OUTPUT_DATA_DIR
C_PUBLISH   = "/opt/ml/processing/output/publish" # SM_PUBLISH_DIR


def _env(model_dir: str, extra: dict | None = None) -> dict[str, str]:
    env = {
        "SM_CHANNEL_TRAIN":   C_DATA,
        "SM_MODEL_DIR":       model_dir,
        "SM_OUTPUT_DATA_DIR": C_REPORTS,
    }
    if extra:
        env.update(extra)
    return env


def _proc(instance_type: str, model_dir: str, sess: sagemaker.Session,
          extra_env: dict | None = None) -> FrameworkProcessor:
    return FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version="1.4-2",
        instance_type=instance_type,
        instance_count=1,
        role=ROLE_ARN,
        sagemaker_session=sess,
        env=_env(model_dir, extra_env),
    )


def _data_in() -> ProcessingInput:
    return ProcessingInput(
        source=f"{S3_DATA}/train-set-features.csv",
        destination=C_DATA,
        input_name="data",
        s3_data_distribution_type="FullyReplicated",
    )


def _models_in(source: str = S3_MODELS) -> ProcessingInput:
    return ProcessingInput(
        source=source,
        destination=C_MODELS_IN,
        input_name="models",
        s3_data_distribution_type="FullyReplicated",
    )


def _models_rw(source: str) -> tuple[ProcessingInput, ProcessingOutput]:
    """Input en C_MODELS_IN (read), output en C_MODELS_RW (write) — paths distintos requerido por SageMaker."""
    return (
        ProcessingInput(
            source=source,
            destination=C_MODELS_IN,
            input_name="models_in",
            s3_data_distribution_type="FullyReplicated",
        ),
        ProcessingOutput(
            source=C_MODELS_RW,
            destination=S3_MODELS,
            output_name="models",
        ),
    )


def _reports_out(subdir: str) -> ProcessingOutput:
    return ProcessingOutput(
        source=f"{C_REPORTS}/{subdir}",
        destination=f"{S3_REPORTS}/{subdir}",
        output_name=subdir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE DE ENTRENAMIENTO  promo-uplift-training
# Paso 1: tune  →  Paso 2: train  →  Paso 3: cv  ┐ (paralelos)
#                                    Paso 4: explain ┘
# ─────────────────────────────────────────────────────────────────────────────

def build_training_pipeline(sess: sagemaker.Session) -> Pipeline:
    # [1] tune — Optuna, escribe tune_meta.json en S3_MODELS
    step_tune = ProcessingStep(
        name="Tune",
        step_args=_proc(IT_TUNE, C_MODELS_RW, sess).run(
            code="src/tune.py",
            source_dir=S3_CODE,
            arguments=_args("tune"),
            inputs=[_data_in()],
            outputs=[
                ProcessingOutput(
                    source=C_MODELS_RW,
                    destination=S3_MODELS,
                    output_name="models",
                )
            ],
        ),
    )

    # [2] train — lee tune_meta.json, entrena LGBM + XGB, elige champion
    tune_uri = step_tune.properties.ProcessingOutputConfig.Outputs["models"].S3Output.S3Uri
    models_inp, models_out = _models_rw(tune_uri)
    step_train = ProcessingStep(
        name="Train",
        step_args=_proc(IT_TRAIN, C_MODELS_RW, sess, extra_env={"SM_INPUT_MODELS_DIR": C_MODELS_IN}).run(
            code="src/train.py",
            source_dir=S3_CODE,
            arguments=_args("train"),
            inputs=[_data_in(), models_inp],
            outputs=[models_out],
        ),
        depends_on=[step_tune],
    )

    trained_uri = step_train.properties.ProcessingOutputConfig.Outputs["models"].S3Output.S3Uri

    # [3] cv — CV temporal 5-fold LGBM + XGB, escribe cv_report.json
    step_cv = ProcessingStep(
        name="CV",
        step_args=_proc(IT_CV, C_MODELS_IN, sess).run(
            code="src/cv.py",
            source_dir=S3_CODE,
            arguments=_args("cv"),
            inputs=[_data_in(), _models_in(trained_uri)],
            outputs=[_reports_out("cv")],
        ),
        depends_on=[step_train],
    )

    # [4] explain — SHAP global (depende de cv para que el gate lo bloquee si falla)
    step_explain = ProcessingStep(
        name="Explain",
        step_args=_proc(IT_EXPLAIN, C_MODELS_IN, sess, extra_env={"AUTO_TRIGGER_SCORING": str(AUTO_TRIGGER_SCORING).lower()}).run(
            code="src/explain.py",
            source_dir=S3_CODE,
            inputs=[_data_in(), _models_in(trained_uri)],
            outputs=[_reports_out("shap")],
        ),
        depends_on=[step_cv],
    )

    return Pipeline(
        name="promo-uplift-training",
        steps=[step_tune, step_train, step_cv, step_explain],
        sagemaker_session=sess,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE DE SCORING  promo-uplift-scoring
# Paso 5: elasticity ┐ (paralelos)   →  Paso 7: staging  →  Paso 8: dryrun  →  Paso 9: publish
# Paso 6: score      ┘
# ─────────────────────────────────────────────────────────────────────────────

def build_scoring_pipeline(sess: sagemaker.Session) -> Pipeline:
    # [5] elasticity — curvas de elasticidad
    step_elasticity = ProcessingStep(
        name="Elasticity",
        step_args=_proc(IT_SCORE, C_MODELS_IN, sess).run(
            code="src/elasticity.py",
            source_dir=S3_CODE,
            arguments=_args("elasticity"),
            inputs=[_data_in(), _models_in()],
            outputs=[_reports_out("elasticity")],
        ),
    )

    # [6] score — predicciones + semáforo (paralelo con elasticity)
    step_score = ProcessingStep(
        name="Score",
        step_args=_proc(IT_SCORE, C_MODELS_IN, sess).run(
            code="src/score.py",
            source_dir=S3_CODE,
            arguments=_args("score"),
            inputs=[_data_in(), _models_in()],
            outputs=[_reports_out("scoring")],
        ),
    )

    # [7] staging — enriquece scoring_output + validaciones bloqueantes
    # staging.py busca scoring_output.csv en REPORTS_SCORING (= SM_OUTPUT_DATA_DIR/scoring)
    # → el output de score se monta en ese mismo path
    scoring_uri = step_score.properties.ProcessingOutputConfig.Outputs["scoring"].S3Output.S3Uri
    step_staging = ProcessingStep(
        name="Staging",
        step_args=_proc(IT_LIGHT, C_MODELS_IN, sess).run(
            code="src/staging.py",
            source_dir=S3_CODE,
            arguments=_args("staging", ["--scoring", f"{C_REPORTS}/scoring_in/scoring_output.csv"]),
            inputs=[
                _data_in(),
                ProcessingInput(
                    source=scoring_uri,
                    destination=f"{C_REPORTS}/scoring_in",
                    input_name="scoring_output",
                    s3_data_distribution_type="FullyReplicated",
                ),
            ],
            outputs=[_reports_out("scoring")],
        ),
        depends_on=[step_score],
    )

    # [8] dryrun — 12 checks cruzados sobre los 4 artefactos → dry_run_approved
    staging_uri    = step_staging.properties.ProcessingOutputConfig.Outputs["scoring"].S3Output.S3Uri
    elasticity_uri = step_elasticity.properties.ProcessingOutputConfig.Outputs["elasticity"].S3Output.S3Uri
    step_dryrun = ProcessingStep(
        name="Dryrun",
        step_args=_proc(IT_LIGHT, C_MODELS_IN, sess, extra_env={"AUTO_TRIGGER_TRAINING": str(AUTO_TRIGGER_TRAINING).lower()}).run(
            code="src/dryrun.py",
            source_dir=S3_CODE,
            arguments=_args("dryrun", ["--staging", f"{C_REPORTS}/scoring_in/staging_output.parquet"]),
            inputs=[
                ProcessingInput(
                    source=staging_uri,
                    destination=f"{C_REPORTS}/scoring_in",
                    input_name="staging_output",
                    s3_data_distribution_type="FullyReplicated",
                ),
                ProcessingInput(
                    source=elasticity_uri,
                    destination=f"{C_REPORTS}/elasticity",
                    input_name="elasticity_output",
                    s3_data_distribution_type="FullyReplicated",
                ),
                ProcessingInput(
                    source=f"{S3_REPORTS}/cv",
                    destination=f"{C_REPORTS}/cv",
                    input_name="cv_report",
                    s3_data_distribution_type="FullyReplicated",
                ),
                _models_in(),
            ],
            outputs=[_reports_out("scoring")],
        ),
        depends_on=[step_staging, step_elasticity],
    )

    # [9] publish — gate dry_run_approved → sube a S3/publish/ → DataOps/Snowflake
    dryrun_uri = step_dryrun.properties.ProcessingOutputConfig.Outputs["scoring"].S3Output.S3Uri
    step_publish = ProcessingStep(
        name="Publish",
        step_args=_proc(
            IT_LIGHT, C_MODELS_IN, sess,
            extra_env={"SM_PUBLISH_DIR": C_PUBLISH},
        ).run(
            code="src/publish.py",
            source_dir=S3_CODE,
            inputs=[
                ProcessingInput(
                    source=dryrun_uri,
                    destination=f"{C_REPORTS}/scoring",
                    input_name="dryrun_output",
                    s3_data_distribution_type="FullyReplicated",
                ),
                ProcessingInput(
                    source=elasticity_uri,
                    destination=f"{C_REPORTS}/elasticity",
                    input_name="elasticity_final",
                    s3_data_distribution_type="FullyReplicated",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    source=C_PUBLISH,
                    destination=S3_PUBLISH,
                    output_name="publish",
                )
            ],
        ),
        depends_on=[step_dryrun],
    )

    return Pipeline(
        name="promo-uplift-scoring",
        steps=[step_elasticity, step_score, step_staging, step_dryrun, step_publish],
        sagemaker_session=sess,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crea/actualiza las definiciones de pipeline en SageMaker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--only", choices=["training", "scoring"],
        help="Procesar solo un pipeline (default: ambos)",
    )
    args = parser.parse_args()

    boto_sess = boto3.Session(region_name=REGION)
    sess      = PipelineSession(boto_session=boto_sess)

    sep = "=" * 60
    print(f"\n{sep}")
    print("  Registro de pipelines M1  —  SageMaker")
    print(f"  Bucket : {BUCKET}")
    print(f"  Región : {REGION}")
    print(f"  Código : {S3_CODE}")
    print(f"{sep}\n")

    if args.only != "scoring":
        pipe = build_training_pipeline(sess)
        pipe.upsert(role_arn=ROLE_ARN)
        print("  ✅  promo-uplift-training  registrado/actualizado")

    if args.only != "training":
        pipe = build_scoring_pipeline(sess)
        pipe.upsert(role_arn=ROLE_ARN)
        print("  ✅  promo-uplift-scoring   registrado/actualizado")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
