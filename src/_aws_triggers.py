# %% [markdown]
# ### promo-uplift-ml · Precisión Promocional
# src/_aws_triggers.py — Encadenado de pipelines SageMaker vía boto3
#
# Helper compartido para que un step dispare la ejecución de otro pipeline
# al terminar (p.ej. Explain → promo-uplift-scoring, Dryrun → promo-uplift-training).
# Gateado por env var para no requerir AWS/boto3 en corridas locales.
#
# Convención de nombres: AUTO_TRIGGER_<PIPELINE_DESTINO> (ver deploy/pipeline_def.py).

# %%
from __future__ import annotations

import os


def trigger_pipeline(
    pipeline_name: str,
    env_var: str,
    execution_display_name: str,
    reason: str,
) -> None:
    """Dispara `pipeline_name` si `env_var` está en "true" en el entorno.

    No-op silencioso si el env var no está activo (corrida local/notebook).
    """
    if os.environ.get(env_var, "false").lower() != "true":
        return

    import boto3  # lazy — solo requerido en ejecución AWS

    region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    sm = boto3.client("sagemaker", region_name=region)
    execution = sm.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName=execution_display_name,
    )
    print(f"\n  🔄  {env_var} activo — {reason}.")
    print(f"  Pipeline disparado: {pipeline_name}")
    print(f"  Execution ARN: {execution['PipelineExecutionArn']}")
