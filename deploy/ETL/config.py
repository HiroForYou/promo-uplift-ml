"""
config.py — Parámetros centrales del pipeline promo-uplift-etl.

Edita aquí para propagar cambios a etl.py y deploy.py
sin tocar el código de cada uno.

RUTAS — env vars para SageMaker (local: constantes de este archivo):
    SM_OUTPUT_DATA_DIR  → OUTPUT_DIR   (destino del CSV de salida)
    ETL_FECHA_INICIO    → FECHA_INICIO (inicio del período a procesar)
    ETL_FECHA_FIN       → FECHA_FIN    (fin del período a procesar)

USO EN SCRIPTS:
    from config import *
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── AWS ───────────────────────────────────────────────────────────────────────
REGION   = "us-west-2"
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", "")

S3_BUCKET_CAR_VNTS  = os.environ.get("RAW_BUCKET_SALES", "my-raw-sales")
S3_BUCKET_S4_INVS   = os.environ.get("RAW_BUCKET_INVENTORY", "my-raw-inventory")
S3_BUCKET_S4_PRCS   = os.environ.get("RAW_BUCKET_PRICES", "my-raw-prices")
S3_BUCKET_S4_DATOS  = os.environ.get("RAW_BUCKET_MASTERDATA", "my-raw-masterdata")

# ── Paths S3 de entrada ───────────────────────────────────────────────────────
S3_TLOGF     = f"s3://{S3_BUCKET_CAR_VNTS}/ventas-fisicas-tlogf"
S3_TLOGF_EXT = f"s3://{S3_BUCKET_CAR_VNTS}/promociones-fisicas-tlogf-ext"
S3_MARD      = f"s3://{S3_BUCKET_S4_INVS}/inventarios-mard"
S3_A155      = f"s3://{S3_BUCKET_S4_PRCS}/precio-venta-a155"
S3_A018      = f"s3://{S3_BUCKET_S4_PRCS}/precio-compra-a018"
S3_KONP      = f"s3://{S3_BUCKET_S4_PRCS}/valorespvpc-konp"
S3_PV_ECC    = f"s3://{S3_BUCKET_S4_PRCS}/precio-venta-ecc"
S3_PC_ECC    = f"s3://{S3_BUCKET_S4_PRCS}/precio-compra-ecc"
S3_CAL       = f"s3://{S3_BUCKET_S4_DATOS}/calendario-promocional"
S3_DIM_SUC   = f"s3://{S3_BUCKET_S4_DATOS}/dimension-sucursales"
S3_DIM_MAT   = f"s3://{S3_BUCKET_S4_DATOS}/dimension-materiales"

# ── Paths S3 de salida ────────────────────────────────────────────────────────
BUCKET_OUT     = os.environ.get("PROMO_S3_BUCKET", "my-promo-bucket")
S3_OUTPUT_DATA = f"s3://{BUCKET_OUT}/m1/data-train"
S3_CODE        = f"s3://{BUCKET_OUT}/m1/code/sourcedir.tar.gz"

# ── Nombre del archivo de salida ──────────────────────────────────────────────
OUTPUT_FILE = "train-set-features.csv"

# ── Período a procesar ────────────────────────────────────────────────────────
FECHA_INICIO = os.environ.get("ETL_FECHA_INICIO", "20240201")
FECHA_FIN    = os.environ.get("ETL_FECHA_FIN",    "20260228")

# ── Ruta de salida (SageMaker > local) ───────────────────────────────────────
C_OUTPUT   = "/opt/ml/processing/output/data"    # path dentro del contenedor
OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DATA_DIR", str(ROOT / "output")))

# ── SageMaker ─────────────────────────────────────────────────────────────────
PIPELINE_NAME = "promo-uplift-etl"
IT_ETL        = "ml.m5.4xlarge"
