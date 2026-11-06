"""
config.py — M1 Demand Response: Precisión Promocional

Parámetros centrales del pipeline. Edita aquí para propagar cambios
a todos los scripts sin tocar el código de cada uno.

ORDEN DEL PIPELINE:
    [1] tune.py   → busca mejores hiperparámetros (LightGBM + XGBoost) → tune_meta.json
    [2] train.py  → carga tune_meta.json → entrena ambos → BLS → guarda champion
    [3] cv.py     → valida el champion con CV temporal 5-fold
    [4] explain.py → SHAP global
    [5] elasticity.py → curvas de elasticidad
    [6] score.py  → scoring MVP + semáforo
    [7] staging.py → validaciones + schema Snowflake
    [8] dryrun.py  → checklist final → dry_run_approved
    [9] publish.py → sube staging_output.parquet y promo_elasticity_curves.parquet a S3/publish/

RUTAS — env vars para SageMaker (local: derivadas de ROOT):
    SM_CHANNEL_TRAIN    → DATA_DIR    (datos de entrada)
    SM_MODEL_DIR        → MODELS_DIR  (artefactos del modelo)
    SM_OUTPUT_DATA_DIR  → base de REPORTS_* (outputs del pipeline)
    SM_PUBLISH_DIR      → PUBLISH_DIR (destino final para DataOps/Snowflake)

Uso en scripts src/:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import ROOT, TRAIN_DATA_PATH, MODELS_DIR, REPORTS_SCORING, ...
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Modelo champion ───────────────────────────────────────────────────────────
# Actualiza este valor con el ganador del BLS scorecard reportado por train.py.
# Opciones típicas:
#   "lgbm_base.pkl"  — LightGBM (train.py con params de tune_meta.json)
#   "xgb_base.pkl"   — XGBoost  (train.py con params de tune_meta.json) ← actual
CHAMPION_MODEL = "xgb_base.pkl"

# ── Rutas principales — override con env vars en SageMaker ───────────────────
# Local: ROOT/data, ROOT/models, ROOT/reports/...
# SageMaker: SM_* montados automáticamente por el job definition del pipeline
DATA_DIR          = Path(os.environ.get("SM_CHANNEL_TRAIN",   str(ROOT / "data")))
MODELS_DIR        = Path(os.environ.get("SM_MODEL_DIR",       str(ROOT / "models")))
# En SageMaker, train.py escribe en MODELS_DIR (canal de salida) pero necesita leer el
# tune_meta.json del step anterior desde un canal de entrada distinto (S3 no permite que
# un ProcessingStep use el mismo path como input y output). En local ambos caen en
# ROOT/models, así que no hay diferencia fuera del pipeline de SageMaker.
INPUT_MODELS_DIR  = Path(os.environ.get("SM_INPUT_MODELS_DIR", str(ROOT / "models")))
_REPORTS_BASE     = Path(os.environ.get("SM_OUTPUT_DATA_DIR", str(ROOT / "reports")))

REPORTS_CV         = _REPORTS_BASE  / "cv"
REPORTS_SCORING    = _REPORTS_BASE  / "scoring"
REPORTS_ELASTICITY = _REPORTS_BASE  / "elasticity"
REPORTS_SHAP       = _REPORTS_BASE  / "shap"

# Destino de publicación → DataOps lo recoge y sincroniza a Snowflake
PUBLISH_DIR        = Path(os.environ.get("SM_PUBLISH_DIR", str(ROOT / "reports" / "publish")))

# ── Nombres de datasets ───────────────────────────────────────────────────────
TRAIN_DATASET_FILE  = "train-set-features.csv"   # dataset principal (output del SQL Athena)
TRAIN_DEPURED_FILE  = "train-set-depured.csv"     # dataset crudo sin features SQL (validación local)

TRAIN_DATA_PATH    = DATA_DIR / TRAIN_DATASET_FILE
TRAIN_DEPURED_PATH = DATA_DIR / TRAIN_DEPURED_FILE

# ── Lectura de CSV ────────────────────────────────────────────────────────────
CSV_SEP = ","                  # separador del CSV de features ("," o ";")

# ── Split temporal ────────────────────────────────────────────────────────────
VAL_CUTOFF = 20251201          # YYYYMMDD — corte train < val

# ── CV temporal ───────────────────────────────────────────────────────────────
CV_N_SPLITS = 5
CV_GAP_DAYS = 28               # días de gap entre fold train y val

# ── Umbrales de calidad — BBP v1.4 §Criterios de Aceptación ──────────────────
# WMAPE (decimal 0–1)
WMAPE_OBJETIVO = 0.12          # ≤ 12% → objetivo deseable
WMAPE_GREEN    = 0.25          # ≤ 25% → gate verde (> 25% = trigger de reentrenamiento)
WMAPE_GATE     = 0.35          # ≤ 35% → gate superado (continuar con aviso)

# Lift Accuracy
LIFT_ACC_OBJETIVO = 0.85       # ≥ 85% → objetivo deseable
LIFT_ACC_MIN      = 0.70       # ≥ 70% → mínimo aceptable
LIFT_ACC_TRIGGER  = 0.60       # < 60% → trigger de reentrenamiento

# Bias relativo (absoluto, decimal)
BIAS_MAX     = 0.10            # ±10% → máximo aceptable
BIAS_OBJETIVO = 0.05           # ±5%  → objetivo deseable
BIAS_TRIGGER  = 0.15           # > ±15% → trigger de reentrenamiento (3 semanas consecutivas)

# R²
R2_MIN      = 0.60             # ≥ 0.60 → mínimo aceptable
R2_OBJETIVO = 0.75             # ≥ 0.75 → objetivo deseable
R2_TRIGGER  = 0.50             # < 0.50 → trigger de reentrenamiento

# Elasticidad
ELAST_MONO_MIN = 0.90          # ≥ 90% curvas monótonas tras isotonic regression

# Staging (validaciones bloqueantes)
STAGING_AVOID_MAX  = 0.95      # > 95% AVOID → aviso bloqueante
STAGING_EXTRAP_MAX = 0.20      # > 20% extrapolados → aviso

# ── Semáforo de scoring ───────────────────────────────────────────────────────
UPLIFT_GO       = 15.0         # uplift_pct ≥ 15% → GO
UPLIFT_CONSIDER =  5.0         # uplift_pct ≥  5% → CONSIDER (< 5% → AVOID)

# ── Curvas de elasticidad ─────────────────────────────────────────────────────
ELASTICITY_MIN_EVENTS = 3      # eventos mínimos por mecánica/combo

# ── Entrenamiento ─────────────────────────────────────────────────────────────
TARGET            = "unidades_vendidas"
N_ESTIMATORS      = 2000       # árboles máximos (early stopping detiene antes si converge)
WINSORIZE_CAP_PCT = 99.5       # percentil de winsorización del target
