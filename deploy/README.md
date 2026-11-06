# Deploy — AWS SageMaker

Pipeline M1 en producción usa dos pipelines registrados en SageMaker:

| Pipeline | Pasos | Schedule |
|---|---|---|
| `promo-uplift-training` | tune → train → cv → explain (1–4) | Mensual o bajo demanda |
| `promo-uplift-scoring` | elasticity + score → staging → dryrun → publish (5–9) | Lunes 02:00 CST |

---

## Empaquetado y ejecución

```bash
# 1. Empaquetar src/ + config.py + requirements.txt → S3
python deploy/deploy_sm.py

# 2. Registrar los dos pipelines en SageMaker
python deploy/pipeline_def.py

# 3. Ejecutar pipeline
python deploy/deploy_sm.py --run
```

`deploy_sm.py` sube el código a `s3://<PROMO_S3_BUCKET>/m1/code/sourcedir.tar.gz`.
`pipeline_def.py` define los steps con `SKLearnProcessor` y los registra en SageMaker Pipelines.

---

## Variables de entorno (`config.py`)

`config.py` detecta si `SM_CHANNEL_TRAIN` está presente y usa paths de container.
Si no existe, usa paths locales (`ROOT/data`, `ROOT/models`, `ROOT/reports/`).

| Variable | Container path | Pasos | Uso |
|---|---|---|---|
| `SM_CHANNEL_TRAIN` | `/opt/ml/processing/input/data` | todos | CSV de features de entrada |
| `SM_MODEL_DIR` | `/opt/ml/processing/models` | 1–4 (write) | tune/train escriben modelos |
| `SM_MODEL_DIR` | `/opt/ml/processing/input/models` | 5–9 (read) | pasos de inferencia leen modelos |
| `SM_OUTPUT_DATA_DIR` | `/opt/ml/processing/output/reports` | 1–8 | cv/, scoring/, elasticity/, shap/ |
| `SM_PUBLISH_DIR` | `/opt/ml/processing/output/publish` | 9 solamente | PROMO_PREDICTIONS/ y PROMO_ELASTICITY_CURVES/ |

El código de los scripts es idéntico en modo local y SageMaker.

---

## Container I/O por step — pipeline de scoring (pasos 5–9)

Cada step corre en un container aislado. SageMaker baja inputs de S3 antes del step y sube outputs a S3 al terminar.

### Paso 5 — `elasticity.py`

| Dirección | Container path | S3 |
|---|---|---|
| Input | `/opt/ml/processing/input/data` | `s3://.../m1/data-train/train-set-features.csv` |
| Input | `/opt/ml/processing/input/models` | `s3://.../m1/models/` |
| Output | `/opt/ml/processing/output/reports/elasticity` | `s3://.../m1/reports/elasticity/` |

### Paso 6 — `score.py`

| Dirección | Container path | S3 |
|---|---|---|
| Input | `/opt/ml/processing/input/data` | `s3://.../m1/data-train/train-set-features.csv` |
| Input | `/opt/ml/processing/input/models` | `s3://.../m1/models/` |
| Output | `/opt/ml/processing/output/reports/scoring` | `s3://.../m1/reports/scoring/` |

### Paso 7 — `staging.py`

| Dirección | Container path | S3 |
|---|---|---|
| Input | `/opt/ml/processing/input/data` | `s3://.../m1/data-train/train-set-features.csv` |
| Input | `/opt/ml/processing/output/reports/scoring` | `s3://.../m1/reports/scoring/` (de paso 6) |
| Output | `/opt/ml/processing/output/reports/scoring` | `s3://.../m1/reports/scoring/` (+ staging_output.parquet) |

### Paso 8 — `dryrun.py`

| Dirección | Container path | S3 |
|---|---|---|
| Input | `/opt/ml/processing/output/reports/scoring` | `s3://.../m1/reports/scoring/` (de paso 7) |
| Input | `/opt/ml/processing/output/reports/elasticity` | `s3://.../m1/reports/elasticity/` |
| Input | `/opt/ml/processing/output/reports/cv` | `s3://.../m1/reports/cv/` |
| Input | `/opt/ml/processing/input/models` | `s3://.../m1/models/` |
| Output | `/opt/ml/processing/output/reports/scoring` | `s3://.../m1/reports/scoring/` (+ dryrun_report.json) |

### Paso 9 — `publish.py` (gate)

| Dirección | Container path | S3 |
|---|---|---|
| Input | `/opt/ml/processing/output/reports/scoring` | `s3://.../m1/reports/scoring/` (dryrun_report.json + staging_output.parquet) |
| Input | `/opt/ml/processing/output/reports/elasticity` | `s3://.../m1/reports/elasticity/` |
| **Output** | `/opt/ml/processing/output/publish/PROMO_PREDICTIONS` | `s3://.../m1/publish/PROMO_PREDICTIONS/` |
| **Output** | `/opt/ml/processing/output/publish/PROMO_ELASTICITY_CURVES` | `s3://.../m1/publish/PROMO_ELASTICITY_CURVES/` |

> **Gate:** Si `dry_run_approved = false` → RuntimeError → step FAILED → SageMaker no sube `/output/publish/` → S3 `publish/` sin cambios → DataOps y Snowflake no reciben datos.

---

## Snowflake vía DataOps

El pipeline **nunca** conecta a Snowflake directamente. El proceso DataOps externo:

1. Lee desde `s3://<PROMO_S3_BUCKET>/m1/publish/`
2. Ejecuta MERGE en Snowflake:
   - `PROMO_PREDICTIONS` — MERGE ON (tienda_id, generico_id, scoring_date)
   - `PROMO_ELASTICITY_CURVES` — DELETE + INSERT por scoring_date

Separación intencional confirmada en alcance MVP (27/05/2026).

---

## Ejecución local equivalente

```bash
# Training
python src/tune.py --n-trials 50
python src/train.py
python src/cv.py
python src/explain.py

# Scoring
python src/elasticity.py --min-events 3
python src/score.py
python src/staging.py
python src/dryrun.py
python src/publish.py
```

En modo local, `config.py` usa `ROOT/data`, `ROOT/models`, `ROOT/reports/` y `ROOT/reports/publish/`.
Ver `deploy/run_local.py` para el script de ejecución completa en local.
