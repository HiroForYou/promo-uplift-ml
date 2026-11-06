"""
etl.py — Script ETL principal (promo-uplift-etl) v3

Flujo por mes:
    - Si la partición ventas_diarias/YYYYMM/ ya existe en S3 → se reutiliza
    - Si NO existe → corre los 19 CTEs y guarda ventas_diarias en S3

Flujo final (todos los meses):
    1. Concatena todas las particiones ventas_diarias
    2. ventas_con_features → train_set_final
    3. Escribe train-set-features.csv en s3://<BUCKET>/m1/data-train/
"""

import io
import os
import sys
import boto3
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    S3_TLOGF, S3_TLOGF_EXT, S3_MARD,
    S3_A155, S3_A018, S3_KONP, S3_PV_ECC, S3_PC_ECC,
    S3_CAL, S3_DIM_SUC, S3_DIM_MAT,
    BUCKET_OUT, OUTPUT_DIR, OUTPUT_FILE,
    FECHA_INICIO, FECHA_FIN, REGION,
)
from ctes import (
    ventas_base_dates,
    promo_lineas_dates,
    cupon_lineas_dates,
    promo_nombre_ext_dates,
    promo_consolidada,
    dim_mat_clean,
    dim_suc_clean,
    fechas_ventas_unicas,
    precio_venta_dedup,
    precio_venta,
    precio_compra_dedup,
    precio_compra,
    inventario_mensual,
    calendario_promo_activo,
    calendario_por_dia,
    ventas_enriquecida,
    ventas_diaria,
    ventas_con_features,
    train_set_final,
)

# ── Rutas S3 ──────────────────────────────────────────────────────────────────
S3_VENTAS_DIARIAS_PREFIX = "m1/data-etl/ventas_diarias"
S3_OUTPUT_KEY            = f"m1/data-train/{OUTPUT_FILE}"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / OUTPUT_FILE


# ── Helpers ───────────────────────────────────────────────────────────────────

def _anomes_range(fecha_inicio: str, fecha_fin: str) -> list[str]:
    y0, m0 = int(fecha_inicio[:4]), int(fecha_inicio[4:6])
    y1, m1 = int(fecha_fin[:4]),    int(fecha_fin[4:6])
    meses = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        meses.append(f"{y}{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return meses


def _mes_a_fechas(mes: str) -> tuple[str, str]:
    """Convierte YYYYMM al primer y último día del mes como YYYYMMDD."""
    import calendar
    y, m = int(mes[:4]), int(mes[4:6])
    ultimo = calendar.monthrange(y, m)[1]
    return f"{y}{m:02d}01", f"{y}{m:02d}{ultimo:02d}"


def _existe_particion(s3_client, bucket: str, mes: str) -> bool:
    key = f"{S3_VENTAS_DIARIAS_PREFIX}/{mes}/ventas_diarias.csv"
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def _guardar_particion_s3(s3_client, bucket: str, mes: str, df: pd.DataFrame) -> None:
    key = f"{S3_VENTAS_DIARIAS_PREFIX}/{mes}/ventas_diarias.csv"
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    buf.seek(0)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    print(f"  Guardado: s3://{bucket}/{key}  ({len(df):,} filas)")


def _leer_particion_s3(s3_client, bucket: str, mes: str) -> pd.DataFrame:
    key = f"{S3_VENTAS_DIARIAS_PREFIX}/{mes}/ventas_diarias.csv"
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def _procesar_mes(mes: str) -> pd.DataFrame:
    """Corre los 17 CTEs (ventas_base → ventas_diaria) para un mes y devuelve el DataFrame."""
    fecha_inicio, fecha_fin = _mes_a_fechas(mes)
    print(f"  Procesando {mes}: {fecha_inicio} → {fecha_fin}")

    df_ventas_base       = ventas_base_dates(S3_TLOGF, fecha_inicio, fecha_fin)
    df_promo_lineas      = promo_lineas_dates(S3_TLOGF, fecha_inicio, fecha_fin)
    df_cupon_lineas      = cupon_lineas_dates(S3_TLOGF, fecha_inicio, fecha_fin)
    df_promo_nombre_ext  = promo_nombre_ext_dates(S3_TLOGF_EXT, fecha_inicio, fecha_fin)
    df_promo_consolidada = promo_consolidada(df_promo_lineas, df_cupon_lineas, df_promo_nombre_ext)
    del df_promo_lineas, df_cupon_lineas, df_promo_nombre_ext

    df_dim_mat_clean      = dim_mat_clean(S3_DIM_MAT)
    df_dim_suc_clean      = dim_suc_clean(S3_DIM_SUC)
    df_fechas_unicas      = fechas_ventas_unicas(df_ventas_base)

    df_pv_dedup  = precio_venta_dedup(S3_A155, S3_KONP, S3_PV_ECC)
    df_pv        = precio_venta(df_pv_dedup, df_fechas_unicas)
    df_pc_dedup  = precio_compra_dedup(S3_A018, S3_KONP, S3_PC_ECC)
    df_pc        = precio_compra(df_pc_dedup, df_fechas_unicas)
    del df_pv_dedup, df_pc_dedup, df_fechas_unicas

    df_inventario        = inventario_mensual(S3_MARD, S3_DIM_MAT)
    df_cal_activo        = calendario_promo_activo(S3_CAL)
    df_cal_dia           = calendario_por_dia(df_cal_activo, fechas_ventas_unicas(df_ventas_base))
    del df_cal_activo

    df_enriquecida = ventas_enriquecida(
        df_ventas_base, df_promo_consolidada, df_dim_mat_clean, df_dim_suc_clean,
        df_pv, df_pc, df_inventario, df_cal_dia,
    )
    del (df_ventas_base, df_promo_consolidada, df_dim_mat_clean, df_dim_suc_clean,
         df_pv, df_pc, df_inventario, df_cal_dia)

    df_diaria = ventas_diaria(df_enriquecida)
    del df_enriquecida

    print(f"  {mes}: {len(df_diaria):,} filas × {len(df_diaria.columns)} columnas")
    return df_diaria


# ── Pipeline ──────────────────────────────────────────────────────────────────

print(f"Período: {FECHA_INICIO} → {FECHA_FIN}")
print(f"Salida : {output_path}\n")

meses = _anomes_range(FECHA_INICIO, FECHA_FIN)
print(f"Meses a procesar ({len(meses)}): {meses}\n")

s3 = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION", REGION))

# 1. Por cada mes: reutilizar si existe, procesar si no existe
chunks = []
for mes in meses:
    if _existe_particion(s3, BUCKET_OUT, mes):
        print(f"[{mes}] Partición encontrada en S3 — leyendo...")
        df_mes = _leer_particion_s3(s3, BUCKET_OUT, mes)
        print(f"  {mes}: {len(df_mes):,} filas × {len(df_mes.columns)} columnas")
    else:
        print(f"[{mes}] Partición NO encontrada — ejecutando CTEs...")
        df_mes = _procesar_mes(mes)
        print(f"[{mes}] Guardando partición en S3...")
        _guardar_particion_s3(s3, BUCKET_OUT, mes, df_mes)

    chunks.append(df_mes)
    del df_mes

# 2. Concatenar todas las particiones
print(f"\nConcatenando {len(chunks)} meses...")
df_ventas_diaria = pd.concat(chunks, ignore_index=True)
del chunks
print(f"  Total: {len(df_ventas_diaria):,} filas × {len(df_ventas_diaria.columns)} columnas")

# 3. ventas_con_features
print("\nIniciando CTE: ventas_con_features...")
df_ventas_con_features = ventas_con_features(df_ventas_diaria)
del df_ventas_diaria
print(f"  ventas_con_features: {len(df_ventas_con_features):,} filas × {len(df_ventas_con_features.columns)} columnas")

# 4. train_set_final
print("\nIniciando CTE: train_set_final...")
df_train_set_final = train_set_final(df_ventas_con_features)
del df_ventas_con_features
print(f"  train_set_final: {len(df_train_set_final):,} filas × {len(df_train_set_final.columns)} columnas")

# 5. Guardar CSV local
print(f"\nGuardando output final...")
df_train_set_final.to_csv(output_path, index=False, encoding="utf-8")
print(f"  → {output_path}  ({len(df_train_set_final):,} filas × {len(df_train_set_final.columns)} columnas)")
del df_train_set_final

# 6. Subir a S3
print(f"\nSubiendo a S3...")
s3.upload_file(str(output_path), BUCKET_OUT, S3_OUTPUT_KEY)
print(f"  → s3://{BUCKET_OUT}/{S3_OUTPUT_KEY}")

print("\n✅ Pipeline completado.")

# ── Auto-trigger Training ─────────────────────────────────────────────────────
_auto = os.environ.get("AUTO_TRIGGER_TRAINING", "false").lower() == "true"
if _auto:
    _sm   = boto3.client("sagemaker", region_name=os.environ.get("AWS_DEFAULT_REGION", REGION))
    _exec = _sm.start_pipeline_execution(
        PipelineName="promo-uplift-training",
        PipelineExecutionDisplayName="auto-trigger-post-etl",
    )
    print(f"\n  AUTO_TRIGGER_TRAINING activo — promo-uplift-training disparado.")
    print(f"  Execution ARN: {_exec['PipelineExecutionArn']}")
