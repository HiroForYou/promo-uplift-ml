# %% [markdown]
# ### promo-uplift-ml · Precisión Promocional
# src/features.py — Pipeline de Feature Engineering
# 
# Semana 3:
#   Tarea 3.1  Features de demanda baseline (categoría 2 del alcance)
#   Tarea 3.2  Features de historial promocional (categoría 4)
#   Tarea 3.3  Features de estacionalidad y ciclo de vida (categorías 3 y 7)
#   Extra      Lags y rolling window sobre el target
# 
# RELACIÓN CON train-set-features.sql
#   El SQL (Bloque F) computa en Athena: Tarea 3.1 + 3.2 + pct_descuento_calc.
#   Este módulo computa siempre: Tarea 3.3 + lags.
#   Las Tareas 3.1/3.2 solo se recomputan en Python si NO vienen en el input
#   (por ejemplo, cuando el input es train-set-depured.csv en lugar de
#   train-set-features.csv).
# 
#   Flujo producción (Semanas 4-5):
#     train-set-features.csv  →  build_features()  →  add 3.3 + lags  →  train.py
# 
#   Flujo local / desarrollo:
#     train-set-depured.csv   →  build_features()  →  add 3.1+3.2+3.3+lags
# 
#   build_features() detecta automáticamente qué features ya están presentes
#   y solo computa las que faltan.
# 
# Prerequisito: DataFrame ya a granularidad tienda × genérico × día.
# Uso como módulo:
#     from src.features import build_features, FEATURE_COLS, CAT_COLS
# 
# Uso como script de validación local:
#     python src/features.py                                    # usa train-set-depured.csv
#     python src/features.py --data data/train-set-features.csv # usa output del SQL

# %%
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ── Constantes ────────────────────────────────────────────────────────────────
GROUP_COLS  = ["tienda_id", "generico_id"]
TARGET_COL  = "unidades_vendidas"

FEATURE_COLS: list[str] = [
    # Precio y descuento — transaccional (0 % nulos, siempre disponible)
    "precio_efectivo",          # monto_venta / unidades_vendidas
    "costo_unitario",           # costo_linea / unidades_vendidas
    "pct_descuento_calc",
    "descuento_total",
    # Precio y descuento — catálogo a155/a018 (~44 % nulos en producción)
    "precio_venta",
    "precio_venta_bruto",
    "precio_compra",
    "margin_bruto",
    "pct_margen",
    "precio_relativo_categoria",
    # Promoción
    "tiene_promo_flag",
    "en_calendario_promo",
    "n_promos_activas_tienda",
    "tienda_promo_sku_sin_descuento",
    # Amplitud de surtido (nuevos en Semana 1)
    "n_tallas_vendidas",
    "n_colores_vendidos",
    # Baseline — Tarea 3.1
    "avg_units_4w",
    "avg_units_8w",
    "avg_units_12w",
    "coef_variacion",
    "flag_baseline_insuficiente",
    # Historial promo — Tarea 3.2
    "frecuencia_promo_90d",
    "dias_desde_ultima_promo",
    "avg_discount_historico",
    "num_promos_previas",
    # Mecánica promocional — Tarea nueva (Fase 1)
    "tipo_mecanica",
    # Inventario
    "stock_disponible",
    # Estacionalidad — Tarea 3.3
    "dia_semana",
    "semana_mes",
    "mes",
    "es_quincena",
    "es_fin_semana",
    "es_inicio_mes",
    # Ciclo de vida — Tarea 3.3
    "semanas_en_mercado",
    "product_lifecycle",
    # Lags y rolling window
    "lag_1d",
    "lag_7d",
    "lag_14d",
    "lag_28d",
    "rolling_7d_mean",
    "rolling_28d_mean",
    "rolling_7d_std",
    # Tienda
    "banner",
    "region_desc",
    "formato",
    # Producto
    "marca_desc",
    "division_desc",
    "deporte_desc",
    "genero_desc",
]

CAT_COLS: list[str] = [
    "banner",
    "region_desc",
    "formato",
    "marca_desc",
    "division_desc",
    "deporte_desc",
    "genero_desc",
    "product_lifecycle",
    "tipo_mecanica",
]

# %% [markdown]
# #### Helpers internos

# %%

def _col(df: pd.DataFrame, name: str, default: int | float = 0) -> pd.Series:
    """Devuelve la columna si existe, si no una serie de `default`."""
    return df[name] if name in df.columns else pd.Series(default, index=df.index)

# %% [markdown]
# #### Mecánica promocional

# %%

_MECANICA_PATTERNS: list[tuple[str, str]] = [
    ("2X1",           r"2X1|2 X 1|DOS\s*POR\s*UNO"),
    ("BUNDLE",        r"BUNDLE|KIT|PAQUETE|COMBO"),
    ("CASHBACK",      r"CASHBACK|CASH\s*BACK"),
    ("LIQUIDACION",   r"LIQUIDACI"),
    ("REGALO_COMPRA", r"REGALO|GRATIS|FREE"),
]


def derive_tipo_mecanica(nombre_promo: pd.Series) -> pd.Series:
    """
    Mapea nombre_promo a 7 categorías de mecánica.

    Categorías: DESCUENTO_PCT (default), 2X1, BUNDLE, CASHBACK,
                LIQUIDACION, REGALO_COMPRA, SIN_PROMO.

    Patrones insensibles a mayúsculas; orden de evaluación de arriba abajo
    (primero coincidente gana). Actualizar _MECANICA_PATTERNS si los nombres
    de promo en TLOGF cambian.
    """
    s      = nombre_promo.fillna("").str.upper()
    result = pd.Series("DESCUENTO_PCT", index=nombre_promo.index, dtype=object)
    for cat, pat in _MECANICA_PATTERNS:
        result[s.str.contains(pat, regex=True, na=False)] = cat
    result[nombre_promo.isna() | (nombre_promo.fillna("").str.strip() == "")] = "SIN_PROMO"
    return result.astype("category")


def build_mecanica_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categoría 8 del alcance: tipo de mecánica promocional como feature categórico.

    Columna generada:
        tipo_mecanica  — DESCUENTO_PCT | 2X1 | BUNDLE | CASHBACK |
                         LIQUIDACION | REGALO_COMPRA | SIN_PROMO

    Derivado de nombre_promo cuando está disponible; si no, infiere desde
    tiene_promo_flag (SIN_PROMO = 0, DESCUENTO_PCT = 1).

    ADVERTENCIA DE DESBALANCEO (BBP §13.tipo_mecanica):
        El historial del retailer está dominado por DESCUENTO_PCT. Si una
        mecánica (p. ej. 2X1, BUNDLE) aparece en < 5% de las filas de
        entrenamiento, el modelo tendrá evidencia escasa para diferenciar su
        curva de elasticidad de DESCUENTO_PCT. En ese caso:
          - Las curvas generadas por elasticity.py para esa mecánica serán
            estadísticamente débiles y deben interpretarse con cautela.
          - Verificar en reports/elasticity/elasticity_summary.csv el n_combos
            por mecánica antes de presentar resultados al cliente.
          - Con ≥ 500 combos×eventos por categoría la diferenciación suele ser
            confiable; con < 100 las curvas son orientativas, no prescriptivas.
    """
    if "nombre_promo" in df.columns:
        df["tipo_mecanica"] = derive_tipo_mecanica(df["nombre_promo"])
    else:
        tipo = np.where(df["tiene_promo_flag"].fillna(0).astype(int) == 1,
                        "DESCUENTO_PCT", "SIN_PROMO")
        df["tipo_mecanica"] = pd.Categorical(tipo)
    return df

# %% [markdown]
# #### Tarea 3.1 — Features de demanda baseline

# %%

def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categoría 2 del alcance: promedios de demanda sobre días sin promo.

    Columnas generadas:
        avg_units_4w        — media 28 días previos (sólo días sin promo)
        avg_units_8w        — media 56 días previos (sólo días sin promo)
        avg_units_12w       — media 84 días previos (sólo días sin promo)
        coef_variacion      — std / mean de unidades (28d sin promo)
        flag_baseline_ins.  — 1 si < 14 días sin promo en ventana de 28 filas previas

    Regla: "sólo días sin promo" = tiene_promo_flag == 0 AND en_calendario_promo == 0
    Las ventanas son en número de FILAS (no de días calendario), equivalente al
    ORDER BY fecha_venta ROWS BETWEEN N PRECEDING AND 1 PRECEDING de Athena.
    """
    en_cal = _col(df, "en_calendario_promo", default=0)
    sin_promo_mask = (df["tiene_promo_flag"] == 0) & (en_cal == 0)

    # Unidades mascaradas (NaN en días de promo)
    df["_u_sp"] = df[TARGET_COL].where(sin_promo_mask)

    g = df.groupby(GROUP_COLS, sort=False)

    for rows, name in [(28, "4w"), (56, "8w"), (84, "12w")]:
        df[f"avg_units_{name}"] = g["_u_sp"].transform(
            lambda x, r=rows: x.shift(1).rolling(r, min_periods=3).mean()
        )

    # Coef. de variación: std/mean (28d, días sin promo)
    rolling_std  = g["_u_sp"].transform(lambda x: x.shift(1).rolling(28, min_periods=3).std())
    rolling_mean = g["_u_sp"].transform(lambda x: x.shift(1).rolling(28, min_periods=3).mean())
    df["coef_variacion"] = rolling_std / rolling_mean.replace(0.0, np.nan)

    # Flag de baseline insuficiente (< 14 días sin promo en ventana 28)
    df["_is_sp"] = sin_promo_mask.astype(float)
    dias_sp_28 = g["_is_sp"].transform(
        lambda x: x.shift(1).rolling(28, min_periods=0).sum()
    )
    df["flag_baseline_insuficiente"] = (dias_sp_28 < 14).astype("int8")

    df = df.drop(columns=["_u_sp", "_is_sp"])
    return df

# %% [markdown]
# #### Tarea 3.2 — Features de historial promocional

# %%

def build_promo_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categoría 4 del alcance: historial promocional por combinación.

    Columnas generadas:
        frecuencia_promo_90d    — # promos en las 90 filas previas
        dias_desde_ultima_promo — días calendario desde la última promo
        avg_discount_historico  — descuento % medio histórico (expanding, solo promos)
        num_promos_previas      — total acumulado de promos anteriores

    Nota: pct_descuento = descuento_total / (monto_venta + descuento_total)
          No requiere precio_compra.
    """
    g = df.groupby(GROUP_COLS, sort=False)

    # frecuencia_promo_90d
    df["frecuencia_promo_90d"] = g["tiene_promo_flag"].transform(
        lambda x: x.shift(1).rolling(90, min_periods=0).sum()
    )

    # dias_desde_ultima_promo
    df["_promo_fecha"] = df["fecha_venta"].where(df["tiene_promo_flag"] == 1)
    df["_ultima_promo"] = g["_promo_fecha"].transform(lambda x: x.ffill().shift(1))
    df["dias_desde_ultima_promo"] = (df["fecha_venta"] - df["_ultima_promo"]).dt.days

    # pct_descuento para historial (no requiere precio_compra)
    denominador = (df["monto_venta"] + df["descuento_total"]).replace(0.0, np.nan)
    df["_pct_desc"] = df["descuento_total"] / denominador
    df["_pct_desc_promo"] = df["_pct_desc"].where(df["tiene_promo_flag"] == 1)

    # avg_discount_historico (expanding, sólo filas con promo)
    df["avg_discount_historico"] = g["_pct_desc_promo"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # num_promos_previas (expanding sum)
    df["num_promos_previas"] = g["tiene_promo_flag"].transform(
        lambda x: x.shift(1).expanding().sum()
    )

    df = df.drop(columns=["_promo_fecha", "_ultima_promo", "_pct_desc", "_pct_desc_promo"])
    return df

# %% [markdown]
# #### Tarea 3.3 — Estacionalidad, ciclo de vida y precio relativo

# %%

def build_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorías 3 y 7 del alcance: estacionalidad, ciclo de vida, precio relativo.

    Columnas generadas:
        dia_semana, semana_mes, mes, es_quincena, es_fin_semana, es_inicio_mes
        semanas_en_mercado, product_lifecycle
        precio_relativo_categoria
    """
    fv = df["fecha_venta"]

    df["dia_semana"]    = fv.dt.dayofweek           # 0 = lunes
    df["semana_mes"]    = fv.dt.day // 7 + 1
    df["mes"]           = fv.dt.month
    df["es_quincena"]   = fv.dt.day.between(13, 17).astype("int8")
    df["es_fin_semana"] = (fv.dt.dayofweek >= 5).astype("int8")
    df["es_inicio_mes"] = (fv.dt.day <= 3).astype("int8")

    # Ciclo de vida del genérico en la tienda
    primera_venta = df.groupby(GROUP_COLS)["fecha_venta"].transform("min")
    df["semanas_en_mercado"] = ((fv - primera_venta).dt.days / 7).astype(int)
    df["product_lifecycle"] = pd.cut(
        df["semanas_en_mercado"],
        bins=[-1, 8, 20, 36, 9_999],
        labels=["NEW", "CORE", "AGING", "CLEARANCE"],
    ).astype(str)

    # Precio relativo a la categoría (division × deporte × fecha)
    if "precio_venta" in df.columns and "division_desc" in df.columns:
        cat_keys = ["division_desc", "deporte_desc", "fecha_venta"]
        # Sólo las columnas que existen
        cat_keys = [c for c in cat_keys if c in df.columns]
        if len(cat_keys) >= 2:
            avg_cat = df.groupby(cat_keys)["precio_venta"].transform("mean")
            df["precio_relativo_categoria"] = df["precio_venta"] / avg_cat.replace(0.0, np.nan)
        else:
            df["precio_relativo_categoria"] = np.nan
    else:
        df["precio_relativo_categoria"] = np.nan

    return df

# %% [markdown]
# #### Lags y rolling window sobre el target

# %%

def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lags y rolling window sobre unidades_vendidas.

    Columnas generadas:
        lag_1d, lag_7d, lag_14d, lag_28d
        rolling_7d_mean, rolling_28d_mean, rolling_7d_std

    Nota: shift(1) garantiza que nunca se filtra info del día actual (no leakage).
    Implementación vectorizada: groupby().shift() y groupby().rolling() usan
    Cython internamente (~10x más rápido que transform(lambda) por grupo).
    """
    n_group_levels = len(GROUP_COLS)
    g = df.groupby(GROUP_COLS, sort=False)

    # Lags: groupby().shift() es C/Cython — no Python por grupo
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}d"] = g[TARGET_COL].shift(lag)

    # Rolling: shift primero (vectorizado), luego rolling dentro de grupos
    shifted = g[TARGET_COL].shift(1)
    df["_s1"] = shifted

    g2 = df.groupby(GROUP_COLS, sort=False)["_s1"]
    df["rolling_7d_mean"]  = (
        g2.rolling(7,  min_periods=3).mean()
        .droplevel(list(range(n_group_levels)))
    )
    df["rolling_28d_mean"] = (
        g2.rolling(28, min_periods=7).mean()
        .droplevel(list(range(n_group_levels)))
    )
    df["rolling_7d_std"]   = (
        g2.rolling(7,  min_periods=3).std()
        .droplevel(list(range(n_group_levels)))
    )

    df = df.drop(columns=["_s1"])
    return df

# %% [markdown]
# #### Features de precio

# %%

def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features de precio y descuento.

    Columnas generadas (siempre, 0 % nulos):
        pct_descuento_calc  — descuento_total / (monto_venta + descuento_total)
        precio_efectivo     — monto_venta / unidades_vendidas  (precio real cobrado)
        costo_unitario      — costo_linea / unidades_vendidas  (proxy de costo)

    Columnas generadas cuando precio_compra está disponible (100% en dataset v2):
        margin_bruto        — precio_venta - precio_compra
        pct_margen          — margin_bruto / precio_venta

    Nota: dataset v2 (jun-2026) tiene 100% cobertura de precio_venta (a155+ecc),
    precio_venta_bruto, y precio_compra (a018+ecc). precio_efectivo se mantiene
    como feature principal (transaccional, siempre disponible en producción).
    """
    units = df["unidades_vendidas"].replace(0, np.nan)

    # Precio efectivo (real, siempre disponible)
    df["precio_efectivo"] = df["monto_venta"] / units
    df["costo_unitario"]  = df["costo_linea"]  / units

    # Descuento %
    denom = (df["monto_venta"] + df["descuento_total"]).replace(0.0, np.nan)
    df["pct_descuento_calc"] = df["descuento_total"] / denom

    # Margen sobre precio catálogo (nulo cuando precio_compra / precio_venta faltan)
    if "precio_compra" in df.columns:
        df["margin_bruto"] = df["precio_venta"] - df["precio_compra"]
        df["pct_margen"]   = df["margin_bruto"] / df["precio_venta"].replace(0, np.nan)
    else:
        df["margin_bruto"] = np.nan
        df["pct_margen"]   = np.nan

    return df

# %% [markdown]
# #### Sentinel y pipeline principal

# %%

# Si estas columnas ya están en el DataFrame de input, se asume que vienen
# del Bloque F de train-set-features.sql y NO se recomputan en Python.
_SQL_BASELINE_SENTINEL  = "avg_units_4w"          # Tarea 3.1
_SQL_PROMO_SENTINEL     = "frecuencia_promo_90d"   # Tarea 3.2
_SQL_PRICE_SENTINEL     = "pct_descuento_calc"     # Precio
_SQL_MECANICA_SENTINEL  = "tipo_mecanica"          # Mecánica (siempre generado en Python)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de feature engineering — compatible con ambos inputs:

      Input A: train-set-depured.csv (datos sin features SQL)
        → computa Tarea 3.1 + 3.2 + precio + 3.3 + lags

      Input B: train-set-features.csv (output del SQL con Bloque F)
        → detecta que 3.1/3.2/precio ya están → solo añade 3.3 + lags

    Detección automática: si la columna sentinel de cada bloque ya existe
    en el DataFrame, ese bloque se saltea sin tocar los valores del SQL.

    Orden cuando todo falta (Input A):
      1. Parsear fecha_venta  2. Ordenar  3. Precio
      4. Baseline (3.1)  5. Historial promo (3.2)
      6. Estacionalidad (3.3)  7. Lags

    El DataFrame original NO se modifica (se trabaja en una copia).
    """
    df = df.copy()

    # 1. Parsear fecha si viene como int YYYYMMDD o string
    if not pd.api.types.is_datetime64_any_dtype(df["fecha_venta"]):
        df["fecha_venta"] = pd.to_datetime(
            df["fecha_venta"].astype(str), format="%Y%m%d"
        )

    # 2. Orden obligatorio para que shift/rolling sean correctos
    df = df.sort_values(GROUP_COLS + ["fecha_venta"]).reset_index(drop=True)

    # Excluir VENTA DE BOLSAS — fuera del alcance del negocio
    if "division_desc" in df.columns:
        df = df[df["division_desc"] != "VENTA DE BOLSAS"].reset_index(drop=True)

    # 3a. Precio transaccional — SIEMPRE. Dataset v2 tiene 100% cobertura de precios
    #     catálogo, pero precio_efectivo se mantiene porque es el precio real cobrado.
    if "precio_efectivo" not in df.columns:
        df["precio_efectivo"] = df["monto_venta"] / df["unidades_vendidas"].replace(0, np.nan)
    if "costo_unitario" not in df.columns:
        df["costo_unitario"] = df["costo_linea"] / df["unidades_vendidas"].replace(0, np.nan)

    # 3b. Precio catálogo (si no viene del SQL)
    if _SQL_PRICE_SENTINEL not in df.columns:
        df = build_price_features(df)

    # 4. Baseline (Tarea 3.1) — solo si no viene del SQL
    if _SQL_BASELINE_SENTINEL not in df.columns:
        df = build_baseline_features(df)

    # 5. Historial promo (Tarea 3.2) — solo si no viene del SQL
    if _SQL_PROMO_SENTINEL not in df.columns:
        df = build_promo_history_features(df)

    # 6. Estacionalidad y ciclo de vida (Tarea 3.3) — siempre: no está en el SQL
    df = build_seasonality_features(df)

    # 7. Lags y rolling — siempre: no están en el SQL
    df = build_lag_features(df)

    # 8. Mecánica promocional — siempre: derivado de nombre_promo en Python
    if _SQL_MECANICA_SENTINEL not in df.columns:
        df = build_mecanica_feature(df)

    return df

# %% [markdown]
# #### Validación local

# %%

def _print_feature_report(df_raw: pd.DataFrame, df_feat: pd.DataFrame) -> None:
    """Imprime resumen de nulos y valores de las features generadas."""
    from src._console import reconfigure_stdout
    reconfigure_stdout()

    nuevas = [c for c in FEATURE_COLS if c not in df_raw.columns and c in df_feat.columns]
    preexistentes = [c for c in FEATURE_COLS if c in df_raw.columns and c in df_feat.columns]
    ausentes = [c for c in FEATURE_COLS if c not in df_feat.columns]

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Feature Engineering — Reporte de validación local")
    print(f"  Filas: {len(df_feat):,}  |  Columnas originales: {len(df_raw.columns)}")
    print(f"  Columnas después de build_features: {len(df_feat.columns)}")
    print(sep)

    print(f"\n  FEATURES NUEVAS GENERADAS ({len(nuevas)}):")
    print(f"  {'Feature':<35s} {'%_nulos':>8s}  {'media / muestra':>20s}")
    print("  " + "-" * 66)
    for col in nuevas:
        s = df_feat[col]
        pct_null = s.isna().mean()
        if pd.api.types.is_numeric_dtype(s):
            sample = f"mean={s.mean():.3f}  std={s.std():.3f}"
        else:
            top = s.dropna().value_counts().index[0] if s.notna().any() else "—"
            sample = f"top='{top}'"
        flag = " ⚠️" if pct_null > 0.5 else ""
        print(f"  {col:<35s} {pct_null:>7.1%}  {sample:>20s}{flag}")

    if preexistentes:
        print(f"\n  FEATURES PREEXISTENTES EN FEATURE_COLS ({len(preexistentes)}): OK")

    if ausentes:
        print(f"\n  ⚠️  AUSENTES (no en df ni generadas): {ausentes}")

    # Nota contextual según tamaño del dataset
    n = len(df_feat)
    if n < 5_000:
        print(f"\n  NOTA: Con {n:,} filas (muestra pequeña, ~1 fila/combo) los lags")
        print(f"  y ventanas producirán ~100 % de nulos. Es esperado.")
        print(f"  Validar con el dataset completo (train-set-features.csv).\n")
    else:
        print(f"\n  NOTA: lag_28d y lag_14d tendrán nulos altos en ventas diarias")
        print(f"  dispersas (no todos los dias hay venta). LightGBM/XGBoost")
        print(f"  manejan NaN nativamente — no es necesario imputar.\n")


def main() -> None:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    except NameError:
        _cwd = Path().resolve()
        sys.path.insert(0, str(_cwd if (_cwd / "config.py").exists() else _cwd.parent))
    from src._console import reconfigure_stdout
    reconfigure_stdout()
    from config import TRAIN_DEPURED_PATH
    default_data = TRAIN_DEPURED_PATH

    parser = argparse.ArgumentParser(
        description=(
            "Valida build_features() localmente.\n"
            "  --data train-set-depured.csv  → computa 3.1+3.2+3.3+lags\n"
            "  --data train-set-features.csv → detecta 3.1+3.2 del SQL, añade 3.3+lags"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=default_data,
                        help="CSV de input (depured o features)")
    args, _ = parser.parse_known_args()

    from config import CSV_SEP
    print(f"Cargando: {args.data}")
    df_raw = pd.read_csv(
        args.data,
        sep=CSV_SEP,
        dtype={"tienda_id": str, "generico_id": str},
        low_memory=False,
    )
    print(f"  {len(df_raw):,} filas × {len(df_raw.columns)} columnas")

    # Convertir numéricas
    for c in df_raw.select_dtypes(include=["object"]).columns:
        try:
            df_raw[c] = pd.to_numeric(df_raw[c])
        except (ValueError, TypeError):
            pass

    print("Ejecutando build_features()...")
    df_feat = build_features(df_raw)

    _print_feature_report(df_raw, df_feat)


if __name__ == "__main__":
    main()
