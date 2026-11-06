"""
Filtra transacciones del train-set-features donde precio_compra o precio_venta
son 0 o nulos (vacíos). Genera data/precio_nulo_o_cero.csv con el resultado.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT  = ROOT / "data" / "train-set-features.csv"
OUTPUT = ROOT / "data" / "precio_nulo_o_cero.csv"

df = pd.read_csv(INPUT, sep=";", dtype=str, low_memory=False)

print(f"Total filas leídas : {len(df):,}")

for col in ("precio_venta", "precio_compra"):
    nulls = df[col].isna().sum()
    zeros = (df[col].str.strip() == "0").sum()
    print(f"  {col}: {nulls:,} nulos, {zeros:,} ceros")

def is_zero_or_null(series: pd.Series) -> pd.Series:
    """True cuando el valor es NaN, vacío o numéricamente 0."""
    num = pd.to_numeric(series, errors="coerce")
    return num.isna() | (num == 0)

mask = is_zero_or_null(df["precio_compra"]) | is_zero_or_null(df["precio_venta"])

resultado = df[mask].sample(n=200, random_state=42)
resultado.to_csv(OUTPUT, index=False, encoding="utf-8")

print(f"\nFilas con precio_compra/precio_venta en 0 o nulo: {len(resultado):,}")
print(f"Archivo generado: {OUTPUT}")

# Desglose por columna afectada
mask_pc = is_zero_or_null(df["precio_compra"])
mask_pv = is_zero_or_null(df["precio_venta"])
print(f"  Solo precio_compra  : {(mask_pc & ~mask_pv).sum():,}")
print(f"  Solo precio_venta   : {(mask_pv & ~mask_pc).sum():,}")
print(f"  Ambas               : {(mask_pc & mask_pv).sum():,}")
