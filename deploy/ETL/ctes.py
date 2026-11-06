import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc
import numpy as np


# ── Helpers de lectura S3 ─────────────────────────────────────────────────────

def _col_map(path: str) -> dict[str, str]:
    """Detecta el schema del Parquet y devuelve {lowercase → nombre_real}.
    partitioning='hive' expone columnas de partición (e.g. Businessdaydate=...)."""
    schema = ds.dataset(path, format="parquet", partitioning="hive").schema
    return {f.name.lower(): f.name for f in schema}


def _actual_cols(cmap: dict, cols: list[str]) -> list[str]:
    """Traduce lista de columnas lowercase al casing real del schema."""
    return [cmap.get(c.lower(), c) for c in cols]


def _read_parquet(path: str, columns: list[str] | None = None, filters=None) -> pd.DataFrame:
    """
    Lee Parquet (local o s3://) y normaliza columnas a minúsculas.
    columns y filters se especifican en minúsculas; el helper mapea al schema real.
    filters: lista de tuplas DNF [(col, op, val), ...]
    """
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    cmap    = {f.name.lower(): f.name for f in dataset.schema}
    schema_types = {f.name: f.type for f in dataset.schema}
    actual = _actual_cols(cmap, columns) if columns else None

    def _coerce_val(col_name, val):
        """Castea val a int solo si la columna del schema es de tipo entero."""
        if isinstance(val, str) and pa.types.is_integer(schema_types.get(col_name, pa.null())):
            try:
                return int(val)
            except ValueError:
                pass
        return val

    actual_filters = (
        [(cmap.get(c.lower(), c), op, _coerce_val(cmap.get(c.lower(), c), val)) for c, op, val in filters]
        if filters else None
    )
    table = pq.read_table(path, columns=actual, filters=actual_filters)
    # decimal128 → float64 para evitar columnas de tipo object con Decimal
    import pyarrow.types as pat
    cast_schema = pa.schema([
        f.with_type(pa.float64()) if pat.is_decimal(f.type) else f
        for f in table.schema
    ])
    table = table.cast(cast_schema)
    df = table.to_pandas()
    df.columns = df.columns.str.lower()
    return df


def _iter_batches(path: str, columns: list[str], filter_expr, batch_size: int = 2_000_000):
    """
    Itera batches de un dataset Parquet con filtro pyarrow.compute,
    normalizando nombres de columna al schema real antes de aplicar el filtro.
    Cada batch se devuelve con columnas en minúsculas.
    """
    dataset = ds.dataset(path, format="parquet")
    cmap    = {f.name.lower(): f.name for f in dataset.schema}
    actual  = _actual_cols(cmap, columns)

    def _remap(expr):
        """Reconstruye una expresión pc reemplazando Field por casing real."""
        if isinstance(expr, pc.Expression):
            s = str(expr)
            for lower, real in cmap.items():
                s = s.replace(f'"{lower}"', f'"{real}"')
            # Reconstruir con eval es frágil; mejor usar el dataset con schema
        return expr

    for batch in dataset.to_batches(
        filter=filter_expr,
        batch_size=batch_size,
        columns=actual,
    ):
        df = batch.to_pandas()
        df.columns = df.columns.str.lower()
        yield df

def ventas_base(tlogf_path: str) -> pd.DataFrame:

    COLS_OUTPUT = [
        "plant", "itemid", "businessdaydate", "transnumber",
        "retailquantity", "salesamount", "cost", "basequantity",
        "promotionid", "rowkey",
    ]
    COLS_FILTER = [
        "businesstype", "datastatus", "recordqualifier",
        "transdebitflag", "distvoid", "taskcanceled", "trainingflag",
        "retailreasoncode",
    ]
    FLAG_COLS = ["transdebitflag", "distvoid", "taskcanceled", "trainingflag"]

    PUSHDOWN_FILTERS = [
        ("businesstype",    "in", pa.array(["1", "4"])),
        ("datastatus",      "=",  "3"),
        ("recordqualifier", "=",  "5"),
    ]

    df = pq.read_table(
        tlogf_path,
        columns=COLS_OUTPUT + COLS_FILTER,
        filters=PUSHDOWN_FILTERS,
    ).to_pandas()

    # 2. Normalizar columnas de flag en batch (una sola pasada)
    df_flags = (
        df[FLAG_COLS + ["retailreasoncode"]]
        .fillna("")
        .astype("string")
        .apply(lambda s: s.str.strip())
    )

    # 3. Filtros residuales que pushdown no resuelve
    mask_flags = (df_flags[FLAG_COLS] != "X").all(axis=1)
    mask_rrc   = df_flags["retailreasoncode"] == ""

    # 4. Selección, renombrado y reset de índice
    return (
        df
        .loc[mask_flags & mask_rrc, COLS_OUTPUT]
        .rename(columns={
            "plant":           "tienda_id",
            "itemid":          "sku_id",
            "businessdaydate": "fecha_venta",
            "retailquantity":  "unidades_vendidas",
            "salesamount":     "monto_venta",
            "cost":            "costo_linea",
            "basequantity":    "uom_base_qty",
            "promotionid":     "promotion_id",
            "rowkey":          "sale_rowkey",
        })
        .reset_index(drop=True)
    )

def ventas_base_dates(
    tlogf_path: str,
    fecha_inicio: str = None,
    fecha_fin: str = None,
) -> pd.DataFrame:
    FLAG_COLS = ["transdebitflag", "distvoid", "taskcanceled", "trainingflag"]

    COLS_OUTPUT = [
        "plant", "itemid", "businessdaydate", "transnumber",
        "retailquantity", "salesamount", "cost", "basequantity",
        "promotionid", "rowkey",
    ]
    COLS_FILTER = FLAG_COLS + ["retailreasoncode"]

    RENAME = {
        "plant": "tienda_id", "itemid": "sku_id",
        "businessdaydate": "fecha_venta",
        "retailquantity": "unidades_vendidas",
        "salesamount": "monto_venta", "cost": "costo_linea",
        "basequantity": "uom_base_qty", "promotionid": "promotion_id",
        "rowkey": "sale_rowkey",
    }

    dataset = ds.dataset(tlogf_path, format="parquet", partitioning="hive")
    cmap    = {f.name.lower(): f.name for f in dataset.schema}
    F       = ds.field

    filtro = (
        pc.is_in(F(cmap["recordqualifier"]), pa.array(["5"])) &
        pc.is_in(F(cmap["businesstype"]),    pa.array(["1", "4"])) &
        pc.equal(F(cmap["datastatus"]), "3")
    )
    if fecha_inicio is not None:
        filtro = filtro & pc.greater_equal(F(cmap["businessdaydate"]), int(fecha_inicio))
    if fecha_fin is not None:
        filtro = filtro & pc.less_equal(F(cmap["businessdaydate"]), int(fecha_fin))

    actual_cols = _actual_cols(cmap, COLS_OUTPUT + COLS_FILTER)

    DECIMAL_COLS = ["retailquantity", "salesamount", "cost", "basequantity"]
    chunks = []
    for batch in dataset.to_batches(filter=filtro, batch_size=2_000_000, columns=actual_cols):
        df = batch.to_pandas()
        df.columns = df.columns.str.lower()
        for c in DECIMAL_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        flags = df[FLAG_COLS + ["retailreasoncode"]].fillna("").astype(str).apply(lambda col: col.str.strip())
        mask = (flags[FLAG_COLS] != "X").all(axis=1) & (flags["retailreasoncode"] == "")
        chunks.append(df.loc[mask, COLS_OUTPUT])

    return pd.concat(chunks, ignore_index=True, copy=False).rename(columns=RENAME)

def promo_lineas(tlogf_path: str) -> pd.DataFrame:

    COLS_OUTPUT = [
        "plant", "businessdaydate", "transnumber", "rowkey",
        "reductionamount", "offerid",
    ]

    COLS_FILTER = [
        "businesstype", "datastatus", "recordqualifier", "discid",
        "transdebitflag", "distvoid", "taskcanceled", "trainingflag",
    ]

    FLAG_COLS = ["transdebitflag", "distvoid", "taskcanceled", "trainingflag"]

    PUSHDOWN_FILTERS = [
        ("businesstype",    "in", pa.array(["1", "4"])),
        ("datastatus",      "=",  "3"),
        ("recordqualifier", "=",  "11"),        # único cambio vs ventas_base
    ]

    df = pq.read_table(
        tlogf_path,
        columns=COLS_OUTPUT + COLS_FILTER,
        filters=PUSHDOWN_FILTERS,
    ).to_pandas()

    df_flags = (
        df[FLAG_COLS + ["discid"]]
        .fillna("")
        .astype("string")
        .apply(lambda s: s.str.strip())
    )

    mask_flags = (df_flags[FLAG_COLS] != "X").all(axis=1)
    mask_discid = df_flags["discid"] == ""      # IS NULL OR discid = ''

    df_final = (
        df
        .loc[mask_flags & mask_discid, COLS_OUTPUT]
        .rename(columns={
            "plant":           "tienda_id",
            "businessdaydate": "fecha_venta",
            "rowkey":          "promo_rowkey",
            "reductionamount": "descuento_monto",
            "offerid":         "offer_id",
        })
        .assign(
            tipo_descuento="PROMO_REGULAR",   # literal 'PROMO_REGULAR'
            disc_id=pd.NA,                    # CAST(NULL AS VARCHAR)
        )
        .reset_index(drop=True)
    )

    return df_final

def promo_lineas_dates(
    tlogf_path: str,
    fecha_inicio: str = None,
    fecha_fin: str = None,
) -> pd.DataFrame:
    """
    fecha_inicio / fecha_fin: 'YYYYMMDD' (inclusive). Sirven para acotar el
    dataset a una ventana chica y validar reglas de negocio con recursos
    limitados, sin tocar la lógica del CTE.
    """

    COLS_OUTPUT = [
        "plant", "businessdaydate", "transnumber", "rowkey",
        "reductionamount", "offerid",
    ]

    COLS_FILTER = [
        "businesstype", "datastatus", "recordqualifier", "discid",
        "transdebitflag", "distvoid", "taskcanceled", "trainingflag",
    ]

    FLAG_COLS = ["transdebitflag", "distvoid", "taskcanceled", "trainingflag"]

    filters = [
        ("businesstype",    "in", pa.array(["1", "4"])),
        ("datastatus",      "=",  "3"),
        ("recordqualifier", "=",  "11"),
    ]
    if fecha_inicio is not None:
        filters.append(("businessdaydate", ">=", fecha_inicio))
    if fecha_fin is not None:
        filters.append(("businessdaydate", "<=", fecha_fin))

    df = _read_parquet(tlogf_path, columns=COLS_OUTPUT + COLS_FILTER, filters=filters)

    df_flags = (
        df[FLAG_COLS + ["discid"]]
        .fillna("")
        .astype("string")
        .apply(lambda s: s.str.strip())
    )

    mask_flags  = (df_flags[FLAG_COLS] != "X").all(axis=1)
    mask_discid = df_flags["discid"] == ""

    return (
        df
        .loc[mask_flags & mask_discid, COLS_OUTPUT]
        .rename(columns={
            "plant":           "tienda_id",
            "businessdaydate": "fecha_venta",
            "rowkey":          "promo_rowkey",
            "reductionamount": "descuento_monto",
            "offerid":         "offer_id",
        })
        .assign(tipo_descuento="PROMO_REGULAR", disc_id=pd.NA)
        .reset_index(drop=True)
    )


def cupon_lineas(tlogf_path: str) -> pd.DataFrame:

    COLS_OUTPUT = [
        "plant", "businessdaydate", "transnumber", "rowkey",
        "reductionamount", "offerid", "discid",
    ]
    COLS_FILTER = [
        "businesstype", "datastatus", "recordqualifier",
        "transdebitflag", "distvoid", "taskcanceled", "trainingflag",
    ]
    FLAG_COLS = ["transdebitflag", "distvoid", "taskcanceled", "trainingflag"]

    PUSHDOWN_FILTERS = [
        ("businesstype",    "in", pa.array(["1", "4"])),
        ("datastatus",      "=",  "3"),
        ("recordqualifier", "=",  "11"),
    ]

    df = pq.read_table(
        tlogf_path,
        columns=COLS_OUTPUT + COLS_FILTER,
        filters=PUSHDOWN_FILTERS,
    ).to_pandas()

    df_flags = (
        df[FLAG_COLS]
        .fillna("")
        .astype("string")
        .apply(lambda s: s.str.strip())
    )

    mask_flags        = (df_flags[FLAG_COLS] != "X").all(axis=1)
    disc_str          = df["discid"].fillna("").astype(str).str.strip()
    mask_discid       = disc_str != ""
    mask_reduction    = df["reductionamount"].notna()

    df_final = (
        df
        .loc[mask_flags & mask_discid & mask_reduction, COLS_OUTPUT]
        .rename(columns={
            "plant":           "tienda_id",
            "businessdaydate": "fecha_venta",
            "rowkey":          "promo_rowkey",
            "reductionamount": "descuento_monto",
            "offerid":         "offer_id",
            "discid":          "disc_id",
        })
        .assign(tipo_descuento="CUPON")
        .reset_index(drop=True)
    )

    return df_final

def cupon_lineas_dates(
    tlogf_path: str,
    fecha_inicio: str = None,
    fecha_fin: str = None,
) -> pd.DataFrame:
    """
    fecha_inicio / fecha_fin: 'YYYYMMDD' (inclusive). Sirven para acotar el
    dataset a una ventana chica y validar reglas de negocio con recursos
    limitados, sin tocar la lógica del CTE.
    """

    COLS_OUTPUT = [
        "plant", "businessdaydate", "transnumber", "rowkey",
        "reductionamount", "offerid", "discid",
    ]
    COLS_FILTER = [
        "businesstype", "datastatus", "recordqualifier",
        "transdebitflag", "distvoid", "taskcanceled", "trainingflag",
    ]
    FLAG_COLS = ["transdebitflag", "distvoid", "taskcanceled", "trainingflag"]

    filters = [
        ("businesstype",    "in", pa.array(["1", "4"])),
        ("datastatus",      "=",  "3"),
        ("recordqualifier", "=",  "11"),
    ]
    if fecha_inicio is not None:
        filters.append(("businessdaydate", ">=", fecha_inicio))
    if fecha_fin is not None:
        filters.append(("businessdaydate", "<=", fecha_fin))

    df = _read_parquet(tlogf_path, columns=COLS_OUTPUT + COLS_FILTER, filters=filters)

    df_flags   = df[FLAG_COLS].fillna("").astype("string").apply(lambda s: s.str.strip())
    mask_flags = (df_flags[FLAG_COLS] != "X").all(axis=1)
    disc_str   = df["discid"].fillna("").astype(str).str.strip()

    return (
        df
        .loc[mask_flags & (disc_str != "") & df["reductionamount"].notna(), COLS_OUTPUT]
        .rename(columns={
            "plant":           "tienda_id",
            "businessdaydate": "fecha_venta",
            "rowkey":          "promo_rowkey",
            "reductionamount": "descuento_monto",
            "offerid":         "offer_id",
            "discid":          "disc_id",
        })
        .assign(tipo_descuento="CUPON")
        .reset_index(drop=True)
    )

def promo_nombre_ext(tlogf_ext_path: str) -> pd.DataFrame:

    COLS_OUTPUT = [
        "retailstoreid", "businessdaydate", "transnumber",
        "parentkey", "fieldvalue",
    ]
    COLS_FILTER = ["recordqualifier", "fieldname"]

    PUSHDOWN_FILTERS = [
        ("recordqualifier", "=", "12"),
        ("fieldname",       "=", "NOM_PROMO"),
    ]

    df = pq.read_table(
        tlogf_ext_path,
        columns=COLS_OUTPUT + COLS_FILTER,
        filters=PUSHDOWN_FILTERS,
    ).to_pandas()

    fv = df["fieldvalue"].fillna("").astype(str).str.strip()
    mask_fieldvalue = fv != ""

    df = df.loc[mask_fieldvalue].copy()
    df["tienda_id"] = df["retailstoreid"].astype(str).str[6:10]

    df_final = (
        df[["tienda_id", "businessdaydate", "transnumber", "parentkey", "fieldvalue"]]
        .rename(columns={
            "businessdaydate": "fecha_venta",
            "parentkey":       "promo_rowkey",
            "fieldvalue":      "nombre_promo",
        })
        .reset_index(drop=True)
    )

    return df_final

def promo_nombre_ext_dates(
    tlogf_ext_path: str,
    fecha_inicio: str = None,
    fecha_fin: str = None,
) -> pd.DataFrame:
    """
    fecha_inicio / fecha_fin: 'YYYYMMDD' (inclusive). Sirven para acotar el
    dataset a una ventana chica y validar reglas de negocio con recursos
    limitados, sin tocar la lógica del CTE.
    """

    COLS_OUTPUT = [
        "retailstoreid", "businessdaydate", "transnumber",
        "parentkey", "fieldvalue",
    ]
    COLS_FILTER = ["recordqualifier", "fieldname"]

    filters = [
        ("recordqualifier", "=", "12"),
        ("fieldname",       "=", "NOM_PROMO"),
    ]
    if fecha_inicio is not None:
        filters.append(("businessdaydate", ">=", fecha_inicio))
    if fecha_fin is not None:
        filters.append(("businessdaydate", "<=", fecha_fin))

    df = _read_parquet(tlogf_ext_path, columns=COLS_OUTPUT + COLS_FILTER, filters=filters)

    df = df.loc[df["fieldvalue"].fillna("").astype(str).str.strip() != ""].copy()
    df["tienda_id"] = df["retailstoreid"].astype(str).str[6:10]

    return (
        df[["tienda_id", "businessdaydate", "transnumber", "parentkey", "fieldvalue"]]
        .rename(columns={
            "businessdaydate": "fecha_venta",
            "parentkey":       "promo_rowkey",
            "fieldvalue":      "nombre_promo",
        })
        .reset_index(drop=True)
    )

def promo_consolidada(df_promo: pd.DataFrame, df_cupon: pd.DataFrame, df_ext: pd.DataFrame) -> pd.DataFrame:

    JOIN_COLS_PROMO = ["promo_rowkey", "transnumber", "fecha_venta", "tienda_id"]
    JOIN_COLS_CUPON = ["transnumber", "fecha_venta", "tienda_id"]
    GROUP_COLS      = ["transnumber", "tienda_id", "fecha_venta"]
    COLS_UNION      = GROUP_COLS + ["descuento_monto", "disc_id",
                                    "offer_id", "tipo_descuento", "nombre_promo"]

    # ── 1. Normalizar llaves a str + category antes de cualquier operación ────
    def _cat(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        df = df.copy()
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip().astype("category")
        return df

    df_promo = _cat(df_promo, JOIN_COLS_PROMO)
    df_cupon = _cat(df_cupon, JOIN_COLS_CUPON)
    df_ext   = _cat(df_ext,   JOIN_COLS_PROMO)

    # ── 2. LEFT JOINs directos contra promo_nombre_ext (sin pre-agregar),
    #        igual que el query de Athena ─────────────────────────────────
    rama_promo = df_promo.merge(
        df_ext[JOIN_COLS_PROMO + ["nombre_promo"]], on=JOIN_COLS_PROMO, how="left"
    )[COLS_UNION]
    rama_cupon = df_cupon.merge(
        df_ext[JOIN_COLS_CUPON + ["nombre_promo"]], on=JOIN_COLS_CUPON, how="left"
    )[COLS_UNION]

    # ── 3. UNION ALL ────────────────────────────────────────────────────────
    union = pd.concat([rama_promo, rama_cupon], ignore_index=True, copy=False)
    union["descuento_monto"] = pd.to_numeric(union["descuento_monto"], errors="coerce")
    for col in ["disc_id", "offer_id", "tipo_descuento", "nombre_promo"]:
        union[col] = union[col].astype("string")

    # ── 4. GROUP BY con observed=True (evita combinaciones vacías) ────────────
    return (
        union
        .groupby(GROUP_COLS, as_index=False, observed=True)
        .agg(
            descuento_total=("descuento_monto", "sum"),
            disc_id        =("disc_id",         "max"),
            offer_id       =("offer_id",         "max"),
            tipo_descuento =("tipo_descuento",   "max"),
            nombre_promo   =("nombre_promo",     "max"),
        )
        .reset_index(drop=True)
    )

def dim_mat_clean(dim_mat_path: str) -> pd.DataFrame:
    COLS = [
        "sku_id", "generico_id", "negocio_desc", "genero_desc",
        "division_desc", "deporte_desc", "familia_desc", "talla",
        "marca_desc", "proveedor_id", "proveedor_desc", "color",
    ]
    EXCLUIDOS = {"(Sin valor)", "(Nulo)", ""}

    df = _read_parquet(dim_mat_path, columns=COLS)

    gid = df["generico_id"].fillna("").astype(str).str.strip()
    mask = ~gid.isin(EXCLUIDOS)

    return df.loc[mask].reset_index(drop=True)

def dim_suc_clean(dim_suc_path: str) -> pd.DataFrame:
    COLS = [
        "centro_id", "centro_desc", "banner_desc", "region_desc",
        "canal_distribucion_id", "canal_distribucion_desc",
        "zona_ventas_id", "zona_ventas_desc", "estado", "ciudad",
        "formato", "lista_precios_id", "organizacion_id",
    ]

    df = _read_parquet(dim_suc_path, columns=COLS)

    df["banner"] = df["banner_desc"].fillna("").astype(str).str.strip()
    df = df.drop(columns=["banner_desc"]).rename(columns={"centro_id": "tienda_id"})

    return df.reset_index(drop=True)

def fechas_ventas_unicas(df_ventas_base: pd.DataFrame) -> pd.DataFrame:
    fechas = df_ventas_base["fecha_venta"].dropna().astype(int).unique()
    return pd.DataFrame({"fecha_venta_ref": sorted(fechas)})


def precio_venta_dedup(a155_path: str, konp_v_path: str, ecc_path: str) -> pd.DataFrame:

    IVA = {"3": 1.16, "03": 1.16, "4": 1.08, "04": 1.08}

    def _parse_fecha(s: pd.Series) -> pd.Series:
        """COALESCE(TRY(dd/mm/yyyy → int), TRY(int directo), 99991231)"""
        resultado = pd.Series(pd.NA, index=s.index, dtype="Int64")

        # intento 1: formato dd/mm/yyyy
        parsed = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
        mask1 = parsed.notna()
        resultado[mask1] = parsed[mask1].dt.strftime("%Y%m%d").astype("Int64")

        # intento 2: ya es YYYYMMDD como string/int
        mask2 = ~mask1
        como_int = pd.to_numeric(s[mask2], errors="coerce").astype("Int64")
        resultado[mask2] = como_int

        return resultado

    def _apply_iva(kbetr: pd.Series, pltyp: pd.Series) -> pd.Series:
        factor = pltyp.astype(str).map(IVA).fillna(1.16)
        return kbetr * factor

    # ── Rama 1: a155 JOIN konp_v ──────────────────────────────────────
    a155 = _read_parquet(a155_path, columns=["matnr", "pltyp", "datab", "datbi", "knumh"])
    konp = _read_parquet(konp_v_path, columns=["knumh", "kbetr"],
                         filters=[("kbetr", ">=", 10.0)])

    rama1 = a155.merge(konp, on="knumh", how="inner")
    rama1["generico_id"]       = rama1["matnr"].astype(str)
    rama1["lista_precios_id"]  = rama1["pltyp"].astype(str)
    rama1["vigencia_desde"]    = _parse_fecha(rama1["datab"].astype(str))
    rama1["vigencia_hasta"]    = _parse_fecha(rama1["datbi"].astype(str)).fillna(99991231)
    rama1["precio_venta_bruto"]   = pd.to_numeric(rama1["kbetr"], errors="coerce")
    rama1["precio_venta_con_iva"] = _apply_iva(rama1["precio_venta_bruto"], rama1["lista_precios_id"])

    # ── Rama 2: ecc ───────────────────────────────────────────────────
    ecc = _read_parquet(ecc_path, columns=["matnr", "pltyp", "datab", "datbi", "kbetr"],
                        filters=[("kbetr", ">=", 10.0)])

    rama2 = ecc.copy()
    rama2["generico_id"]       = rama2["matnr"].astype(str)
    rama2["lista_precios_id"]  = rama2["pltyp"].astype(str)
    rama2["vigencia_desde"]    = _parse_fecha(rama2["datab"].astype(str))
    rama2["vigencia_hasta"]    = _parse_fecha(rama2["datbi"].astype(str)).fillna(99991231)
    rama2["precio_venta_bruto"]   = pd.to_numeric(rama2["kbetr"], errors="coerce")
    rama2["precio_venta_con_iva"] = _apply_iva(rama2["precio_venta_bruto"], rama2["lista_precios_id"])

    COLS_OUT = [
        "generico_id", "lista_precios_id", "vigencia_desde",
        "vigencia_hasta", "precio_venta_bruto", "precio_venta_con_iva",
    ]

    # ── UNION ALL + ROW_NUMBER (keep max precio_venta_bruto por partición) ─
    union = pd.concat([rama1[COLS_OUT], rama2[COLS_OUT]], ignore_index=True, copy=False)

    PARTITION = ["generico_id", "lista_precios_id", "vigencia_desde", "vigencia_hasta"]
    return (
        union
        .sort_values("precio_venta_bruto", ascending=False)
        .drop_duplicates(subset=PARTITION, keep="first")
        .reset_index(drop=True)
    )

def precio_venta(df_pvd: pd.DataFrame, df_fechas: pd.DataFrame) -> pd.DataFrame:

    COLS_OUT = [
        "generico_id", "lista_precios_id", "vigencia_desde", "vigencia_hasta",
        "precio_venta_bruto", "precio_venta_con_iva", "fecha_venta_ref", "rn_dia",
    ]

    # Pre-ordenar una sola vez según el ORDER BY del ROW_NUMBER
    df_sorted = df_pvd.assign(
        _ord1=(pd.to_numeric(df_pvd["vigencia_hasta"], errors="coerce") == 99991231).astype("int8")
    ).sort_values(
        ["generico_id", "lista_precios_id", "_ord1", "vigencia_desde", "precio_venta_bruto"],
        ascending=[True, True, True, False, False],
    ).drop(columns=["_ord1"]).reset_index(drop=True)

    # Numpy arrays para el filtro de rango — evita overhead de pandas por iteración
    vd = pd.to_numeric(df_sorted["vigencia_desde"], errors="coerce").to_numpy()
    vh = pd.to_numeric(df_sorted["vigencia_hasta"], errors="coerce").to_numpy()

    DEDUP_KEY = ["generico_id", "lista_precios_id"]
    BATCH_SIZE = 100   # fechas por lote antes de concat parcial
    fechas = df_fechas["fecha_venta_ref"].tolist()

    result_chunks = []

    for i in range(0, len(fechas), BATCH_SIZE):
        batch = fechas[i:i + BATCH_SIZE]
        batch_rows = []
        for fecha in batch:
            mask = (vd <= fecha) & (vh >= fecha)
            if not mask.any():
                continue
            # Todas las filas candidatas del día (no solo la ganadora),
            # con rn_dia real vía cumcount — igual al ROW_NUMBER() de SQL.
            chunk = df_sorted.loc[mask].copy()
            chunk["fecha_venta_ref"] = fecha
            chunk["rn_dia"] = chunk.groupby(DEDUP_KEY, observed=True).cumcount() + 1
            batch_rows.append(chunk)

        if batch_rows:
            result_chunks.append(
                pd.concat(batch_rows, ignore_index=True, copy=False)[COLS_OUT]
            )
        batch_rows.clear()

    return pd.concat(result_chunks, ignore_index=True, copy=False)

def precio_compra_dedup(a018_path: str, konp_path: str, ecc_compra_path: str) -> pd.DataFrame:

    def _parse_fecha(s: pd.Series) -> pd.Series:
        resultado = pd.Series(pd.NA, index=s.index, dtype="Int64")
        parsed = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
        mask1 = parsed.notna()
        resultado[mask1] = parsed[mask1].dt.strftime("%Y%m%d").astype("Int64")
        como_int = pd.to_numeric(s[~mask1], errors="coerce").astype("Int64")
        resultado[~mask1] = como_int
        return resultado

    # ── Rama 1: a018 JOIN konp ────────────────────────────────────────
    a018 = _read_parquet(a018_path, columns=["matnr", "datab", "datbi", "knumh"])
    konp_f = _read_parquet(konp_path, columns=["knumh", "kbetr"],
                           filters=[("kbetr", ">", 0.0)])

    rama1 = a018.merge(konp_f, on="knumh", how="inner")
    rama1["generico_id"]    = rama1["matnr"].astype(str)
    rama1["vigencia_desde"] = _parse_fecha(rama1["datab"].astype(str))
    rama1["vigencia_hasta"] = _parse_fecha(rama1["datbi"].astype(str)).fillna(99991231)
    rama1["precio_compra"]  = pd.to_numeric(rama1["kbetr"], errors="coerce")

    # ── Rama 2: ecc compra ────────────────────────────────────────────
    ecc_f = _read_parquet(ecc_compra_path, columns=["matnr", "datab", "datbi", "kbetr"],
                          filters=[("kbetr", ">", 0.0)])

    rama2 = ecc_f.copy()
    rama2["generico_id"]    = rama2["matnr"].astype(str)
    rama2["vigencia_desde"] = _parse_fecha(rama2["datab"].astype(str))
    rama2["vigencia_hasta"] = _parse_fecha(rama2["datbi"].astype(str)).fillna(99991231)
    rama2["precio_compra"]  = pd.to_numeric(rama2["kbetr"], errors="coerce")

    COLS_OUT = ["generico_id", "vigencia_desde", "vigencia_hasta", "precio_compra"]
    PARTITION = ["generico_id", "vigencia_desde", "vigencia_hasta"]

    union = pd.concat([rama1[COLS_OUT], rama2[COLS_OUT]], ignore_index=True, copy=False)

    # ROW_NUMBER PARTITION BY generico_id, vigencia_desde, vigencia_hasta ORDER BY precio_compra DESC
    return (
        union
        .sort_values("precio_compra", ascending=False)
        .drop_duplicates(subset=PARTITION, keep="first")
        .reset_index(drop=True)
    )

def precio_compra(df_pcd: pd.DataFrame, df_fechas: pd.DataFrame) -> pd.DataFrame:

    COLS_OUT = ["generico_id", "precio_compra", "fecha_venta_ref", "rn_dia"]
    PARTITION = ["generico_id", "fecha_venta_ref"]

    # Pre-ordenar una sola vez según ORDER BY del ROW_NUMBER
    df_sorted = df_pcd.assign(
        _ord1=(pd.to_numeric(df_pcd["vigencia_hasta"], errors="coerce") == 99991231).astype("int8")
    ).sort_values(
        ["generico_id", "_ord1", "vigencia_desde", "precio_compra"],
        ascending=[True, True, False, False],
    ).drop(columns=["_ord1"]).reset_index(drop=True)

    vd = pd.to_numeric(df_sorted["vigencia_desde"], errors="coerce").to_numpy()
    vh = pd.to_numeric(df_sorted["vigencia_hasta"], errors="coerce").to_numpy()

    DEDUP_KEY = ["generico_id"]
    BATCH_SIZE = 100
    fechas = df_fechas["fecha_venta_ref"].tolist()
    result_chunks = []

    for i in range(0, len(fechas), BATCH_SIZE):
        batch = fechas[i:i + BATCH_SIZE]
        batch_rows = []
        for fecha in batch:
            mask = (vd <= fecha) & (vh >= fecha)
            if not mask.any():
                continue
            # Todas las filas candidatas del día (no solo la ganadora),
            # con rn_dia real vía cumcount — igual al ROW_NUMBER() de SQL.
            chunk = df_sorted.loc[mask, ["generico_id", "precio_compra"]].copy()
            chunk["fecha_venta_ref"] = fecha
            chunk["rn_dia"] = chunk.groupby(DEDUP_KEY, observed=True).cumcount() + 1
            batch_rows.append(chunk)

        if batch_rows:
            result_chunks.append(
                pd.concat(batch_rows, ignore_index=True, copy=False)[COLS_OUT]
            )
        batch_rows.clear()

    return pd.concat(result_chunks, ignore_index=True, copy=False)

def inventario_mensual(mard_path: str, dim_mat_path: str) -> pd.DataFrame:

    mard = _read_parquet(mard_path, columns=["matnr", "werks", "anomes", "labst"])
    dim  = _read_parquet(dim_mat_path, columns=["sku_id", "generico_id"])

    # Filtro dim_mat (mismo WHERE que dim_mat_clean)
    gid  = dim["generico_id"].fillna("").astype(str).str.strip()
    dim  = dim.loc[~gid.isin({"(Sin valor)", "(Nulo)", ""})].copy()

    # Puente talla → genérico: CAST(m.matnr AS VARCHAR) = dm.sku_id
    mard["matnr"] = mard["matnr"].astype(str)

    df = mard.merge(dim, left_on="matnr", right_on="sku_id", how="inner")

    # GREATEST(COALESCE(labst, 0), 0) — stock nunca negativo
    df["labst"] = pd.to_numeric(df["labst"], errors="coerce").fillna(0)
    df["labst"] = df["labst"].clip(lower=0)

    return (
        df.groupby(["generico_id", "werks", "anomes"], as_index=False, observed=True)
        .agg(stock_disponible=("labst", "sum"))
        .rename(columns={"werks": "tienda_id", "anomes": "anomes"})
        .assign(anomes=lambda d: pd.to_numeric(d["anomes"], errors="coerce").astype("Int64"))
        .reset_index(drop=True)
    )

def calendario_promo_activo(cal_path: str) -> pd.DataFrame:
    COLS = ["bu_id", "promotion_id", "name", "effective_date", "expiry_date", "status_tcd"]

    df = _read_parquet(cal_path, columns=COLS)

    mask = df["status_tcd"].fillna("").astype(str).str.strip() == "AC"

    return (
        df.loc[mask, ["bu_id", "promotion_id", "name", "effective_date", "expiry_date"]]
        .assign(tienda_id=lambda d: d["bu_id"].astype(str).str.strip())
        .drop(columns=["bu_id"])
        .reset_index(drop=True)
    )

def calendario_por_dia(df_cpa: pd.DataFrame, df_fechas: pd.DataFrame) -> pd.DataFrame:

    # Convertir fechas a numérico una sola vez
    eff = pd.to_numeric(df_cpa["effective_date"], errors="coerce").to_numpy()
    exp = pd.to_numeric(df_cpa["expiry_date"],    errors="coerce").to_numpy()

    fechas = df_fechas["fecha_venta_ref"].tolist()
    chunks = []

    for fecha in fechas:
        # BETWEEN effective_date AND expiry_date
        mask = (eff <= fecha) & (exp >= fecha)
        if not mask.any():
            continue

        chunk = df_cpa.loc[mask, ["tienda_id", "promotion_id", "name", "effective_date"]].copy()
        chunk["fecha_venta"] = fecha

        # COUNT(DISTINCT promotion_id) + max_by(name, effective_date)
        agg = chunk.groupby("tienda_id", as_index=False, observed=True).agg(
            n_promos_activas_tienda=("promotion_id", "nunique"),
            # max_by(name, effective_date) = name del registro con effective_date máximo
            promo_principal_tienda=("name", lambda s: s.loc[
                chunk.loc[s.index, "effective_date"]
                .pipe(pd.to_numeric, errors="coerce")
                .idxmax()
            ]),
        )
        agg["fecha_venta"]          = fecha
        agg["en_calendario_promo"]  = 1
        chunks.append(agg)

    COLS_OUT = [
        "tienda_id", "fecha_venta", "en_calendario_promo",
        "n_promos_activas_tienda", "promo_principal_tienda",
    ]
    if not chunks:
        return pd.DataFrame(columns=COLS_OUT)
    return pd.concat(chunks, ignore_index=True, copy=False)[COLS_OUT]


def ventas_enriquecida(
    df_vb:  pd.DataFrame,   # ventas_base
    df_pc:  pd.DataFrame,   # promo_consolidada
    df_dm:  pd.DataFrame,   # dim_mat_clean
    df_ds:  pd.DataFrame,   # dim_suc_clean
    df_pv:  pd.DataFrame,   # precio_venta
    df_pc2: pd.DataFrame,   # precio_compra
    df_inv: pd.DataFrame,   # inventario_mensual
    df_cpd: pd.DataFrame,   # calendario_por_dia
) -> pd.DataFrame:

    COLS_OUT = [
        "tienda_id", "generico_id", "fecha_venta", "transnumber",
        "unidades_vendidas", "monto_venta", "costo_linea", "uom_base_qty",
        "descuento_total", "disc_id", "nombre_promo", "tipo_descuento", "tiene_promo_flag",
        "negocio_desc", "genero_desc", "division_desc", "deporte_desc", "familia_desc",
        "talla", "marca_desc", "proveedor_id", "proveedor_desc", "color",
        "centro_desc", "banner", "region_desc", "canal_distribucion_id",
        "canal_distribucion_desc", "zona_ventas_id", "zona_ventas_desc",
        "estado", "ciudad", "formato", "lista_precios_id", "organizacion_id",
        "precio_venta", "precio_venta_bruto", "precio_compra",
        "stock_disponible", "en_calendario_promo", "n_promos_activas_tienda",
        "promo_principal_tienda",
    ]

    vb = df_vb.copy()
    # fecha_venta puede llegar como int32 (partición Hive) — normalizar a str
    vb["fecha_venta"]  = vb["fecha_venta"].astype(str)
    # Llaves de fecha pre-calculadas una sola vez (CAST(fecha_venta AS INTEGER) del query)
    vb["_fecha_int"]  = pd.to_numeric(vb["fecha_venta"], errors="coerce").astype("Int64")
    vb["_anomes_int"] = vb["fecha_venta"].str[:6].pipe(pd.to_numeric, errors="coerce").astype("Int64")

    # promo_consolidada trae las llaves como "category" (optimización interna de esa función);
    # las volvemos str acá para que el merge no dependa de ese detalle de implementación.
    pc = df_pc.copy()
    for c in ["transnumber", "tienda_id", "fecha_venta"]:
        pc[c] = pc[c].astype(str)

    # ── LEFT JOIN promo_consolidada ────────────────────────────────────────
    merged = vb.merge(
        pc, on=["transnumber", "tienda_id", "fecha_venta"],
        how="left", indicator="_match_promo",
    )
    merged["tiene_promo_flag"] = (merged["_match_promo"] == "both").astype("int8")
    merged = merged.drop(columns=["_match_promo"])
    merged["descuento_total"] = pd.to_numeric(merged["descuento_total"], errors="coerce").fillna(0)
    merged["tipo_descuento"]  = merged["tipo_descuento"].fillna("SIN_DESCUENTO")

    # ── INNER JOIN dim_mat_clean / dim_suc_clean ───────────────────────────
    merged = merged.merge(df_dm, on="sku_id", how="inner")
    merged = merged.merge(df_ds, on="tienda_id", how="inner")

    # ── LEFT JOIN precio_venta, filtrando rn_dia = 1 antes del join ────────
    pv1 = df_pv.loc[
        df_pv["rn_dia"] == 1,
        ["generico_id", "lista_precios_id", "fecha_venta_ref", "precio_venta_bruto", "precio_venta_con_iva"],
    ]
    merged = merged.merge(
        pv1,
        left_on=["generico_id", "lista_precios_id", "_fecha_int"],
        right_on=["generico_id", "lista_precios_id", "fecha_venta_ref"],
        how="left",
    ).drop(columns=["fecha_venta_ref"]).rename(columns={"precio_venta_con_iva": "precio_venta"})

    # ── LEFT JOIN precio_compra, filtrando rn_dia = 1 antes del join ───────
    pc2_1 = df_pc2.loc[df_pc2["rn_dia"] == 1, ["generico_id", "fecha_venta_ref", "precio_compra"]]
    merged = merged.merge(
        pc2_1,
        left_on=["generico_id", "_fecha_int"],
        right_on=["generico_id", "fecha_venta_ref"],
        how="left",
    ).drop(columns=["fecha_venta_ref"])

    # ── LEFT JOIN inventario_mensual (CAST(SUBSTR(fecha_venta,1,6) AS INT)) ─
    merged = merged.merge(
        df_inv,
        left_on=["generico_id", "tienda_id", "_anomes_int"],
        right_on=["generico_id", "tienda_id", "anomes"],
        how="left",
    ).drop(columns=["anomes"])
    merged["stock_disponible"] = (
        pd.to_numeric(merged["stock_disponible"], errors="coerce").fillna(0).clip(lower=0)
    )

    # ── LEFT JOIN calendario_por_dia ────────────────────────────────────────
    # renombramos su "fecha_venta" para no chocar con el "fecha_venta" (string) de vb
    cpd = df_cpd.rename(columns={"fecha_venta": "_cpd_fecha"})
    merged = merged.merge(
        cpd,
        left_on=["tienda_id", "_fecha_int"],
        right_on=["tienda_id", "_cpd_fecha"],
        how="left",
    ).drop(columns=["_cpd_fecha"])
    merged["en_calendario_promo"] = (
        pd.to_numeric(merged["en_calendario_promo"], errors="coerce").fillna(0).astype("int8")
    )
    merged["n_promos_activas_tienda"] = (
        pd.to_numeric(merged["n_promos_activas_tienda"], errors="coerce").fillna(0).astype("int64")
    )

    return (
        merged
        .drop(columns=["_fecha_int", "_anomes_int"])
        [COLS_OUT]
        .reset_index(drop=True)
    )


def ventas_diaria(df_ve: pd.DataFrame) -> pd.DataFrame:

    GROUP_COLS = ["tienda_id", "generico_id", "fecha_venta"]

    ve = df_ve

    # unidades_vendidas / monto_venta / costo_linea / uom_base_qty pueden llegar
    # como dtype "string" desde tlogf — sin castear, el SUM() del groupby
    # concatena texto en vez de sumar números para grupos con 2+ transacciones
    # (mismo bug que descuento_monto en promo_consolidada).
    for col in ["unidades_vendidas", "monto_venta", "costo_linea", "uom_base_qty"]:
        ve[col] = pd.to_numeric(ve[col], errors="raise")

    # Flags planos para reconstruir el CASE de tipo_descuento sin lambda por grupo
    ve["_es_promo_regular"]       = (ve["tipo_descuento"] == "PROMO_REGULAR").astype("int8")
    ve["_es_cupon"]               = (ve["tipo_descuento"] == "CUPON").astype("int8")
    ve["en_calendario_promo"]     = pd.to_numeric(ve["en_calendario_promo"], errors="coerce").fillna(0)
    ve["n_promos_activas_tienda"] = pd.to_numeric(ve["n_promos_activas_tienda"], errors="coerce").fillna(0)

    agg = ve.groupby(GROUP_COLS, as_index=False, observed=True).agg(
        unidades_vendidas       =("unidades_vendidas", "sum"),
        monto_venta             =("monto_venta", "sum"),
        costo_linea             =("costo_linea", "sum"),
        uom_base_qty            =("uom_base_qty", "sum"),

        tiene_promo_flag        =("tiene_promo_flag", "max"),
        descuento_total         =("descuento_total", "sum"),
        nombre_promo            =("nombre_promo", "max"),

        _hay_promo_regular      =("_es_promo_regular", "max"),
        _hay_cupon              =("_es_cupon", "max"),

        disc_id                 =("disc_id", "max"),
        precio_venta            =("precio_venta", "max"),
        precio_venta_bruto      =("precio_venta_bruto", "max"),
        precio_compra           =("precio_compra", "max"),
        stock_disponible        =("stock_disponible", "max"),

        en_calendario_promo     =("en_calendario_promo", "max"),
        n_promos_activas_tienda =("n_promos_activas_tienda", "max"),
        promo_principal_tienda  =("promo_principal_tienda", "max"),

        negocio_desc            =("negocio_desc", "max"),
        genero_desc             =("genero_desc", "max"),
        division_desc           =("division_desc", "max"),
        deporte_desc            =("deporte_desc", "max"),
        familia_desc            =("familia_desc", "max"),
        n_tallas_vendidas       =("talla", "nunique"),
        n_colores_vendidos      =("color", "nunique"),
        marca_desc              =("marca_desc", "max"),
        proveedor_id            =("proveedor_id", "max"),
        proveedor_desc          =("proveedor_desc", "max"),
        centro_desc             =("centro_desc", "max"),
        banner                  =("banner", "max"),
        region_desc             =("region_desc", "max"),
        canal_distribucion_id   =("canal_distribucion_id", "max"),
        canal_distribucion_desc =("canal_distribucion_desc", "max"),
        zona_ventas_id          =("zona_ventas_id", "max"),
        zona_ventas_desc        =("zona_ventas_desc", "max"),
        estado                  =("estado", "max"),
        ciudad                  =("ciudad", "max"),
        formato                 =("formato", "max"),
        lista_precios_id        =("lista_precios_id", "max"),
        organizacion_id         =("organizacion_id", "max"),
    )

    agg["tipo_descuento"] = "SIN_DESCUENTO"
    agg.loc[agg["_hay_cupon"] == 1, "tipo_descuento"] = "CUPON"
    agg.loc[agg["_hay_promo_regular"] == 1, "tipo_descuento"] = "PROMO_REGULAR"
    agg.loc[(agg["_hay_promo_regular"] == 1) & (agg["_hay_cupon"] == 1), "tipo_descuento"] = "AMBOS"

    agg["tienda_promo_sku_sin_descuento"] = (
        (agg["en_calendario_promo"] == 1) & (agg["tiene_promo_flag"] == 0)
    ).astype("int8")

    agg = agg.drop(columns=["_hay_promo_regular", "_hay_cupon"])

    COLS_OUT = [
        "tienda_id", "generico_id", "fecha_venta",
        "unidades_vendidas", "monto_venta", "costo_linea", "uom_base_qty",
        "tiene_promo_flag", "descuento_total", "nombre_promo", "tipo_descuento",
        "disc_id", "precio_venta", "precio_venta_bruto", "precio_compra", "stock_disponible",
        "en_calendario_promo", "n_promos_activas_tienda", "promo_principal_tienda",
        "tienda_promo_sku_sin_descuento",
        "negocio_desc", "genero_desc", "division_desc", "deporte_desc", "familia_desc",
        "n_tallas_vendidas", "n_colores_vendidos", "marca_desc", "proveedor_id", "proveedor_desc",
        "centro_desc", "banner", "region_desc", "canal_distribucion_id", "canal_distribucion_desc",
        "zona_ventas_id", "zona_ventas_desc", "estado", "ciudad", "formato",
        "lista_precios_id", "organizacion_id",
    ]

    return agg[COLS_OUT].reset_index(drop=True)


def ventas_con_features(df_vd: pd.DataFrame) -> pd.DataFrame:

    GROUP_COLS = ["tienda_id", "generico_id"]
    vd = df_vd.sort_values(GROUP_COLS + ["fecha_venta"]).reset_index(drop=True)

    NUM_COLS = ["monto_venta", "unidades_vendidas", "costo_linea", "descuento_total",
                "precio_venta", "precio_compra"]
    for col in NUM_COLS:
        vd[col] = pd.to_numeric(vd[col], errors="coerce")

    keys = [vd["tienda_id"], vd["generico_id"]]

    def _rolling_lag(s: pd.Series, window: int, stat: str) -> pd.Series:
        """Equivale a ROWS BETWEEN window PRECEDING AND 1 PRECEDING:
        se calcula el agregado incluyendo la fila actual (rolling normal)
        y luego se recorre 1 posición hacia abajo (shift) por partición —
        matemáticamente idéntico a "primero shiftear, luego rolling", pero
        vectorizado vía groupby().rolling() en vez de un shift previo."""
        r = getattr(s.groupby(keys).rolling(window, min_periods=1), stat)()
        r = r.reset_index(level=[0, 1], drop=True)
        return r.groupby(keys).shift(1)

    def _expanding_lag(s: pd.Series, stat: str) -> pd.Series:
        """Equivale a ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING."""
        e = getattr(s.groupby(keys).expanding(min_periods=1), stat)()
        e = e.reset_index(level=[0, 1], drop=True)
        return e.groupby(keys).shift(1)

    # ── Precio transaccional / margen (fila a fila, sin ventana) ───────────
    vd["precio_efectivo"] = vd["monto_venta"] / vd["unidades_vendidas"].replace(0, np.nan)
    vd["costo_unitario"]  = vd["costo_linea"] / vd["unidades_vendidas"].replace(0, np.nan)
    vd["pct_descuento_calc"] = vd["descuento_total"] / (
        vd["monto_venta"] + vd["descuento_total"]
    ).replace(0, np.nan)
    vd["margin_bruto"] = vd["precio_venta"] - vd["precio_compra"]
    vd["pct_margen"]   = (vd["precio_venta"] - vd["precio_compra"]) / vd["precio_venta"].replace(0, np.nan)

    # ── TAREA 3.1: features de demanda baseline ────────────────────────────
    sin_promo_sin_cal = vd["unidades_vendidas"].where(
        (vd["tiene_promo_flag"] == 0) & (vd["en_calendario_promo"] == 0)
    )
    sin_promo = vd["unidades_vendidas"].where(vd["tiene_promo_flag"] == 0)

    vd["avg_units_4w"]  = _rolling_lag(sin_promo_sin_cal, 28, "mean")
    vd["avg_units_8w"]  = _rolling_lag(sin_promo_sin_cal, 56, "mean")
    vd["avg_units_12w"] = _rolling_lag(sin_promo_sin_cal, 84, "mean")

    std_28 = _rolling_lag(sin_promo, 28, "std")
    avg_28_sin_promo = _rolling_lag(sin_promo, 28, "mean")
    vd["coef_variacion"] = std_28 / avg_28_sin_promo.replace(0, np.nan)

    n_dias_baseline_28 = _rolling_lag(
        ((vd["tiene_promo_flag"] == 0) & (vd["en_calendario_promo"] == 0)).astype("int8"),
        28, "sum",
    )
    # COALESCE: ventana vacía (primeras filas) → 0 días de baseline, no NULL
    vd["flag_baseline_insuficiente"] = (n_dias_baseline_28.fillna(0) < 14).astype("int8")

    # ── TAREA 3.2: features de historial promocional ───────────────────────
    es_promo_flag = (vd["tiene_promo_flag"] == 1).astype("int8")

    vd["frecuencia_promo_90d"] = _rolling_lag(es_promo_flag, 90, "sum").fillna(0).astype("int64")

    # última fecha con promo, expanding max (string YYYYMMDD ordena igual que int)
    fecha_venta_int = pd.to_numeric(vd["fecha_venta"], errors="coerce")
    fecha_si_promo = fecha_venta_int.where(vd["tiene_promo_flag"] == 1)
    ultima_fecha_promo = (
        fecha_si_promo.groupby(keys).cummax()
        .groupby(keys).ffill()
        .groupby(keys).shift(1)
    )

    fecha_actual_dt = pd.to_datetime(fecha_venta_int.astype("Int64").astype(str), format="%Y%m%d")
    fecha_ultima_promo_dt = pd.to_datetime(ultima_fecha_promo.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    vd["dias_desde_ultima_promo"] = (fecha_actual_dt - fecha_ultima_promo_dt).dt.days

    pct_si_promo = vd["pct_descuento_calc"].where(vd["tiene_promo_flag"] == 1)
    vd["avg_discount_historico"] = _expanding_lag(pct_si_promo, "mean")

    vd["num_promos_previas"] = _expanding_lag(es_promo_flag, "sum").fillna(0).astype("int64")

    COLS_NUEVAS = [
        "precio_efectivo", "costo_unitario", "pct_descuento_calc", "margin_bruto", "pct_margen",
        "avg_units_4w", "avg_units_8w", "avg_units_12w", "coef_variacion", "flag_baseline_insuficiente",
        "frecuencia_promo_90d", "dias_desde_ultima_promo", "avg_discount_historico", "num_promos_previas",
    ]

    return vd[list(df_vd.columns) + COLS_NUEVAS].reset_index(drop=True)


def train_set_final(df_vcf: pd.DataFrame) -> pd.DataFrame:
    return df_vcf[df_vcf["flag_baseline_insuficiente"] == 0].reset_index(drop=True)