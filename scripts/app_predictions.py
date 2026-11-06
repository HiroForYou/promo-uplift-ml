"""
Dashboard de Recomendaciones Promocionales — M1 Demand Response
Réplica de la vista principal "Recomendaciones" del dashboard Power BI.

Uso:
    streamlit run scripts/app_predictions.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT    = Path(__file__).resolve().parent.parent
STAGING = ROOT / "reports" / "scoring" / "staging_output.parquet"

NAVY    = "#1C3461"
GREEN   = "#1A5C35"
AMBER   = "#905100"
RED     = "#882412"
SURFACE = "#F3F2EE"

SEMAFORO_COLOR = {"GO": GREEN,     "CONSIDER": AMBER,     "AVOID": RED}
SEMAFORO_BG    = {"GO": "#D4EDDA", "CONSIDER": "#FFF3CD", "AVOID": "#F8D7DA"}
SEMAFORO_EMOJI = {"GO": "🟢",      "CONSIDER": "🟡",      "AVOID": "🔴"}

st.set_page_config(
    page_title="Recomendaciones Promocionales — M1",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .kpi-card {
    background:#fff; border:1px solid #CCCAC2; border-radius:4px;
    padding:16px 20px; text-align:center; border-top:4px solid;
  }
  .kpi-label { font-size:10px; font-weight:700; letter-spacing:.10em;
               text-transform:uppercase; color:#6D6C65; margin-bottom:6px; }
  .kpi-value { font-size:32px; font-weight:800; line-height:1; }
  .kpi-pct   { font-size:13px; color:#6D6C65; margin-top:4px; }
  .section-title { font-size:13px; font-weight:700; margin:18px 0 8px;
                   color:#1A1A17; border-bottom:2px solid #E0DED9; padding-bottom:6px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Cargando predicciones…")
def load_data() -> pd.DataFrame:
    if not STAGING.exists():
        st.error(f"No se encontró: `{STAGING.relative_to(ROOT)}`\n\nEjecuta primero: `python src/staging.py`")
        st.stop()
    df = pd.read_parquet(STAGING)
    df["tienda_id"] = df["tienda_id"].astype(str)
    df["generico_id"] = df["generico_id"].astype(str)
    df["recommendation_semaforo"] = df["recommendation_semaforo"].str.upper()
    df["scoring_date"] = pd.to_datetime(df["scoring_date"])
    order = {"GO": 0, "CONSIDER": 1, "AVOID": 2}
    df["_order"] = df.recommendation_semaforo.map(order)
    df = df.sort_values("_order").drop(columns="_order")
    return df


df_all = load_data()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="background:{NAVY};padding:18px 24px 14px;border-radius:4px;margin-bottom:20px">'
    f'<div style="color:rgba(255,255,255,.5);font-size:10px;font-weight:700;letter-spacing:.14em;'
    f'text-transform:uppercase;margin-bottom:4px">M1 Demand Response &middot; Precisión Promocional</div>'
    f'<div style="color:#fff;font-size:20px;font-weight:700">📊 Dashboard de Recomendaciones Promocionales</div>'
    f'<div style="color:rgba(255,255,255,.55);font-size:12px;margin-top:4px">'
    f'Semáforo GO / CONSIDER / AVOID · uplift, confianza y margen por combo tienda × genérico</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="color:{NAVY};font-weight:700;font-size:13px;margin-bottom:10px">🔎 Filtros</div>',
                unsafe_allow_html=True)

    sem_opts = st.multiselect(
        "Semáforo",
        ["GO", "CONSIDER", "AVOID"],
        default=["GO", "CONSIDER", "AVOID"],
    )
    div_opts = st.multiselect(
        "División",
        sorted(df_all.division_desc.dropna().unique()),
        default=sorted(df_all.division_desc.dropna().unique()),
    )
    ban_opts = st.multiselect(
        "Banner",
        sorted(df_all.banner.dropna().unique()),
        default=sorted(df_all.banner.dropna().unique()),
    )

    with st.expander("Filtros avanzados"):
        reg_opts   = st.multiselect("Región", sorted(df_all.region_desc.dropna().unique()),
                                    default=sorted(df_all.region_desc.dropna().unique()))
        marca_opts = st.multiselect("Marca", sorted(df_all.marca_desc.dropna().unique()),
                                    default=sorted(df_all.marca_desc.dropna().unique()))
        min_conf   = st.slider("Confidence score mínimo", 0.0, 1.0, 0.0, 0.05,
                               help="cobertura_historica × (1 − fraccion_extrapolada)")
        min_uplift = st.slider("Uplift mínimo (%)", -50, 100, -50, 5)

    st.markdown("---")
    st.caption(f"Scoring date: {df_all.scoring_date.iloc[0].date()}")
    st.caption(f"Total combos: {len(df_all):,}")

# ── Filtrar ────────────────────────────────────────────────────────────────────
df = df_all[
    df_all.recommendation_semaforo.isin(sem_opts) &
    df_all.division_desc.isin(div_opts) &
    df_all.banner.isin(ban_opts) &
    df_all.region_desc.isin(reg_opts) &
    df_all.marca_desc.isin(marca_opts) &
    (df_all.confidence_score >= min_conf) &
    (df_all.uplift_pct >= min_uplift)
].copy()

if df.empty:
    st.warning("No hay combos que cumplan los filtros seleccionados.")
    st.stop()

# ── KPI cards ─────────────────────────────────────────────────────────────────
total  = len(df)
counts = df.recommendation_semaforo.value_counts()

c_go, c_co, c_av, c_tot = st.columns(4)
for col, sem in [(c_go, "GO"), (c_co, "CONSIDER"), (c_av, "AVOID")]:
    n   = counts.get(sem, 0)
    pct = 100 * n / total if total else 0
    col.markdown(
        f'<div class="kpi-card" style="border-top-color:{SEMAFORO_COLOR[sem]}">'
        f'<div class="kpi-label">{SEMAFORO_EMOJI[sem]} {sem}</div>'
        f'<div class="kpi-value" style="color:{SEMAFORO_COLOR[sem]}">{n:,}</div>'
        f'<div class="kpi-pct">{pct:.1f}% de combos</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
c_tot.markdown(
    f'<div class="kpi-card" style="border-top-color:{NAVY}">'
    f'<div class="kpi-label">Total combos</div>'
    f'<div class="kpi-value" style="color:{NAVY}">{total:,}</div>'
    f'<div class="kpi-pct">Uplift medio: {df.uplift_pct.mean():.1f}%</div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Gráficos: 2 columnas ───────────────────────────────────────────────────────
SEM_ORDER   = ["GO", "CONSIDER", "AVOID"]
SEM_COLORS  = {"GO": GREEN, "CONSIDER": AMBER, "AVOID": RED}
CHART_STYLE = dict(plot_bgcolor="#FEFEFE", paper_bgcolor="#FEFEFE",
                   font=dict(size=11, color="#333"))

col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="section-title">Semáforo por División</div>', unsafe_allow_html=True)
    div_sem = (
        df.groupby(["division_desc", "recommendation_semaforo"])
        .size().reset_index(name="n")
    )
    fig1 = px.bar(
        div_sem, x="division_desc", y="n", color="recommendation_semaforo",
        color_discrete_map=SEM_COLORS,
        category_orders={"recommendation_semaforo": SEM_ORDER},
        labels={"division_desc": "", "n": "Combos", "recommendation_semaforo": ""},
        text="n",
    )
    fig1.update_traces(textposition="inside", textfont_size=11)
    fig1.update_layout(
        **CHART_STYLE, barmode="stack",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0, font_size=11),
        margin=dict(l=40, r=10, t=10, b=70), height=300,
        xaxis=dict(gridcolor="#E8E7E3", tickfont=dict(size=11)),
        yaxis=dict(gridcolor="#E8E7E3", title="Combos", tickfont=dict(size=11)),
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    st.markdown('<div class="section-title">Semáforo por Banner</div>', unsafe_allow_html=True)
    ban_sem = (
        df.groupby(["banner", "recommendation_semaforo"])
        .size().reset_index(name="n")
    )
    fig2 = px.bar(
        ban_sem, x="n", y="banner", color="recommendation_semaforo",
        color_discrete_map=SEM_COLORS, orientation="h",
        category_orders={"recommendation_semaforo": SEM_ORDER},
        labels={"banner": "", "n": "Combos", "recommendation_semaforo": ""},
    )
    fig2.update_layout(
        **CHART_STYLE, barmode="stack", showlegend=False,
        margin=dict(l=60, r=10, t=10, b=20), height=300,
        xaxis=dict(gridcolor="#E8E7E3", title="Combos", tickfont=dict(size=11)),
        yaxis=dict(gridcolor="#E8E7E3", tickfont=dict(size=11)),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Tabs: tabla completa / top oportunidades ───────────────────────────────────
tab_tabla, tab_top = st.tabs(["📋 Tabla de recomendaciones", "🏆 Top 10 oportunidades"])

DISPLAY_COLS = {
    "recommendation_semaforo": "Semáforo",
    "tienda_id": "Tienda",
    "generico_id": "Genérico",
    "division_desc": "División",
    "marca_desc": "Marca",
    "banner": "Banner",
    "region_desc": "Región",
    "uplift_pct": "Uplift %",
    "confidence_score": "Confianza",
    "margin_roi": "Margin ROI",
    "net_incremental_margin": "NIM (MXN)",
    "ingreso_incremental_pesos": "Ingreso Incr. (MXN)",
    "is_extrapolated": "Extrapolado",
    "recommendation_reason": "Razón",
}

NUM_FORMATS = {
    "Uplift %": "{:.1f}%",
    "Confianza": "{:.2f}",
    "Margin ROI": "{:.2f}",
    "NIM (MXN)": "${:,.0f}",
    "Ingreso Incr. (MXN)": "${:,.0f}",
}

TABLE_STYLE = [
    {"selector": "thead th", "props": [
        ("background-color", NAVY), ("color", "#fff"),
        ("font-size", "11px"), ("font-weight", "700"),
        ("text-align", "center"), ("white-space", "nowrap"),
    ]},
    {"selector": "td", "props": [("white-space", "nowrap"), ("font-size", "11px")]},
]


def style_semaforo(val):
    color = SEMAFORO_COLOR.get(str(val), "#000")
    bg    = SEMAFORO_BG.get(str(val), "#fff")
    return f"background-color:{bg};color:{color};font-weight:700;text-align:center"

def style_extrap(val):
    if val is True or val == "True":
        return f"background-color:#FFF3CD;color:{AMBER}"
    return ""


with tab_tabla:
    tab_cols = [c for c in DISPLAY_COLS if c in df.columns]
    tbl = df[tab_cols].rename(columns=DISPLAY_COLS).copy()
    for col, fmt in NUM_FORMATS.items():
        if col in tbl.columns:
            tbl[col] = tbl[col].apply(lambda v: fmt.format(v) if pd.notna(v) else "—")

    styled = tbl.style.map(style_semaforo, subset=["Semáforo"]).set_table_styles(TABLE_STYLE)
    if "Extrapolado" in tbl.columns:
        styled = styled.map(style_extrap, subset=["Extrapolado"])
    st.dataframe(styled, use_container_width=True, height=420)

    st.caption(f"{total:,} combos · scoring date {df_all.scoring_date.iloc[0].date()} · BBP v1.4 §14")


with tab_top:
    top_cols = ["recommendation_semaforo", "tienda_id", "generico_id",
                "division_desc", "marca_desc", "banner",
                "uplift_pct", "confidence_score", "margin_roi",
                "net_incremental_margin", "ingreso_incremental_pesos"]
    top = (
        df[df.recommendation_semaforo.isin(["GO", "CONSIDER"])]
        .nlargest(10, "margin_roi")
        [[c for c in top_cols if c in df.columns]]
        .rename(columns={
            "recommendation_semaforo": "Semáforo",
            "tienda_id": "Tienda", "generico_id": "Genérico",
            "division_desc": "División", "marca_desc": "Marca", "banner": "Banner",
            "uplift_pct": "Uplift %", "confidence_score": "Confianza",
            "margin_roi": "ROI", "net_incremental_margin": "NIM (MXN)",
            "ingreso_incremental_pesos": "Ingreso Incr. (MXN)",
        })
    )

    if top.empty:
        st.info("No hay combos GO o CONSIDER con los filtros actuales.")
    else:
        st.dataframe(
            top.style
            .map(style_semaforo, subset=["Semáforo"])
            .format({"Uplift %": "{:.1f}%", "Confianza": "{:.2f}",
                     "ROI": "{:.2f}", "NIM (MXN)": "${:,.0f}", "Ingreso Incr. (MXN)": "${:,.0f}"})
            .set_table_styles(TABLE_STYLE),
            use_container_width=True,
        )
        st.caption("Top 10 combos GO + CONSIDER ordenados por Margin ROI descendente.")
