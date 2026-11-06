"""
Simulador de Curvas de Elasticidad — M1 Demand Response
Réplica de la vista "Simulador de Escenarios" del dashboard Power BI.

Uso:
    streamlit run scripts/app_elasticity.py
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "reports" / "elasticity" / "promo_elasticity_curves.parquet"

# ── Colores de marca ───────────────────────────────────────────────────────────
NAVY      = "#1C3461"
GREEN     = "#1A5C35"
AMBER     = "#905100"
RED_DARK  = "#882412"
ORANGE    = "#D86C00"
SURFACE   = "#F3F2EE"
BORDER    = "#CCCAC2"

PALETTE = [
    "#1C3461", "#882412", "#1A5C35", "#905100",
    "#5C2D91", "#006B75", "#8B4513", "#2F4F4F",
]

st.set_page_config(
    page_title="Simulador de Elasticidad — M1",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .metric-card {
    background: #fff; border: 1px solid #CCCAC2; border-top: 3px solid #1C3461;
    border-radius: 4px; padding: 14px 18px; text-align: center;
  }
  .metric-label { font-size: 11px; font-weight: 700; letter-spacing: .08em;
                  text-transform: uppercase; color: #6D6C65; margin-bottom: 4px; }
  .metric-value { font-size: 26px; font-weight: 700; color: #1A1A17; line-height: 1.1; }
  .metric-sub   { font-size: 11px; color: #6D6C65; margin-top: 3px; }
  .extrap-badge { background: #FFF3CD; border: 1px solid #D86C00; border-radius: 3px;
                  color: #905100; font-size: 11px; font-weight: 600;
                  padding: 3px 8px; display: inline-block; }
  .tag-mono { font-family: monospace; font-size: 11px; background: #F3F2EE;
              border: 1px solid #CCCAC2; border-radius: 3px; padding: 2px 6px; }
</style>
""", unsafe_allow_html=True)


# ── Carga de datos ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando curvas de elasticidad…")
def load_data() -> pd.DataFrame:
    if not PARQUET.exists():
        st.error(f"No se encontró el archivo: `{PARQUET.relative_to(ROOT)}`\n\n"
                 "Ejecuta primero: `python src/elasticity.py`")
        st.stop()
    df = pd.read_parquet(PARQUET)
    df["division_desc"] = df["division_desc"].astype(str)
    # Calcular uplift relativo al baseline (discount_pct = 0)
    baseline = (
        df[df.discount_pct == 0]
        [["tienda_id", "generico_id", "nombre_promo", "predicted_units"]]
        .rename(columns={"predicted_units": "baseline_units"})
    )
    df = df.merge(baseline, on=["tienda_id", "generico_id", "nombre_promo"], how="left")
    df["uplift_pct"] = (
        (df.predicted_units - df.baseline_units) / df.baseline_units.replace(0, float("nan")) * 100
    ).round(1)
    return df


df = load_data()

# ── Encabezado ─────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="background:{NAVY};padding:20px 24px 16px;border-radius:4px;margin-bottom:20px">'
    f'<div style="color:rgba(255,255,255,.55);font-size:10px;font-weight:700;'
    f'letter-spacing:.14em;text-transform:uppercase;margin-bottom:6px">'
    f'M1 Demand Response &middot; Precisión Promocional</div>'
    f'<div style="color:#fff;font-size:20px;font-weight:700">📈 Simulador de Curvas de Elasticidad</div>'
    f'<div style="color:rgba(255,255,255,.6);font-size:12px;margin-top:5px">'
    f'Replica la vista "Simulador de Escenarios" del dashboard Power BI &mdash; '
    f'curvas uplift × nivel de descuento por mecánica</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Sidebar: filtros ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="color:{NAVY};font-weight:700;font-size:13px;margin-bottom:8px">🔍 Seleccionar combo</div>',
                unsafe_allow_html=True)

    divisions = sorted(df.division_desc.unique())
    div_sel = st.selectbox("División", ["Todas"] + divisions)

    df_filt = df if div_sel == "Todas" else df[df.division_desc == div_sel]

    combos = (
        df_filt[["tienda_id", "generico_id"]]
        .drop_duplicates()
        .assign(label=lambda d: "Tienda " + d.tienda_id.astype(str) +
                                " × Genérico " + d.generico_id.astype(str))
        .sort_values(["tienda_id", "generico_id"])
    )

    combo_label = st.selectbox(
        f"Combo ({len(combos):,} disponibles)",
        combos.label.tolist(),
    )
    row = combos[combos.label == combo_label].iloc[0]
    tienda_sel  = int(row.tienda_id)
    generico_sel = int(row.generico_id)

    st.markdown("---")

    # Mecánicas disponibles para este combo
    mecanicas_combo = sorted(
        df[(df.tienda_id == tienda_sel) & (df.generico_id == generico_sel)]
        .nombre_promo.unique()
    )
    mec_sel = st.multiselect(
        f"Mecánicas ({len(mecanicas_combo)} disponibles)",
        mecanicas_combo,
        default=mecanicas_combo,
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;color:#6D6C65">'
        '💡 <b>is_extrapolated = True</b>: el descuento simulado supera el máximo histórico '
        'observado. Se muestra con símbolo ◇ y fondo naranja en Power BI. '
        'Nunca activa semáforo GO automáticamente.</div>',
        unsafe_allow_html=True,
    )

# ── Datos del combo seleccionado ───────────────────────────────────────────────
combo_data = df[
    (df.tienda_id == tienda_sel) &
    (df.generico_id == generico_sel) &
    (df.nombre_promo.isin(mec_sel))
].copy()

if combo_data.empty:
    st.warning("No hay curvas de elasticidad para esta selección.")
    st.stop()

division_combo = combo_data.division_desc.iloc[0]
n_mec = combo_data.nombre_promo.nunique()
has_extrap = combo_data.is_extrapolated.any()
max_uplift = combo_data.uplift_pct.max()

# ── Métricas resumen ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Tienda</div>'
        f'<div class="metric-value">{tienda_sel}</div>'
        f'<div class="metric-sub">División: {division_combo}</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Genérico</div>'
        f'<div class="metric-value">{generico_sel}</div>'
        f'<div class="metric-sub">Mecánicas: {n_mec}</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    color_uplift = GREEN if max_uplift >= 15 else (AMBER if max_uplift >= 5 else RED_DARK)
    st.markdown(
        f'<div class="metric-card" style="border-top-color:{color_uplift}">'
        f'<div class="metric-label">Máx. Uplift</div>'
        f'<div class="metric-value" style="color:{color_uplift}">+{max_uplift:.1f}%</div>'
        f'<div class="metric-sub">en algún nivel de descuento</div></div>',
        unsafe_allow_html=True,
    )
with c4:
    extrap_label = "⚠️ Sí" if has_extrap else "✅ No"
    extrap_color = ORANGE if has_extrap else GREEN
    st.markdown(
        f'<div class="metric-card" style="border-top-color:{extrap_color}">'
        f'<div class="metric-label">Extrapolado</div>'
        f'<div class="metric-value" style="color:{extrap_color};font-size:20px">{extrap_label}</div>'
        f'<div class="metric-sub">algún punto fuera del rango histórico</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Gráfico: curvas de uplift ──────────────────────────────────────────────────
col_chart, col_table = st.columns([3, 2])

with col_chart:
    st.markdown(f'<div style="font-weight:700;color:{NAVY};margin-bottom:6px">Curva de Uplift por Mecánica</div>',
                unsafe_allow_html=True)

    fig = go.Figure()

    for idx, mec in enumerate(sorted(combo_data.nombre_promo.unique())):
        mec_data = combo_data[combo_data.nombre_promo == mec].sort_values("discount_pct")
        color = PALETTE[idx % len(PALETTE)]

        # Puntos no extrapolados (línea sólida)
        solid = mec_data[~mec_data.is_extrapolated]
        if not solid.empty:
            fig.add_trace(go.Scatter(
                x=solid.discount_pct, y=solid.uplift_pct,
                mode="lines+markers",
                name=mec[:35],
                line=dict(color=color, width=2.5),
                marker=dict(size=7, symbol="circle"),
                hovertemplate=(
                    f"<b>{mec[:30]}</b><br>"
                    "Descuento: %{x}%<br>"
                    "Uplift: <b>%{y:.1f}%</b><br>"
                    "<extra></extra>"
                ),
            ))

        # Puntos extrapolados (marcador distinto + trazo discontinuo)
        extrap = mec_data[mec_data.is_extrapolated]
        if not extrap.empty:
            # Conectar el último punto sólido con los extrapolados
            if not solid.empty:
                connector = pd.concat([solid.tail(1), extrap])
            else:
                connector = extrap
            fig.add_trace(go.Scatter(
                x=connector.discount_pct, y=connector.uplift_pct,
                mode="lines", showlegend=False,
                line=dict(color=color, width=1.5, dash="dot"),
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=extrap.discount_pct, y=extrap.uplift_pct,
                mode="markers",
                name=f"{mec[:25]} (extrapol.)",
                marker=dict(size=9, symbol="diamond-open", color=ORANGE, line=dict(color=ORANGE, width=2)),
                hovertemplate=(
                    f"<b>⚠️ EXTRAPOLADO</b><br>{mec[:30]}<br>"
                    "Descuento: %{x}%<br>Uplift: <b>%{y:.1f}%</b><br>"
                    "<extra></extra>"
                ),
            ))

    # Líneas de referencia del semáforo
    fig.add_hline(y=15, line=dict(color=GREEN, width=1, dash="dash"),
                  annotation_text="GO ≥15%", annotation_font_color=GREEN,
                  annotation_position="right")
    fig.add_hline(y=5, line=dict(color=AMBER, width=1, dash="dash"),
                  annotation_text="CONSIDER ≥5%", annotation_font_color=AMBER,
                  annotation_position="right")
    fig.add_hline(y=0, line=dict(color="#999", width=1))

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Nivel de descuento (%)", font=dict(size=12, color="#333")),
            tickvals=[0, 5, 10, 15, 20, 25, 30, 40, 50],
            ticktext=["0%", "5%", "10%", "15%", "20%", "25%", "30%", "40%", "50%"],
            tickfont=dict(size=11, color="#333"),
            gridcolor="#E8E7E3",
            linecolor="#CCCAC2",
        ),
        yaxis=dict(
            title=dict(text="Uplift en unidades (%)", font=dict(size=12, color="#333")),
            tickformat=".0f",
            ticksuffix="%",
            tickfont=dict(size=11, color="#333"),
            gridcolor="#E8E7E3",
            linecolor="#CCCAC2",
        ),
        plot_bgcolor="#FEFEFE", paper_bgcolor="#FEFEFE",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="left", x=0,
                    font=dict(size=10)),
        margin=dict(l=60, r=90, t=15, b=90),
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div style="font-size:10px;color:#6D6C65;margin-top:-8px">'
        '◇ Puntos extrapolados = descuento fuera del rango histórico observado. '
        'Semáforo máximo: CONSIDER. &nbsp;|&nbsp; '
        'Líneas de referencia: GO ≥15% · CONSIDER ≥5%.</div>',
        unsafe_allow_html=True,
    )

# ── Tabla por nivel de descuento ───────────────────────────────────────────────
with col_table:
    st.markdown(f'<div style="font-weight:700;color:{NAVY};margin-bottom:6px">Uplift por nivel de descuento</div>',
                unsafe_allow_html=True)

    pivot = (
        combo_data
        .groupby(["discount_pct", "nombre_promo"])
        .agg(uplift_pct=("uplift_pct", "first"), is_extrapolated=("is_extrapolated", "first"))
        .reset_index()
        .pivot(index="discount_pct", columns="nombre_promo", values="uplift_pct")
        .reset_index()
        .rename(columns={"discount_pct": "Descuento %"})
    )

    def color_cell(val):
        if pd.isna(val):
            return ""
        if val >= 15:
            return f"background-color:#D4EDDA;color:{GREEN};font-weight:700"
        if val >= 5:
            return f"background-color:#FFF3CD;color:{AMBER};font-weight:600"
        if val < 0:
            return f"background-color:#F8D7DA;color:{RED_DARK}"
        return ""

    numeric_cols = [c for c in pivot.columns if c != "Descuento %"]
    styled = (
        pivot.style
        .format({"Descuento %": lambda v: f"{v}%"})
        .format({c: lambda v: f"+{v:.1f}%" if v > 0 else f"{v:.1f}%" for c in numeric_cols},
                na_rep="—")
        .map(color_cell, subset=numeric_cols)
        .set_properties(**{"font-size": "11px", "text-align": "center"})
        .set_table_styles([
            {"selector": "thead th", "props": [
                ("background-color", NAVY), ("color", "#fff"),
                ("font-size", "10px"), ("text-align", "center"),
            ]},
        ])
    )
    st.dataframe(styled, use_container_width=True, height=360)

    # Extrapolated indicator
    extrap_by_mec = (
        combo_data[combo_data.is_extrapolated]
        .groupby("nombre_promo")["discount_pct"]
        .min()
    )
    if not extrap_by_mec.empty:
        st.markdown('<div class="extrap-badge">⚠️ Extrapolación desde:</div>', unsafe_allow_html=True)
        for mec, from_pct in extrap_by_mec.items():
            st.markdown(
                f'<div style="font-size:11px;margin-top:4px">'
                f'<span class="tag-mono">{mec[:35]}</span> → ≥{from_pct}%</div>',
                unsafe_allow_html=True,
            )

# ── Tabla detallada raw ────────────────────────────────────────────────────────
with st.expander("Ver datos crudos del parquet"):
    show = combo_data[["nombre_promo", "discount_pct", "predicted_units",
                        "baseline_units", "uplift_pct", "is_extrapolated",
                        "is_monotone_adjusted"]].sort_values(["nombre_promo", "discount_pct"])
    st.dataframe(show.reset_index(drop=True), use_container_width=True)
    st.caption(f"Fuente: `{PARQUET.relative_to(ROOT)}` — {len(df):,} filas totales · "
               f"{df.groupby(['tienda_id','generico_id']).ngroups:,} combos únicos")
