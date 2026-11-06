"""
build_diagrams.py — Genera los diagramas SVG del README (EN y ES).

    python scripts/build_diagrams.py

Salida: docs/images/<nombre>.svg (EN) y docs/images/<nombre>.es.svg (ES).
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"

INK, RUST, MUTED = "#14212B", "#A8420E", "#8B949B"

HEAD = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="Inter, system-ui, -apple-system, 'Segoe UI', sans-serif">
  <style>
    .flow {{ stroke-dasharray: 6 6; animation: flow 1.2s linear infinite; }}
    @keyframes flow {{ to {{ stroke-dashoffset: -24; }} }}
    .box {{ fill: #F4F6F7; stroke: #CBD0D3; stroke-width: 1.5; }}
    .title {{ fill: #14212B; font-family: Fraunces, Georgia, 'Times New Roman', serif; font-size: 17px; font-weight: 600; }}
    .sub {{ fill: #4A5963; font-size: 11.5px; }}
    .label {{ fill: #4A5963; font-size: 11.5px; font-style: italic; }}
    .mono {{ font-family: ui-monospace, 'JetBrains Mono', Consolas, monospace; }}
  </style>
  <rect width="{w}" height="{h}" fill="#E6E9EA"/>
  <defs>
    <marker id="ink" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="{ink}"/></marker>
    <marker id="rust" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="{rust}"/></marker>
    <marker id="muted" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="{muted}"/></marker>
  </defs>
"""


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x, y, w, h, title, *subs, dark=False, accent=False, mono=()):
    """Caja con título y hasta 3 líneas de subtítulo. mono: índices de subs en monoespaciada."""
    cx = x + w / 2
    n = len(subs)
    ty = y + h / 2 - 7 * n + 6
    if dark:
        out = f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{INK}"/>\n'
        out += f'  <text class="title" x="{cx}" y="{ty}" text-anchor="middle" style="fill:#FFFFFF">{esc(title)}</text>\n'
        for i, s in enumerate(subs):
            cls = ' class="mono"' if i in mono else ""
            out += f'  <text{cls} x="{cx}" y="{ty + 20 + 15 * i}" text-anchor="middle" font-size="11.5" fill="#CBD0D3">{esc(s)}</text>\n'
        return out
    stroke = f' stroke="{RUST}"' if accent else ""
    out = f'  <rect class="box" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"{stroke}/>\n'
    out += f'  <text class="title" x="{cx}" y="{ty}" text-anchor="middle">{esc(title)}</text>\n'
    for i, s in enumerate(subs):
        cls = "sub mono" if i in mono else "sub"
        out += f'  <text class="{cls}" x="{cx}" y="{ty + 20 + 15 * i}" text-anchor="middle">{esc(s)}</text>\n'
    return out


def arrow(d, color="ink", width=2):
    c = {"ink": INK, "rust": RUST, "muted": MUTED}[color]
    return f'  <path class="flow" d="{d}" stroke="{c}" stroke-width="{width}" fill="none" marker-end="url(#{color})"/>\n'


def label(x, y, text, anchor="middle"):
    return f'  <text class="label" x="{x}" y="{y}" text-anchor="{anchor}">{esc(text)}</text>\n'


def caption(x, y, text):
    return f'  <text x="{x}" y="{y}" fill="#4A5963" font-size="12.5" text-anchor="middle">{esc(text)}</text>\n'


# ── 1. Arquitectura ───────────────────────────────────────────────────────────
def architecture(t):
    w, h = 1100, 540
    s = HEAD.format(w=w, h=h, ink=INK, rust=RUST, muted=MUTED)
    s += box(20, 200, 145, 70, t["raw"], t["raw_s1"], t["raw_s2"])
    s += box(200, 200, 150, 70, "deploy/ETL", "promo-uplift-etl", t["etl_s"], mono=(0,))
    s += box(385, 200, 160, 70, t["feat"], "train-set-features.csv", t["feat_s"], mono=(0,))
    s += box(590, 60, 270, 80, t["train"], "tune → train → cv ‖ explain", t["train_s"], mono=(0,))
    s += box(910, 60, 170, 80, "models/", t["model_s1"], t["model_s2"], dark=True, mono=(1,))
    s += box(590, 330, 270, 80, t["score"], "elasticity ‖ score → staging", "→ dryrun → publish", mono=(0, 1))
    s += f'  <text class="sub" x="725" y="428" text-anchor="middle">{esc(t["score_s"])}</text>\n'
    s += box(910, 330, 170, 80, "S3 publish/", "PROMO_PREDICTIONS", "PROMO_ELASTICITY_CURVES", accent=True, mono=(0, 1))
    s += box(910, 450, 170, 62, "Snowflake", t["bi_s"])

    s += arrow("M165,235 H200")
    s += arrow("M350,235 H385")
    s += label(275, 290, t["l_months"])
    s += arrow("M545,222 C567,222 567,100 590,100")
    s += arrow("M545,248 C567,248 567,370 590,370")
    s += label(560, 180, t["l_daily"], "end")
    s += arrow("M860,100 H910")
    s += arrow("M995,140 C995,260 900,300 860,345")
    s += label(975, 250, t["l_load"], "end")
    s += arrow("M860,385 H910", "rust")
    s += arrow("M995,410 V450", "rust")
    s += label(1003, 438, "DataOps MERGE", "start")
    s += arrow("M620,330 V140", "muted", 1.5)
    s += label(628, 200, t["l_fail"], "start")
    s += arrow("M830,140 V330", "muted", 1.5)
    s += label(822, 280, t["l_after"], "end")
    s += caption(w / 2 - 20, 528, t["cap_a"])
    return s + "</svg>\n"


# ── 2. Pipeline de scoring y gate ─────────────────────────────────────────────
def scoring_gate(t):
    w, h = 1000, 420
    s = HEAD.format(w=w, h=h, ink=INK, rust=RUST, muted=MUTED)
    s += f'  <rect class="box" x="20" y="70" width="150" height="250" rx="6"/>\n'
    s += f'  <text class="title" x="95" y="100" text-anchor="middle">{esc(t["inputs"])}</text>\n'
    s += f'  <text class="sub mono" x="95" y="132" text-anchor="middle">train-set-features</text>\n'
    s += f'  <text class="sub" x="95" y="148" text-anchor="middle">{esc(t["in_feat"])}</text>\n'
    s += f'  <text class="sub mono" x="95" y="282" text-anchor="middle">xgb_base.pkl</text>\n'
    s += f'  <text class="sub" x="95" y="298" text-anchor="middle">{esc(t["in_model"])}</text>\n'
    s += box(215, 70, 165, 64, "elasticity.py", t["el_s"])
    s += box(215, 250, 165, 64, "score.py", t["sc_s"])
    s += box(425, 250, 160, 64, "staging.py", t["st_s"])
    s += box(425, 150, 160, 50, "cv_report.json", t["cv_s"])
    s += box(630, 155, 150, 95, "dryrun.py", t["dr_s1"], t["dr_s2"], dark=True)
    s += box(825, 70, 160, 80, "publish/", t["pub_s1"], t["pub_s2"], accent=True)
    s += box(825, 260, 160, 80, t["fail"], t["fail_s1"], t["fail_s2"])

    s += arrow("M170,102 H215")
    s += arrow("M170,282 H215")
    s += arrow("M380,282 H425")
    s += arrow("M380,102 C600,102 605,180 630,180")
    s += label(470, 92, t["l_curves"])
    s += arrow("M585,282 C610,282 610,230 630,230")
    s += arrow("M585,175 C605,175 610,205 630,205", "muted", 1.5)
    s += arrow("M780,185 C805,185 805,110 825,110", "rust")
    s += label(808, 108, t["l_ok"], "end")
    s += arrow("M780,225 C805,225 805,300 825,300", "muted", 1.5)
    s += label(812, 268, t["l_ko"], "end")
    s += caption(w / 2, 395, t["cap_s"])
    return s + "</svg>\n"


# ── 3. Semáforo ───────────────────────────────────────────────────────────────
def recommendation(t):
    w, h = 1100, 340
    s = HEAD.format(w=w, h=h, ink=INK, rust=RUST, muted=MUTED)
    s += box(20, 135, 155, 70, t["combo"], t["combo_s"])
    s += box(205, 135, 140, 70, "XGBoost", "uplift_pct", dark=True, mono=(0,))
    s += box(375, 135, 160, 70, t["thr"], "≥ 15% GO · ≥ 5% CONSIDER")
    s += box(565, 135, 160, 70, t["r1"], t["r1_s"])
    s += box(755, 135, 165, 70, t["r2"], t["r2_s"])
    s += box(955, 40, 125, 60, "GO", t["go_s"], dark=True)
    s += box(955, 140, 125, 60, "CONSIDER", t["co_s"], accent=True)
    s += box(955, 240, 125, 60, "AVOID", t["av_s"])

    s += arrow("M175,170 H205")
    s += arrow("M345,170 H375")
    s += arrow("M535,170 H565")
    s += arrow("M725,170 H755")
    s += arrow("M920,160 C940,160 935,70 955,70")
    s += arrow("M920,170 H955", "rust")
    s += arrow("M920,180 C940,180 935,270 955,270", "muted", 1.5)
    s += label(645, 120, t["l_r1"])
    s += label(837, 120, t["l_r2"])
    s += caption(w / 2, 325, t["cap_r"])
    return s + "</svg>\n"


TEXT = {
    "en": {
        "raw": "SAP raw data", "raw_s1": "S3 Parquet", "raw_s2": "sales · prices · stock",
        "etl_s": "19 CTEs, monthly", "feat": "Feature set", "feat_s": "store × generic × day",
        "train": "Training pipeline", "train_s": "on demand · auto on gate fail",
        "model_s1": "champion model", "model_s2": "xgb_base.pkl",
        "score": "Scoring pipeline", "score_s": "weekly · Monday 02:00 CST",
        "bi_s": "read by Power BI",
        "l_months": "cached partitions", "l_daily": "features", "l_load": "champion",
        "l_fail": "gate failed → retrain", "l_after": "training done → rescore",
        "cap_a": "Two SageMaker pipelines share one model store. Nothing reaches Snowflake unless the dry run gate passes.",
        "inputs": "Inputs", "in_feat": "latest snapshot per combo", "in_model": "champion",
        "el_s": "curves · 0-50%, 9 points", "sc_s": "uplift · confidence · light",
        "st_s": "validations · margin ROI", "cv_s": "from training",
        "dr_s1": "12 cross-artifact", "dr_s2": "checks",
        "pub_s1": "predictions + curves", "pub_s2": "→ DataOps → Snowflake",
        "fail": "RuntimeError", "fail_s1": "publish/ unchanged", "fail_s2": "retraining triggered",
        "l_curves": "elasticity curves", "l_ok": "approved", "l_ko": "any check fails",
        "cap_s": "dryrun.py is the only step that sees every artifact. publish.py copies files only when it approves.",
        "combo": "Combo", "combo_s": "store · generic · discount",
        "thr": "Base light", "r1": "Extrapolated?", "r1_s": "discount > hist. max",
        "r2": "Thin baseline?", "r2_s": "too little history",
        "go_s": "run the promo", "co_s": "review first", "av_s": "skip it",
        "l_r1": "GO capped at CONSIDER", "l_r2": "forced to AVOID",
        "cap_r": "The model sets the base light from uplift. The two guard rules can only lower it. Thresholds live in config.py.",
    },
    "es": {
        "raw": "Datos SAP", "raw_s1": "S3 Parquet", "raw_s2": "ventas · precios · stock",
        "etl_s": "19 CTEs, por mes", "feat": "Dataset", "feat_s": "tienda × genérico × día",
        "train": "Pipeline training", "train_s": "bajo demanda · auto si falla el gate",
        "model_s1": "modelo champion", "model_s2": "xgb_base.pkl",
        "score": "Pipeline scoring", "score_s": "semanal · lunes 02:00 CST",
        "bi_s": "lo consume Power BI",
        "l_months": "particiones en caché", "l_daily": "features", "l_load": "champion",
        "l_fail": "gate fallido → reentrena", "l_after": "training listo → re-score",
        "cap_a": "Dos pipelines de SageMaker comparten un mismo store de modelos. Nada llega a Snowflake sin aprobar el dry run.",
        "inputs": "Entradas", "in_feat": "último snapshot por combo", "in_model": "champion",
        "el_s": "curvas · 0-50%, 9 puntos", "sc_s": "uplift · confianza · semáforo",
        "st_s": "validaciones · ROI margen", "cv_s": "del training",
        "dr_s1": "12 checks entre", "dr_s2": "artefactos",
        "pub_s1": "predicciones + curvas", "pub_s2": "→ DataOps → Snowflake",
        "fail": "RuntimeError", "fail_s1": "publish/ sin cambios", "fail_s2": "se dispara reentreno",
        "l_curves": "curvas de elasticidad", "l_ok": "aprobado", "l_ko": "falla algún check",
        "cap_s": "dryrun.py es el único paso que ve todos los artefactos. publish.py copia solo si el dry run aprueba.",
        "combo": "Combo", "combo_s": "tienda · genérico · descuento",
        "thr": "Semáforo base", "r1": "¿Extrapolado?", "r1_s": "descuento > máx. hist.",
        "r2": "¿Baseline corto?", "r2_s": "historial insuficiente",
        "go_s": "lanzar la promo", "co_s": "revisar antes", "av_s": "no lanzar",
        "l_r1": "GO baja a CONSIDER", "l_r2": "pasa a AVOID",
        "cap_r": "El modelo fija el semáforo base según el uplift. Las dos reglas de control solo pueden bajarlo. Umbrales en config.py.",
    },
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for lang, t in TEXT.items():
        suffix = "" if lang == "en" else ".es"
        (OUT / f"architecture{suffix}.svg").write_text(architecture(t), encoding="utf-8")
        (OUT / f"scoring-gate{suffix}.svg").write_text(scoring_gate(t), encoding="utf-8")
        (OUT / f"recommendation{suffix}.svg").write_text(recommendation(t), encoding="utf-8")
    print(f"Diagramas generados en {OUT}")


if __name__ == "__main__":
    main()
