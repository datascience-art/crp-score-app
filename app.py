"""
app.py — CRP Medios y Entretenimiento SAC
Aplicación de scoring de posts para redes sociales.
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import date
from scipy.special import inv_boxcox

from features import transform_inputs
from utils.helpers import franja_horaria

# ══════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CRP Score · Predictor de Posts",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════
# CSS CORPORATIVO — CRP Medios y Entretenimiento
# Dirección estética: dark editorial, tipografía fuerte, rojo CRP
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Variables ── */
:root {
    --bg:        #080C14;
    --surface:   #0F1520;
    --surface2:  #162030;
    --border:    #1E2D45;
    --accent:    #D9262C;
    --accent2:   #FF4B51;
    --gold:      #F5A623;
    --green:     #1DB96A;
    --yellow:    #F5C842;
    --text:      #EBF0FA;
    --muted:     #6B7E9F;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
}

/* ── Reset global ── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg) !important; }

/* ── Header corporativo ── */
.crp-header {
    background: linear-gradient(135deg, #0D1422 0%, #12203A 50%, #0D1422 100%);
    border-bottom: 2px solid var(--accent);
    padding: 1.4rem 2rem 1.2rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 1.4rem;
    position: relative;
    overflow: hidden;
}
.crp-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(217,38,44,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.crp-logo-badge {
    background: var(--accent);
    color: white;
    font-family: var(--font-head);
    font-weight: 800;
    font-size: 1.35rem;
    padding: 0.45rem 0.9rem;
    border-radius: 6px;
    letter-spacing: 0.04em;
    flex-shrink: 0;
}
.crp-header-text h1 {
    font-family: var(--font-head);
    font-weight: 700;
    font-size: 1.25rem;
    margin: 0;
    color: var(--text) !important;
    letter-spacing: -0.01em;
}
.crp-header-text p {
    font-size: 0.78rem;
    color: var(--muted);
    margin: 0.1rem 0 0 0;
    font-weight: 300;
}
.crp-header-badge {
    margin-left: auto;
    background: rgba(29,185,106,0.12);
    border: 1px solid rgba(29,185,106,0.3);
    color: var(--green);
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    letter-spacing: 0.03em;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: var(--font-head);
    font-weight: 600;
    font-size: 0.85rem;
    border-radius: 7px;
    padding: 0.55rem 1.4rem;
    border: none !important;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* ── Cards ── */
.crp-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.crp-card-title {
    font-family: var(--font-head);
    font-weight: 700;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
}

/* ── Score display ── */
.score-hero {
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.score-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--gold), var(--accent));
}
.score-number {
    font-family: var(--font-head);
    font-weight: 800;
    font-size: 4.5rem;
    line-height: 1;
    margin: 0.5rem 0;
    letter-spacing: -0.03em;
}
.score-label {
    font-family: var(--font-head);
    font-weight: 600;
    font-size: 1.05rem;
    margin-top: 0.4rem;
}
.score-consejo {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 0.6rem;
    font-weight: 300;
}

/* ── Barra de score ── */
.score-bar-wrap {
    background: var(--border);
    border-radius: 99px;
    height: 8px;
    margin: 1rem 0 0.3rem 0;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s ease;
}

/* ── Metric chips ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    margin-top: 1rem;
}
.metric-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
}
.metric-chip-label {
    font-size: 0.7rem;
    color: var(--muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.metric-chip-value {
    font-family: var(--font-head);
    font-weight: 700;
    font-size: 1.25rem;
    color: var(--text);
}
.metric-chip-sub {
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 0.15rem;
}

/* ── Rango badges ── */
.rango-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: var(--font-head);
    letter-spacing: 0.03em;
}
.rango-0    { background: rgba(107,126,159,0.15); color: #6B7E9F; border: 1px solid #1E2D45; }
.rango-1    { background: rgba(59,130,246,0.15);  color: #60A5FA; border: 1px solid rgba(59,130,246,0.3); }
.rango-2    { background: rgba(245,200,66,0.15);  color: #F5C842; border: 1px solid rgba(245,200,66,0.3); }
.rango-3    { background: rgba(245,166,35,0.15);  color: #F5A623; border: 1px solid rgba(245,166,35,0.3); }
.rango-4    { background: rgba(29,185,106,0.15);  color: #1DB96A; border: 1px solid rgba(29,185,106,0.3); }

/* ── Consistencia M1-M2 ── */
.consistencia-ok  { color: var(--green); font-size: 0.78rem; font-weight: 500; }
.consistencia-no  { color: var(--gold);  font-size: 0.78rem; font-weight: 500; }

/* ── Desglose score ── */
.desglose-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.82rem;
}
.desglose-row:last-child { border-bottom: none; }
.desglose-label { color: var(--muted); }
.desglose-val   { font-family: var(--font-head); font-weight: 600; color: var(--text); }
.desglose-peso  { font-size: 0.7rem; color: var(--muted); margin-left: 0.4rem; }

/* ── Posts agregados (pestaña 2) ── */
.post-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.83rem;
}
.post-item-info { display: flex; flex-direction: column; gap: 0.15rem; }
.post-item-title { font-family: var(--font-head); font-weight: 600; color: var(--text); }
.post-item-meta  { color: var(--muted); font-size: 0.75rem; }

/* ── Tabla ranking ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
.stDataFrame table { background: var(--surface) !important; }

/* ── Inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div,
.stTextArea > div > div,
.stDateInput > div > div,
.stNumberInput > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div:focus-within,
.stTextArea > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(217,38,44,0.15) !important;
}
label { color: var(--muted) !important; font-size: 0.8rem !important; font-weight: 500 !important; }

/* ── Botones ── */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.5rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: var(--accent2) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(217,38,44,0.35) !important;
}
.stDownloadButton > button {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Info/warning/error ── */
.stAlert { border-radius: 10px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# CARGA DE MODELOS — una sola vez en toda la sesión
# ══════════════════════════════════════════════════════════════════════
MODELS_DIR = Path("models")

@st.cache_resource(show_spinner="Cargando modelos CRP…")
def cargar_modelos():
    clf    = joblib.load(MODELS_DIR / "clf_binario.pkl")
    reg    = joblib.load(MODELS_DIR / "reg_boxcox.pkl")
    clf_m2 = joblib.load(MODELS_DIR / "clf_rangos.pkl")
    enc    = joblib.load(MODELS_DIR / "ordinal_encoder.pkl")
    with open(MODELS_DIR / "pipeline_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open(MODELS_DIR / "lookups.json", encoding="utf-8") as f:
        lookups = json.load(f)
    return clf, reg, clf_m2, enc, meta, lookups

clf, reg, clf_m2, enc, meta, lookups = cargar_modelos()

LAMBDA_BC    = meta["LAMBDA_BC"]
IC_GLOBAL_90 = meta["IC_GLOBAL_90"]
LABELS_RANGO = meta["LABELS_RANGO"]
BINS_RANGO   = meta["BINS_RANGO"]
IC_POR_RADIO = lookups.get("IC_POR_RADIO", {})

# ══════════════════════════════════════════════════════════════════════
# VALORES PARA SELECTBOXES (desde los datos reales del modelo)
# ══════════════════════════════════════════════════════════════════════
def _opciones(lookup_key: str, default_extra: str = "no aplica") -> list:
    """Extrae opciones únicas desde los lookups y agrega 'no aplica' si no existe."""
    vals = sorted(lookups.get(lookup_key, {}).keys())
    if default_extra not in vals:
        vals = [default_extra] + vals
    return vals

# Valores hardcodeados desde el análisis del dataset
RADIOS     = sorted(lookups["map_ing_radio"].keys())
FORMATOS   = ["photo", "video", "link", "Texto"]
CATEGORIAS = sorted(lookups["map_ing_cat"].keys())
TIPOS_CTA  = ["no aplica","aplaudir","asistir","ayudar","celebrar","comentar",
               "comprar","etiquetar","participar","registrarse","seguir",
               "sintonizar","subir contenido","visitar","votar"]
TIPOS_ENLACE = ["no aplica","app store","articulo web","evento","facebook",
                "instagram","sitio oficial","tienda online","youtube"]
TONOS_EMO  = ["admiracion","alegria","amor","celebratorio","confianza",
               "curiosidad","empoderamiento","entusiasmo","gratitud","humor",
               "indignacion","inspiracion","melancolia","nostalgia","orgullo",
               "serenidad","sorpresa","ternura","tristeza"]
ESTILOS    = ["celebratorio","conversacional","critico","descriptivo","didactico",
               "emotivo","humoristico","informativo","inspirador","narrativo",
               "opinativo","persuasivo","publicitario","sarcastico"]
SITUACIONES = ["no aplica","actividad recreativa","actuacion","alfombra roja",
                "anecdota","anuncio","behind the scenes","celebracion","concierto",
                "encuentro","ensayo","evento","homenaje","interaccion social",
                "meet and greet","premiacion","presentacion de disco","programa en directo",
                "sesion de fotos","tour promocional","videoclip","visita"]
CELEBRIDADES = ["no aplica"]
TIPOS_HUMOR  = ["no aplica","doble sentido","humor absurdo","humor blanco",
                 "humor fisico","humor situacional","juegos de palabras","memes","parodia","sarcasmo"]
EMOCIONES_SEC = ["no aplica","admiracion","amor","curiosidad","empoderamiento",
                  "entusiasmo","gratitud","humor","indignacion","inspiracion",
                  "melancolia","orgullo","satisfaccion","serenidad","sorpresa","ternura"]
SUBCATS = ["anuncio","baile","celebracion","concierto","concurso","cronica",
           "entretenimiento","evento","humor","interes humano","musica","noticia",
           "noticia viral","premiacion","presentacion","programa","reportaje","videoclip"]

# ══════════════════════════════════════════════════════════════════════
# FUNCIÓN DE PREDICCIÓN CENTRAL
# ══════════════════════════════════════════════════════════════════════
def _ic_para_radio(radio: str) -> float:
    key = radio.replace("Radio ", "").replace("Radio", "").strip()
    entry = IC_POR_RADIO.get(key, {})
    ic = entry.get("ic_honesto", IC_GLOBAL_90)
    if ic is None or (isinstance(ic, float) and np.isnan(ic)):
        return IC_GLOBAL_90
    return float(ic)

def _rango_desde_monto(monto: float) -> str:
    for i in range(len(BINS_RANGO) - 1):
        if BINS_RANGO[i] <= monto < BINS_RANGO[i + 1]:
            return LABELS_RANGO[i]
    return LABELS_RANGO[-1]

def predecir(post_data: dict) -> dict:
    df_feat = transform_inputs(post_data, meta, lookups, enc)

    prob_gana = float(clf.predict_proba(df_feat)[:, 1][0])
    monto_bc  = reg.predict(df_feat)[0]
    monto_raw = float(np.clip(inv_boxcox(monto_bc, LAMBDA_BC), 0, None))
    monto     = prob_gana * monto_raw

    radio  = str(post_data.get("Radio", ""))
    ic     = _ic_para_radio(radio)
    ic_inf = round(max(0.0, monto - ic), 2)
    ic_sup = round(monto + ic, 2)

    rango_m1 = _rango_desde_monto(monto)

    proba_m2 = clf_m2.predict_proba(df_feat)[0]
    idx_m2   = int(clf_m2.predict(df_feat)[0])
    rango_m2 = LABELS_RANGO[idx_m2]
    conf_m2  = float(proba_m2.max())
    probs_m2 = {LABELS_RANGO[i]: round(float(p), 4) for i, p in enumerate(proba_m2)}

    # Score 0-100
    s_prob  = prob_gana * 100
    s_monto = min(monto / 35.0, 1.0) * 100
    s_ic    = max(0.0, 1.0 - ic / 50.0) * 100
    s_cons  = 100.0 if rango_m1 == rango_m2 else 0.0
    score   = round(0.40*s_prob + 0.25*s_monto + 0.20*s_ic + 0.15*s_cons, 1)

    if score >= 75:
        etiqueta, emoji, consejo = "Alta probabilidad", "🟢", "El modelo sugiere publicar este post."
    elif score >= 50:
        etiqueta, emoji, consejo = "Potencial medio", "🟡", "Considera optimizar el Copy o la franja horaria."
    else:
        etiqueta, emoji, consejo = "Bajo potencial", "🔴", "Revisa el formato o el tema del contenido."

    return {
        "M1_prob_gana": round(prob_gana, 4),
        "M1_monto":     round(monto, 2),
        "M1_ic_inf":    ic_inf,
        "M1_ic_sup":    ic_sup,
        "M1_ic_ancho":  round(ic, 2),
        "M1_rango":     rango_m1,
        "M2_rango":     rango_m2,
        "M2_confianza": round(conf_m2, 4),
        "M2_probs":     probs_m2,
        "score":        score,
        "etiqueta":     etiqueta,
        "emoji":        emoji,
        "consejo":      consejo,
        "detalle": {
            "Prob. ganar (×40%)":      round(s_prob, 1),
            "Monto estimado (×25%)":   round(s_monto, 1),
            "Confianza IC (×20%)":     round(s_ic, 1),
            "Consistencia M1-M2 (×15%)": round(s_cons, 1),
        }
    }

# ══════════════════════════════════════════════════════════════════════
# HELPERS DE RENDER
# ══════════════════════════════════════════════════════════════════════
RANGO_CLASS = {"$0": "rango-0", "$0-$5": "rango-1",
               "$5-$10": "rango-2", "$10-$35": "rango-3", ">$35": "rango-4"}

def _score_color(score: float) -> str:
    if score >= 75: return "#1DB96A"
    if score >= 50: return "#F5C842"
    return "#D9262C"

def render_resultado(res: dict):
    sc = res["score"]
    color = _score_color(sc)

    # ── Hero score ──
    st.markdown(f"""
    <div class="score-hero">
        <div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;
                    letter-spacing:0.12em;color:var(--muted);margin-bottom:0.2rem;">
            Score de publicación
        </div>
        <div class="score-number" style="color:{color}">{sc}</div>
        <div class="score-label">{res['emoji']} {res['etiqueta']}</div>
        <div class="score-bar-wrap">
            <div class="score-bar-fill"
                 style="width:{sc}%;background:linear-gradient(90deg,{color}AA,{color});">
            </div>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-size:0.68rem;color:var(--muted);margin-top:0.2rem;">
            <span>0</span><span>50</span><span>100</span>
        </div>
        <div class="score-consejo">{res['consejo']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics grid ──
    rango_m1_cls = RANGO_CLASS.get(res["M1_rango"], "rango-0")
    rango_m2_cls = RANGO_CLASS.get(res["M2_rango"], "rango-0")
    consistencia = (
        f'<span class="consistencia-ok">✓ Coinciden</span>'
        if res["M1_rango"] == res["M2_rango"]
        else f'<span class="consistencia-no">⚠ Difieren</span>'
    )

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-chip">
            <div class="metric-chip-label">Prob. de ganar</div>
            <div class="metric-chip-value">{res['M1_prob_gana']*100:.1f}%</div>
            <div class="metric-chip-sub">Clasificador binario M1</div>
        </div>
        <div class="metric-chip">
            <div class="metric-chip-label">Monto estimado</div>
            <div class="metric-chip-value">${res['M1_monto']:.2f}</div>
            <div class="metric-chip-sub">IC 90%: ${res['M1_ic_inf']} — ${res['M1_ic_sup']}</div>
        </div>
        <div class="metric-chip">
            <div class="metric-chip-label">Rango M1 · M2</div>
            <div class="metric-chip-value" style="display:flex;gap:0.4rem;align-items:center;">
                <span class="rango-badge {rango_m1_cls}">{res['M1_rango']}</span>
                <span class="rango-badge {rango_m2_cls}">{res['M2_rango']}</span>
            </div>
            <div class="metric-chip-sub">{consistencia}</div>
        </div>
        <div class="metric-chip">
            <div class="metric-chip-label">Confianza M2</div>
            <div class="metric-chip-value">{res['M2_confianza']*100:.1f}%</div>
            <div class="metric-chip-sub">Clasificador 5 rangos</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Desglose score ──
    st.markdown('<div class="crp-card"><div class="crp-card-title">Desglose del Score</div>', unsafe_allow_html=True)
    for label, val in res["detalle"].items():
        st.markdown(f"""
        <div class="desglose-row">
            <span class="desglose-label">{label}</span>
            <span class="desglose-val">{val:.1f} <span class="desglose-peso">/ 100</span></span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Probs M2 ──
    with st.expander("Ver probabilidades por rango (M2)"):
        cols = st.columns(5)
        for i, (rango, prob) in enumerate(res["M2_probs"].items()):
            cls = RANGO_CLASS.get(rango, "rango-0")
            with cols[i]:
                st.markdown(f"""
                <div style="text-align:center;">
                    <span class="rango-badge {cls}">{rango}</span>
                    <div style="font-family:var(--font-head);font-weight:700;
                                font-size:1.1rem;margin-top:0.4rem;">{prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)


def formulario(key_prefix: str = "") -> dict | None:
    """Renderiza el formulario y retorna post_data si se envía, None si no."""
    with st.form(key=f"form_{key_prefix}", clear_on_submit=False):
        st.markdown('<div class="crp-card-title">📡 Datos del Post</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            radio   = st.selectbox("Radio *", RADIOS)
            formato = st.selectbox("Formato *", FORMATOS)
            fecha   = st.date_input("Fecha de publicación *", value=date.today())
            hora    = st.slider("Hora de publicación *", 0, 23, 12,
                                format="%dh",
                                help=f"Franja: {franja_horaria(12)}")
        with c2:
            cat   = st.selectbox("Categoría de contenido *", CATEGORIAS)
            subcat= st.selectbox("Subcategoría", SUBCATS)
            tono  = st.selectbox("Tono emocional *", TONOS_EMO)
            estilo= st.selectbox("Estilo comunicativo *", ESTILOS)

        copy = st.text_area("Copy del post *",
                            placeholder="Escribe o pega aquí el texto del post…",
                            height=100)
        link = st.text_input("Link (opcional)", placeholder="https://…")

        st.markdown("---")
        st.markdown('<div class="crp-card-title">⚙️ Parámetros opcionales</div>', unsafe_allow_html=True)

        c3, c4, c5 = st.columns(3)
        with c3:
            cta          = st.selectbox("Llamado a la acción", TIPOS_CTA)
            tipo_enlace  = st.selectbox("Tipo de enlace", TIPOS_ENLACE)
        with c4:
            situacion    = st.selectbox("Situación / contexto", SITUACIONES)
            celebridad   = st.text_input("Celebridad principal", value="no aplica",
                                         placeholder="ej. Bad Bunny")
        with c5:
            tipo_humor   = st.selectbox("Tipo de humor", TIPOS_HUMOR)
            emocion_sec  = st.selectbox("Emoción secundaria", EMOCIONES_SEC)

        submitted = st.form_submit_button("🔍  Analizar post", use_container_width=True)

    if submitted:
        if not copy.strip():
            st.error("El Copy es obligatorio para predecir.")
            return None
        return {
            "Radio":                    radio,
            "Formato":                  formato,
            "fecha":                    pd.Timestamp(fecha),
            "hora":                     hora,
            "categoria_contenido":      cat,
            "subcategoria_contenido":   subcat,
            "tono_emocional_principal": tono,
            "estilo_comunicativo":      estilo,
            "tipo_llamado_a_la_accion": cta,
            "tipo_enlace":              tipo_enlace,
            "situacion":                situacion,
            "celebridad_principal":     celebridad if celebridad.strip() else "no aplica",
            "tipo_humor":               tipo_humor,
            "emocion_secundaria":       emocion_sec,
            "Copy":                     copy,
            "Link":                     link if link.strip() else None,
        }
    return None


# ══════════════════════════════════════════════════════════════════════
# HEADER CORPORATIVO
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="crp-header">
    <div class="crp-logo-badge">CRP</div>
    <div class="crp-header-text">
        <h1>Score Predictor · Redes Sociales</h1>
        <p>CRP Medios y Entretenimiento SAC · Herramienta interna de análisis de contenido</p>
    </div>
    <div class="crp-header-badge">● Modelo v17 activo</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PESTAÑAS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["📋  Analizar un post", "📊  Comparar múltiples posts"])

# ──────────────────────────────────────────────────────────────────────
# PESTAÑA 1 — Un post
# ──────────────────────────────────────────────────────────────────────
with tab1:
    col_form, col_res = st.columns([1.1, 0.9], gap="large")

    with col_form:
        post_data = formulario(key_prefix="tab1")

    with col_res:
        if post_data:
            try:
                with st.spinner("Analizando…"):
                    res = predecir(post_data)
                st.session_state["ultimo_resultado"] = res
                render_resultado(res)
            except Exception as e:
                st.error(f"Error en la predicción: {e}")
        elif "ultimo_resultado" in st.session_state:
            st.markdown('<p style="color:var(--muted);font-size:0.8rem;">Último resultado:</p>',
                        unsafe_allow_html=True)
            render_resultado(st.session_state["ultimo_resultado"])
        else:
            st.markdown("""
            <div style="height:100%;display:flex;flex-direction:column;
                        align-items:center;justify-content:center;
                        padding:3rem 1rem;text-align:center;">
                <div style="font-size:2.5rem;margin-bottom:1rem;">📡</div>
                <div style="font-family:var(--font-head);font-weight:700;
                             font-size:1.1rem;color:var(--text);margin-bottom:0.5rem;">
                    Completa el formulario
                </div>
                <div style="color:var(--muted);font-size:0.82rem;max-width:260px;line-height:1.6;">
                    Ingresa los datos del post y haz clic en
                    <strong style="color:var(--accent);">Analizar post</strong>
                    para ver el score y la predicción de ingresos.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# PESTAÑA 2 — Múltiples posts
# ──────────────────────────────────────────────────────────────────────
with tab2:
    if "posts_cola" not in st.session_state:
        st.session_state["posts_cola"] = []

    col_agregar, col_lista = st.columns([1.1, 0.9], gap="large")

    with col_agregar:
        st.markdown('<div class="crp-card-title">➕ Agregar post a la cola</div>', unsafe_allow_html=True)

        with st.form("form_tab2", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                radio2   = st.selectbox("Radio *", RADIOS,         key="r2_radio")
                formato2 = st.selectbox("Formato *", FORMATOS,     key="r2_formato")
                fecha2   = st.date_input("Fecha *", value=date.today(), key="r2_fecha")
                hora2    = st.slider("Hora *", 0, 23, 12, format="%dh", key="r2_hora")
            with c2:
                cat2    = st.selectbox("Categoría *", CATEGORIAS,  key="r2_cat")
                subcat2 = st.selectbox("Subcategoría", SUBCATS,    key="r2_subcat")
                tono2   = st.selectbox("Tono emocional *", TONOS_EMO, key="r2_tono")
                estilo2 = st.selectbox("Estilo *", ESTILOS,        key="r2_estilo")

            copy2 = st.text_area("Copy *", placeholder="Texto del post…",
                                 height=80, key="r2_copy")
            link2 = st.text_input("Link (opcional)", key="r2_link")

            with st.expander("Opcionales"):
                c3, c4 = st.columns(2)
                with c3:
                    cta2       = st.selectbox("CTA", TIPOS_CTA,          key="r2_cta")
                    tipo_enl2  = st.selectbox("Tipo enlace", TIPOS_ENLACE,key="r2_enlace")
                    situacion2 = st.selectbox("Situación", SITUACIONES,   key="r2_sit")
                with c4:
                    celeb2     = st.text_input("Celebridad", value="no aplica", key="r2_celeb")
                    humor2     = st.selectbox("Tipo humor", TIPOS_HUMOR,  key="r2_humor")
                    emosec2    = st.selectbox("Emoción sec.", EMOCIONES_SEC, key="r2_emosec")

            agregar = st.form_submit_button("➕  Agregar a la cola", use_container_width=True)

        if agregar:
            if not copy2.strip():
                st.error("El Copy es obligatorio.")
            elif len(st.session_state["posts_cola"]) >= 50:
                st.warning("Máximo 50 posts por sesión.")
            else:
                st.session_state["posts_cola"].append({
                    "Radio": radio2, "Formato": formato2,
                    "fecha": pd.Timestamp(fecha2), "hora": hora2,
                    "categoria_contenido": cat2, "subcategoria_contenido": subcat2,
                    "tono_emocional_principal": tono2, "estilo_comunicativo": estilo2,
                    "tipo_llamado_a_la_accion": cta2, "tipo_enlace": tipo_enl2,
                    "situacion": situacion2,
                    "celebridad_principal": celeb2 if celeb2.strip() else "no aplica",
                    "tipo_humor": humor2, "emocion_secundaria": emosec2,
                    "Copy": copy2,
                    "Link": link2 if link2.strip() else None,
                    "_copy_preview": copy2[:50] + ("…" if len(copy2) > 50 else ""),
                })
                st.success(f"Post agregado. Total en cola: {len(st.session_state['posts_cola'])}")

    with col_lista:
        n = len(st.session_state["posts_cola"])
        st.markdown(f'<div class="crp-card-title">🗂️ Cola de posts ({n} / 50)</div>',
                    unsafe_allow_html=True)

        if n == 0:
            st.markdown("""
            <div style="padding:2.5rem;text-align:center;color:var(--muted);font-size:0.83rem;">
                Agrega al menos un post para comenzar el análisis comparativo.
            </div>""", unsafe_allow_html=True)
        else:
            # Lista de posts con botón eliminar
            for i, p in enumerate(st.session_state["posts_cola"]):
                c_info, c_del = st.columns([5, 1])
                with c_info:
                    st.markdown(f"""
                    <div class="post-item">
                        <div class="post-item-info">
                            <div class="post-item-title">#{i+1} · {p['Radio']} · {p['Formato']}</div>
                            <div class="post-item-meta">{p['_copy_preview']}</div>
                            <div class="post-item-meta">{p['fecha'].strftime('%d/%m/%Y')} · {p['hora']}h · {p['categoria_contenido']}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                with c_del:
                    if st.button("✕", key=f"del_{i}", help="Eliminar este post"):
                        st.session_state["posts_cola"].pop(i)
                        st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            bc1, bc2 = st.columns(2)
            with bc1:
                analizar_todos = st.button("🔍  Predecir todos", use_container_width=True)
            with bc2:
                if st.button("🗑️  Limpiar cola", use_container_width=True):
                    st.session_state["posts_cola"] = []
                    if "ranking_df" in st.session_state:
                        del st.session_state["ranking_df"]
                    st.rerun()

            if analizar_todos and n > 0:
                resultados = []
                progress = st.progress(0, text="Analizando posts…")
                for i, p in enumerate(st.session_state["posts_cola"]):
                    try:
                        r = predecir(p)
                        resultados.append({
                            "#":              i + 1,
                            "Radio":          p["Radio"],
                            "Formato":        p["Formato"],
                            "Hora":           f"{p['hora']}h",
                            "Copy":           p["_copy_preview"],
                            "Score":          r["score"],
                            "Etiqueta":       f"{r['emoji']} {r['etiqueta']}",
                            "Prob. ganar":    f"{r['M1_prob_gana']*100:.1f}%",
                            "Monto est.":     f"${r['M1_monto']:.2f}",
                            "IC 90%":         f"${r['M1_ic_inf']} — ${r['M1_ic_sup']}",
                            "Rango M1":       r["M1_rango"],
                            "Rango M2":       r["M2_rango"],
                            "Consistencia":   "✓" if r["M1_rango"] == r["M2_rango"] else "⚠",
                        })
                    except Exception as e:
                        resultados.append({"#": i+1, "Radio": p["Radio"],
                                           "Score": -1, "Etiqueta": f"❌ Error: {e}"})
                    progress.progress((i + 1) / n, text=f"Analizando {i+1}/{n}…")

                progress.empty()
                df_rank = pd.DataFrame(resultados).sort_values("Score", ascending=False)
                st.session_state["ranking_df"] = df_rank

    # Mostrar tabla ranking debajo (ancho completo)
    if "ranking_df" in st.session_state:
        st.markdown("---")
        st.markdown('<div class="crp-card-title">🏆 Ranking de Posts por Score</div>',
                    unsafe_allow_html=True)
        df_show = st.session_state["ranking_df"]
        st.dataframe(
            df_show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=100, format="%.1f"
                ),
            }
        )
        # Descarga CSV
        csv = df_show.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Descargar resultados CSV",
            data=csv,
            file_name="crp_ranking_posts.csv",
            mime="text/csv",
        )
