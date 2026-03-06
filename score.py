"""
score.py
Carga los modelos UNA SOLA VEZ al importar y expone la función predecir().

Uso:
    from score import predecir
    resultado = predecir(post_data)
"""
import json
import numpy as np
import joblib
from pathlib import Path
from scipy.special import inv_boxcox

from features import transform_inputs

# ── Rutas ─────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")

# ── Carga única al importar (no se repite en cada predicción) ─────────
clf    = joblib.load(MODELS_DIR / "clf_binario.pkl")
reg    = joblib.load(MODELS_DIR / "reg_boxcox.pkl")
clf_m2 = joblib.load(MODELS_DIR / "clf_rangos.pkl")
enc    = joblib.load(MODELS_DIR / "ordinal_encoder.pkl")

with open(MODELS_DIR / "pipeline_meta.json", encoding="utf-8") as f:
    meta = json.load(f)

with open(MODELS_DIR / "lookups.json", encoding="utf-8") as f:
    lookups = json.load(f)

LAMBDA_BC    = meta["LAMBDA_BC"]
IC_GLOBAL_90 = meta["IC_GLOBAL_90"]
LABELS_RANGO = meta["LABELS_RANGO"]
BINS_RANGO   = meta["BINS_RANGO"]
IC_POR_RADIO = lookups.get("IC_POR_RADIO", {})

# ── Helpers internos ──────────────────────────────────────────────────

def _rango_desde_monto(monto: float) -> str:
    """Convierte un monto en dólares a su etiqueta de rango."""
    for i in range(len(BINS_RANGO) - 1):
        if BINS_RANGO[i] <= monto < BINS_RANGO[i + 1]:
            return LABELS_RANGO[i]
    return LABELS_RANGO[-1]


def _ic_para_radio(radio: str) -> float:
    """Devuelve el IC conformal del radio o el global si no existe."""
    key = radio.replace("Radio ", "").replace("Radio", "").strip()
    entry = IC_POR_RADIO.get(key, {})
    ic = entry.get("ic_honesto", IC_GLOBAL_90)
    if ic is None or (isinstance(ic, float) and np.isnan(ic)):
        return IC_GLOBAL_90
    return float(ic)


def _calcular_score(prob_gana: float, monto: float,
                    ic_ancho: float, rango_m1: str, rango_m2: str) -> dict:
    """
    Calcula score 0-100 con pesos:
      40% prob_gana | 25% monto | 20% IC ancho | 15% consistencia M1-M2
    """
    score_prob  = prob_gana * 100
    score_monto = min(monto / 35.0, 1.0) * 100
    score_ic    = max(0.0, 1.0 - ic_ancho / 50.0) * 100
    score_cons  = 100.0 if rango_m1 == rango_m2 else 0.0

    score_final = (
        0.40 * score_prob  +
        0.25 * score_monto +
        0.20 * score_ic    +
        0.15 * score_cons
    )

    if score_final >= 75:
        etiqueta = "Alta probabilidad"
        emoji    = "🟢"
        consejo  = "El modelo sugiere publicar este post."
    elif score_final >= 50:
        etiqueta = "Potencial medio"
        emoji    = "🟡"
        consejo  = "Considera optimizar el Copy o cambiar la franja horaria."
    else:
        etiqueta = "Bajo potencial"
        emoji    = "🔴"
        consejo  = "Revisa el formato o el tema del contenido."

    return {
        "score_final":  round(score_final, 1),
        "etiqueta":     etiqueta,
        "emoji":        emoji,
        "consejo":      consejo,
        "detalle": {
            "score_prob":  round(score_prob,  1),
            "score_monto": round(score_monto, 1),
            "score_ic":    round(score_ic,    1),
            "score_cons":  round(score_cons,  1),
        }
    }


# ── Función principal ─────────────────────────────────────────────────

def predecir(post_data: dict) -> dict:
    """
    Recibe el diccionario del formulario y devuelve todos los resultados.

    Retorna
    -------
    dict con:
        M1_prob_gana       float   probabilidad de ganar algo (0-1)
        M1_monto           float   monto predicho en USD
        M1_ic_inferior     float   límite inferior IC 90%
        M1_ic_superior     float   límite superior IC 90%
        M1_ic_ancho        float   ancho del IC
        M1_rango           str     rango del monto predicho
        M2_rango           str     rango predicho por M2
        M2_confianza       float   confianza de M2 (0-1)
        M2_probs           dict    probabilidad por cada rango
        score_final        float   score 0-100
        etiqueta           str     "Alta probabilidad" / "Potencial medio" / "Bajo potencial"
        emoji              str     🟢 / 🟡 / 🔴
        consejo            str     texto de recomendación
        detalle            dict    desglose de los 4 componentes del score
    """
    # 1. Transformar inputs → 43 features
    df_feat = transform_inputs(post_data, meta, lookups, enc)

    # 2. M1 — Clasificador binario (¿gana algo?)
    prob_gana = float(clf.predict_proba(df_feat)[:, 1][0])

    # 3. M1 — Regresor Box-Cox (monto dado que gana)
    monto_bc = reg.predict(df_feat)[0]
    monto    = float(np.clip(inv_boxcox(monto_bc, LAMBDA_BC), 0, None))
    monto    = prob_gana * monto   # monto esperado = prob × monto condicional

    # 4. IC conformal por radio
    radio   = str(post_data.get("Radio", ""))
    ic      = _ic_para_radio(radio)
    ic_inf  = round(max(0.0, monto - ic), 2)
    ic_sup  = round(monto + ic, 2)

    # 5. Rango de M1 (desde el monto)
    rango_m1 = _rango_desde_monto(monto)

    # 6. M2 — Clasificador 5 rangos
    proba_m2  = clf_m2.predict_proba(df_feat)[0]
    idx_m2    = int(clf_m2.predict(df_feat)[0])
    rango_m2  = LABELS_RANGO[idx_m2]
    conf_m2   = float(proba_m2.max())
    probs_m2  = {LABELS_RANGO[i]: round(float(p), 4) for i, p in enumerate(proba_m2)}

    # 7. Score 0-100 + etiqueta
    score_dict = _calcular_score(prob_gana, monto, ic, rango_m1, rango_m2)

    return {
        "M1_prob_gana":   round(prob_gana, 4),
        "M1_monto":       round(monto, 2),
        "M1_ic_inferior": ic_inf,
        "M1_ic_superior": ic_sup,
        "M1_ic_ancho":    round(ic, 2),
        "M1_rango":       rango_m1,
        "M2_rango":       rango_m2,
        "M2_confianza":   round(conf_m2, 4),
        "M2_probs":       probs_m2,
        **score_dict,
    }
