"""
features.py
Traduce los inputs del formulario en los 43 features que el modelo espera.

Uso:
    from features import transform_inputs
    df_features = transform_inputs(post_data, meta, lookups, enc)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from utils.helpers import tiene_emoji, franja_horaria


def transform_inputs(post_data: dict, meta: dict, lookups: dict, enc) -> pd.DataFrame:
    """
    Recibe el diccionario del formulario y devuelve un DataFrame
    de 1 fila × 43 columnas listo para pasarle a los modelos.

    Parámetros
    ----------
    post_data : dict con las claves del formulario (ver abajo)
    meta      : dict cargado desde models/pipeline_meta.json
    lookups   : dict cargado desde models/lookups.json
    enc       : OrdinalEncoder cargado desde models/ordinal_encoder.pkl

    Claves esperadas en post_data
    -----------------------------
    Obligatorias (usuario las llena):
        Radio, Formato, categoria_contenido, subcategoria_contenido,
        tono_emocional_principal, estilo_comunicativo,
        tipo_llamado_a_la_accion, tipo_enlace, situacion,
        celebridad_principal, tipo_humor, emocion_secundaria,
        Copy, Link (str o None), fecha (date), hora (int 0-23)

    Derivadas automáticamente (NO hace falta pasarlas):
        tipo, incluye_enlace_externo, tiene_llamado_a_la_accion,
        tiene_tono_humoristico
    """
    row = {}

    # ── Inputs directos del usuario ───────────────────────────────────
    radio   = str(post_data.get("Radio",   "_na_"))
    formato = str(post_data.get("Formato", "_na_"))
    tipo    = formato          # tipo se deriva de Formato
    copy    = str(post_data.get("Copy",  ""))
    link    = post_data.get("Link", None)
    hora    = int(post_data.get("hora", 12))
    fecha   = pd.to_datetime(post_data.get("fecha", pd.Timestamp.today()))

    cat_contenido  = str(post_data.get("categoria_contenido",      "_na_"))
    subcat         = str(post_data.get("subcategoria_contenido",   "_na_"))
    tono_emo       = str(post_data.get("tono_emocional_principal", "_na_"))
    estilo_com     = str(post_data.get("estilo_comunicativo",      "_na_"))
    cta            = str(post_data.get("tipo_llamado_a_la_accion", "no aplica"))
    tipo_enlace    = str(post_data.get("tipo_enlace",              "no aplica"))
    situacion      = str(post_data.get("situacion",                "no aplica"))
    celebridad     = str(post_data.get("celebridad_principal",     "no aplica"))
    tipo_humor_val = str(post_data.get("tipo_humor",               "no aplica"))
    emocion_sec    = str(post_data.get("emocion_secundaria",       "no aplica"))

    # ── Derivados automáticos ─────────────────────────────────────────
    incluye_enlace_ext   = "si"  if (link and str(link).strip() != "") else "no"
    tiene_cta            = "si"  if cta != "no aplica" else "no"
    tiene_tono_humor     = "si"  if tipo_humor_val != "no aplica" else "no"

    # ── Temporales ────────────────────────────────────────────────────
    row["hora"]          = hora
    row["dia_semana"]    = fecha.dayofweek
    row["mes"]           = fecha.month
    row["trimestre"]     = (fecha.month - 1) // 3 + 1
    row["semana_anio"]   = int(fecha.isocalendar()[1])
    row["es_fin_semana"] = int(fecha.dayofweek >= 5)
    row["hora_peak_am"]  = int(6 <= hora <= 8)
    row["hora_peak_pm"]  = int(18 <= hora <= 22)
    row["hora_sin"]      = np.sin(2 * np.pi * hora / 24)
    row["hora_cos"]      = np.cos(2 * np.pi * hora / 24)
    row["dia_sin"]       = np.sin(2 * np.pi * fecha.dayofweek / 7)
    row["mes_sin"]       = np.sin(2 * np.pi * (fecha.month - 1) / 12)
    row["mes_cos"]       = np.cos(2 * np.pi * (fecha.month - 1) / 12)

    # ── Contenido ─────────────────────────────────────────────────────
    row["es_video"]   = int(formato == "video")
    row["len_copy"]   = len(copy)
    row["tiene_link"] = int(incluye_enlace_ext == "si")
    row["tiene_emoji"]= tiene_emoji(copy)

    # ── Franja e interacciones categóricas ────────────────────────────
    franja = franja_horaria(hora)
    row["radio_x_formato"]  = f"{radio}_{formato}"
    row["radio_x_tipo"]     = f"{radio}_{tipo}"
    row["formato_x_franja"] = f"{formato}_{franja}"
    row["video_x_radio"]    = f"{row['es_video']}_{radio}"

    # ── Aggregation lookups (desde lookups.json) ──────────────────────
    g_ing  = meta["global_ing"]
    g_post = meta["global_posts_sem"]
    g_var  = meta["global_varianza"]

    row["media_ing_tipo"]      = lookups["map_ing_tipo"].get(tipo,         g_ing)
    row["media_ing_formato"]   = lookups["map_ing_formato"].get(formato,   g_ing)
    row["media_ing_categoria"] = lookups["map_ing_cat"].get(cat_contenido, g_ing)
    row["media_ing_radio"]     = lookups["map_ing_radio"].get(radio,       g_ing)
    row["posts_radio_semana"]  = lookups["map_posts_radio"].get(radio,     g_post)
    row["varianza_ing_radio"]  = lookups["map_varianza"].get(radio,        g_var)

    # ── Columnas categóricas (todas como string) ──────────────────────
    row["Radio"]                   = radio
    row["Formato"]                 = formato
    row["tipo"]                    = tipo
    row["subcategoria_contenido"]  = subcat
    row["categoria_contenido"]     = cat_contenido
    row["tono_emocional_principal"]= tono_emo
    row["estilo_comunicativo"]     = estilo_com
    row["celebridad_principal"]    = celebridad
    row["tipo_enlace"]             = tipo_enlace
    row["incluye_enlace_externo"]  = incluye_enlace_ext
    row["tipo_humor"]              = tipo_humor_val
    row["situacion"]               = situacion
    row["tiene_tono_humoristico"]  = tiene_tono_humor
    row["tipo_llamado_a_la_accion"]= cta
    row["tiene_llamado_a_la_accion"]= tiene_cta
    row["emocion_secundaria"]      = emocion_sec
    row["video_x_radio"]           = row["video_x_radio"]

    # ── Construir DataFrame y aplicar OrdinalEncoder ──────────────────
    df = pd.DataFrame([row])
    cat_cols = meta["CAT_COLS"]

    for c in cat_cols:
        if c not in df.columns:
            df[c] = "_na_"
        df[c] = df[c].fillna("_na_").astype(str)

    df[cat_cols] = enc.transform(df[cat_cols])

    # ── Ordenar columnas exactamente como el modelo las espera ────────
    features = meta["FEATURES"]
    for f in features:
        if f not in df.columns:
            df[f] = 0   # fallback seguro para features ausentes

    return df[features]
