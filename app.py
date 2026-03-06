"""
app.py — CRP Medios y Entretenimiento SAC
Score Predictor · Redes Sociales — v17
"""
import json, joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import date
from scipy.special import inv_boxcox
from features import transform_inputs
from utils.helpers import franja_horaria

# ── Página ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRP Score Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ejecutivo CRP ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
  --crp-green:   #38A84B;
  --crp-green2:  #2E8F3E;
  --crp-red:     #E8343A;
  --crp-blue:    #1A5FA6;
  --crp-cyan:    #3DB8D4;
  --crp-yellow:  #F5B82E;
  --crp-orange:  #F07D2E;
  --bg:          #F4F6F9;
  --surface:     #FFFFFF;
  --surface2:    #F9FAFB;
  --border:      #E5E9EF;
  --border2:     #D0D7E2;
  --text:        #111827;
  --text2:       #374151;
  --muted:       #6B7280;
  --muted2:      #9CA3AF;
  --font:        'Plus Jakarta Sans', sans-serif;
  --radius:      10px;
  --shadow:      0 1px 4px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.06);
  --shadow-md:   0 4px 20px rgba(0,0,0,0.10);
}

html, body, [class*="css"] {
  font-family: var(--font) !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }

/* ── Ocultar elementos Streamlit ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1400px !important; }

/* ── Header ── */
.crp-topbar {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 0.9rem 2rem;
  display: flex;
  align-items: center;
  gap: 1.2rem;
  margin: -1rem -1rem 1.5rem -1rem;
}
.crp-logo {
  width: 40px; height: 40px;
  background: var(--crp-green);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.crp-logo svg { width: 22px; height: 22px; fill: white; }
.crp-title { font-weight: 700; font-size: 1rem; color: var(--text); letter-spacing: -0.02em; }
.crp-subtitle { font-size: 0.72rem; color: var(--muted); font-weight: 400; margin-top: 1px; }
.crp-pill {
  margin-left: auto;
  background: #ECFDF3;
  color: var(--crp-green2);
  font-size: 0.7rem; font-weight: 600;
  padding: 0.3rem 0.8rem;
  border-radius: 20px;
  border: 1px solid #A7F3C0;
  letter-spacing: 0.02em;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 3px !important;
  gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  font-family: var(--font) !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  border-radius: 7px !important;
  border: none !important;
  padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
  background: var(--crp-green) !important;
  color: white !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem !important; }

/* ── Cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.4rem;
  box-shadow: var(--shadow);
}
.card-title {
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin-bottom: 1rem;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid var(--border);
}

/* ── Score hero ── */
.score-hero {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.8rem 1.5rem 1.5rem;
  box-shadow: var(--shadow);
  position: relative;
  overflow: hidden;
}
.score-hero-accent {
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
  border-radius: var(--radius) var(--radius) 0 0;
}
.score-number {
  font-size: 4rem;
  font-weight: 800;
  line-height: 1;
  letter-spacing: -0.04em;
}
.score-bar-bg {
  background: var(--border);
  border-radius: 99px;
  height: 6px;
  margin: 1rem 0 0.3rem;
  overflow: hidden;
}
.score-bar-fill { height: 100%; border-radius: 99px; }
.score-label {
  font-size: 0.88rem;
  font-weight: 600;
  margin-top: 0.35rem;
}
.score-tip {
  font-size: 0.77rem;
  color: var(--muted);
  margin-top: 0.4rem;
  font-weight: 400;
  line-height: 1.5;
}

/* ── Metric chips ── */
.chips-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.6rem;
  margin-top: 0.75rem;
}
.chip {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.75rem 0.9rem;
}
.chip-label { font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted2); margin-bottom: 0.25rem; }
.chip-val   { font-size: 1.15rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em; }
.chip-sub   { font-size: 0.68rem; color: var(--muted); margin-top: 0.1rem; }

/* ── Rango badges ── */
.badge {
  display: inline-flex;
  align-items: center;
  padding: 0.18rem 0.6rem;
  border-radius: 99px;
  font-size: 0.72rem;
  font-weight: 600;
}
.b0 { background:#F3F4F6; color:#6B7280; }
.b1 { background:#EFF6FF; color:#3B82F6; }
.b2 { background:#FEFCE8; color:#CA8A04; }
.b3 { background:#FFF7ED; color:#EA580C; }
.b4 { background:#ECFDF5; color:#059669; }

/* ── Desglose ── */
.row-desglose {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.45rem 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.79rem;
}
.row-desglose:last-child { border-bottom: none; }

/* ── Timing table ── */
.timing-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.55rem 0.75rem;
  border-radius: 8px;
  margin-bottom: 0.3rem;
  font-size: 0.82rem;
  border: 1px solid transparent;
}
.timing-row.selected {
  background: #ECFDF3;
  border-color: #A7F3C0;
}
.timing-row.top { background: var(--surface2); }
.timing-hora  { font-weight: 700; font-size: 0.88rem; width: 36px; color: var(--text); }
.timing-franja { font-size: 0.7rem; color: var(--muted); width: 72px; }
.timing-bar-wrap { flex: 1; background: var(--border); border-radius: 99px; height: 5px; overflow: hidden; }
.timing-bar-fill { height: 100%; border-radius: 99px; background: var(--crp-green); }
.timing-score { font-weight: 700; font-size: 0.85rem; width: 38px; text-align: right; }
.timing-badge { font-size: 0.65rem; font-weight: 600; padding: 0.1rem 0.45rem; border-radius: 99px; }
.timing-top1 { background: #FFF7ED; color: #EA580C; }
.timing-sel  { background: #ECFDF3; color: #059669; }

/* ── Post items cola ── */
.post-row {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.65rem 0.9rem;
  margin-bottom: 0.4rem;
  font-size: 0.8rem;
}
.post-row-title { font-weight: 600; color: var(--text); }
.post-row-meta  { color: var(--muted); font-size: 0.72rem; margin-top: 2px; }

/* ── Inputs — solo label y borde sutil, NO tocar el widget nativo ── */
label, .stSelectbox label, .stTextInput label,
.stTextArea label, .stDateInput label, .stSlider label {
  color: var(--text2) !important;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
}
.stSelectbox [data-baseweb="select"] > div,
.stTextInput [data-baseweb="input"],
.stTextArea [data-baseweb="textarea"] {
  border-color: var(--border2) !important;
  border-radius: 8px !important;
  background: var(--surface) !important;
  font-size: 0.83rem !important;
  font-family: var(--font) !important;
}
.stSelectbox [data-baseweb="select"] > div:focus-within,
.stTextInput [data-baseweb="input"]:focus-within {
  border-color: var(--crp-green) !important;
  box-shadow: 0 0 0 2px rgba(56,168,75,0.15) !important;
}

/* ── Botones ── */
.stButton > button {
  background: var(--crp-green) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: var(--font) !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  padding: 0.55rem 1.2rem !important;
  transition: background 0.15s, transform 0.1s !important;
}
.stButton > button:hover {
  background: var(--crp-green2) !important;
  transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
  background: var(--surface) !important;
  color: var(--text2) !important;
  border: 1px solid var(--border2) !important;
}
.stDownloadButton > button {
  background: var(--surface) !important;
  color: var(--text2) !important;
  border: 1px solid var(--border2) !important;
}
.stForm [data-testid="stFormSubmitButton"] > button {
  background: var(--crp-green) !important;
  width: 100% !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  color: var(--muted) !important;
  background: var(--surface2) !important;
  border-radius: 8px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

/* ── Dataframe ── */
.stDataFrame { border-radius: var(--radius) !important; overflow: hidden !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Modelos (carga única) ─────────────────────────────────────────────
MODELS_DIR = Path("models")

@st.cache_resource(show_spinner="Inicializando modelos CRP…")
def cargar_modelos():
    clf    = joblib.load(MODELS_DIR / "clf_binario.pkl")
    reg    = joblib.load(MODELS_DIR / "reg_boxcox.pkl")
    clf_m2 = joblib.load(MODELS_DIR / "clf_rangos.pkl")
    enc    = joblib.load(MODELS_DIR / "ordinal_encoder.pkl")
    with open(MODELS_DIR / "pipeline_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open(MODELS_DIR / "lookups.json", encoding="utf-8") as f:
        lkp = json.load(f)
    return clf, reg, clf_m2, enc, meta, lkp

clf, reg, clf_m2, enc, meta, lookups = cargar_modelos()
LAMBDA_BC    = meta["LAMBDA_BC"]
IC_GLOBAL_90 = meta["IC_GLOBAL_90"]
LABELS_RANGO = meta["LABELS_RANGO"]
BINS_RANGO   = meta["BINS_RANGO"]
IC_POR_RADIO = lookups.get("IC_POR_RADIO", {})

# ── Opciones completas ────────────────────────────────────────────────
RADIOS   = ["Radio La Inolvidable", "Radio Moda", "Radio Mágica", "Radio Nueva Q", "Radio Planeta", "Radio Ritmo Romántica", "Radiomar"]
FORMATOS = ["Texto", "link", "photo", "video"]

CATEGORIAS = ["arte", "astrologia", "ciencia", "clima", "cultura", "deportes", "economia", "educacion", "espectaculos", "espectáculos", "estilo de vida", "gastronomia", "historia", "humor", "medio ambiente", "moda", "motivacional", "musica", "narrativo", "negocios", "no aplica", "noticias", "opinión", "política", "psicologia", "religion", "romance", "salud y bienestar", "sociedad", "tecnologia", "viajes", "videojuegos"]
SUBCATS = ["analisis", "anecdota", "anuncio", "baile", "behind the scenes", "celebracion", "clip", "comentario", "comunicado oficial", "concierto", "concurso", "consejo", "conversación", "cronica", "declaraciones", "desafío", "documental", "ensayo", "entretenimiento", "entrevista", "escena de telenovela", "evento", "exhibición", "festival", "guía", "horoscopo", "humor", "interaccion con audiencia", "interacción con audiencia", "interes humano", "juego", "juego de palabras", "juego interactivo", "lanzamiento", "meet and greet", "meme", "mensaje", "narrativa personal", "no aplica", "noticia", "noticia entretenimiento", "noticia internacional", "noticia local", "opinión", "partido", "pregunta abierta", "pregunta interactiva", "premiacion", "premios", "presentacion", "propuesta de matrimonio", "ranking", "reflexion", "reportaje", "reseña", "reto", "rutina de ejercicios", "saludo", "saludo especial", "segmento radial", "sesion en vivo", "sorteo", "spoiler", "testimonio", "transmision especial", "trend", "tributo", "tutorial", "video"]
TONOS_EMO = ["admiracion", "alegria", "amor", "celebratorio", "confianza", "curiosidad", "emocion", "emotivo", "empatia", "empoderamiento", "entusiasmo", "esperanza", "gratitud", "humor", "indignacion", "inspiracion", "miedo", "neutro", "no aplica", "nostalgia", "orgullo", "preocupacion", "satisfaccion", "serenidad", "sorpresa", "ternura", "tristeza"]
ESTILOS = ["celebratorio", "conversacional", "critico", "descriptivo", "didactico", "emotivo", "formal", "humoristico", "informar emocionar", "informativo", "inspirador", "narrativo", "no aplica", "opinativo", "persuasivo", "polemico", "publicitario", "sarcastico", "testimonial"]
SITUACIONES = ["actuacion", "alfombra roja", "anecdota", "anuncio", "análisis", "behind the scenes", "cantando en la ducha", "celebracion", "comprando", "comunicado oficial", "concierto", "concierto, homenaje, entrevista, alfombra roja", "concurso en vivo", "conferencia de prensa", "controversia", "conversacion", "conversación", "declaraciones", "encuentro", "ensayo", "entrenamiento", "entrevista", "escena dramática", "evento en vivo", "festival", "filmacion", "hecho insólito", "homenaje", "hospitalización", "incidente", "interaccion social", "interacción social", "juego en vivo", "lectura de horóscopo", "manifestación", "meet and greet", "narracion de anecdota", "narrativa personal", "no aplica", "participacion en trend", "partido", "premiacion", "preparacion", "presentacion", "presentacion de disco", "programa en directo", "programa en directo, entrevista", "propuesta de matrimonio", "recuperacion", "reportaje", "reto", "reunion", "sesion de fotos", "sesion en vivo", "testimonio", "tour promocional", "transmision especial", "tráfico", "viaje", "videoclip", "visita"]
TIPOS_CTA = ["aplaudir", "asistir", "ayudar", "comentar", "comprar", "contactar", "denunciar", "descargar app", "escuchar", "etiquetar", "gritar", "llamar", "no aplica", "participar", "planificar", "prepararse", "reflexionar", "registrarse", "respetar", "seguir", "sintonizar", "subir contenido", "usar hashtag", "ver", "visitar", "votar"]
TIPOS_ENLACE = ["app store", "articulo web", "evento", "facebook", "instagram", "landing page", "no aplica", "sitio oficial", "tienda online", "tiktok", "whatsapp", "youtube"]
TIPOS_HUMOR = ["doble sentido", "humor absurdo", "humor blanco", "humor físico", "humor situacional", "juegos de palabras", "memes", "no aplica", "parodia", "sarcasmo"]
EMOCIONES_SEC = ["admiracion", "alegria", "alivio", "amor", "asco", "celebratorio", "confianza", "confusion", "curiosidad", "empatia", "empoderamiento", "entusiasmo", "esperanza", "gratitud", "humor", "indignación", "inspiracion", "intriga", "ira", "melancolia", "miedo", "no aplica", "nostalgia", "orgullo", "preocupacion", "resentimiento", "satisfaccion", "serenidad", "sorpresa", "ternura", "tristeza", "verguenza"]

CELEBRIDADES = ["ada chura", "adrien brody", "agatha lys", "agua marina", "aguamarina", "aida martínez", "al mckay", "alberto terrazos", "ale", "ale baigorria", "ale fuller", "ale seijas", "alejandra baigorria", "alejandra baigorria, said palao", "alejandra guerrero", "alejandra guzmán", "alejandro lerner", "alejandro sanz", "alejandro villagómez", "alessia rovegno", "alexander blas", "alexandra flores", "alexandra mendez", "amaya hermanos", "amor rebelde", "amy gutiérrez", "américo", "ana claudia", "ana claudia urbina", "ana lucia urbina", "ana lucía", "ana lucía urbina", "ana paula consorte", "ana siucho", "anahí", "anahí de cárdenas", "analu", "analú", "andrea bosio", "andrea torres", "andrés hurtado", "andy", "andy byron", "andy montañez", "anelí", "angelique boyer", "angelique boyer, sebastián rulli", "anghello chávez", "angélica vale", "anne hathaway", "anthony", "antonio cartagena", "anuel", "anuel aa", "anuel, yailin", "aracely", "aracely ordóñez", "arcangel", "arcángel", "ariana gonzáles y steven franco", "ariana grande", "austin palao", "avicii", "axel medina", "azucena calvay", "bad bunny", "banksy", "barbarita de cuba", "baruj ocharán", "bee gees", "beele", "belinda", "belinda, cazzu", "bella luz", "benny blanco", "benson boone", "beyoncé", "beéle", "billie eilish", "bizarrap", "blanca ramirez", "blanca ramírez", "brenda song", "brendan fraser", "brian may", "briela", "briela cirilo", "britney spears", "bruce willis", "brunella torpoco", "bruno agostini", "bruno mars", "bryan torres, samahara lobatón", "cameron boyce", "camila", "camila escribens", "camila rodríguez", "camila talavera", "camilo", "camilo blanes ornelas", "camilo sesto", "camilo y evaluna", "camilín", "camucha negrete", "carla garcía", "carlo ancelotti", "carloncho", "carlos vílchez", "carolina braedt", "carolina jaramillo", "carolina vives", "cassandra sánchez de lamadrid", "cazzu", "celia cruz", "chabuca granda", "charlie carmona", "chayanne", "chechito", "cher", "chris alegría", "chris martin", "chris soifer", "christian cueva", "christian cueva, jefferson farfán", "christian domínguez", "christian nodal", "christian nodal, ángela aguilar", "christian rodríguez portugal", "christian thorsen", "christian yaipén", "cielo fernández", "clavito y su chela", "corazón serrano", "cris", "cristian castro", "cristiano ronaldo", "cristorata", "cuto guadalupe", "cyndi lauper", "césar bk", "césar évora", "daddy yankee", "dani daniel", "daniel curtis lee", "daniela cilloniz", "daniela darco", "daniela darcourt", "daniela feijoó", "danna", "darinka ramírez", "david almandoz", "david gilmour", "de la ghetto", "demi lovato", "demi moore", "dennys quevedo", "devon werkheiser", "deyvis orosco", "deyvis paredes", "dilbert aguilar", "dina paucar", "dj will", "don javier yaipén", "don víctor yaipén", "donna summer", "donnie yaipén", "dr. alejandro cruzata martínez", "dua lipa", "dulce maría", "earth, wind & fire experience by al mckay", "ebony delgado", "ed sheeran", "eddie santiago", "edison flores", "eduardo santamarina", "edwin guerrero", "edwin guerrero neira", "edwin sierra", "edwin sierra, oscar del rio", "edwin, oscar", "el brayan", "el chico de las noticias", "eladio carrión", "elijah wood", "elton john", "elvis presley", "elías montalvo", "emanuel noir", "eminem", "emmanuel", "erick osores", "eryc castillo", "estrella torres", "ethel pozo", "evaluna", "evaluna montaner", "farfán", "farruko", "feid", "fernanda urbina", "fernando armas", "fernando colunga", "fiorella cayo", "fito páez", "flavia laos", "flavia lópez", "florinda meza", "foster the people", "francisco cavero", "frank mendizabal", "frank sinatra", "frankie ruiz", "freddie mercury", "freddy morales", "fuerza regida", "fátima bosch", "gaby spanic", "george michael", "georgina rodríguez", "gerard piqué", "giacomo bocchio", "gian marco", "gian piero díaz", "giani", "gianpiero", "gilberto santa rosa", "gilberto tapia", "gino pesaressi", "gisela valcárcel", "giuliana rengifo", "gloria trevi", "gran orquesta internacional", "greeicy", "greeicy rendón", "grupo 5", "guillermo dávila", "guillermo rossini", "guns n' roses", "gustavo salcedo", "hailey bieber", "hanna de haash", "harry styles", "hector boza", "heidi klum", "hermanos yaipén", "hernán barcos", "hernán romero", "hugo garcía", "hugo garcía, isabella ladera", "hyuna", "héctor lavoe", "ibai", "ibai llanos", "irvin saavedra", "isabel enriquez", "isabel lascurain", "isabella ladera", "isadora figueroa", "ishowspeed", "ismael miranda", "itatí cantoral", "j balvin", "jackson mora", "jaime chincha", "jairo varela", "james gunn", "jamie lee curtis", "janet barboza", "jaze", "jean pierre puppi", "jeanette", "jefferson farfán", "jerry rivera", "jesaaelys", "jesaaelys ayala", "jhay cortez", "jin", "joe jonas", "joe jonas y demi lovato", "joe keery", "john lennon", "jonas brothers", "jonathan", "jorge fossati", "jorge henderson", "josie diez canseco", "josie totah", "josimar", "josé alberto 'el canario'", "josé jerí", "josé josé", "josé luis perales", "josé luis rodríguez", "josé luis rodríguez 'el puma'", "jou mend", "jowell y randy", "juan carlos hurtado", "juan carlos ramírez", "juan gabriel", "juan román riquelme", "julian zucchi", "justin bieber", "justin y hailey bieber", "kanye west", "kapo", "karen schwarz", "karla bacigalupo", "karol g", "kassandra chanamé", "kate candela", "katteyes", "katy perry", "katy perry, orlando bloom", "katy sheen", "kendall jenner", "kendrick lamar", "kenia os", "kevin jonas", "kevin quevedo", "kevin y karla", "key candela", "kiara", "kiara franco", "kiara lozano", "kike", "kike farro", "kike vega", "kimberly", "koki", "koky salgado", "korina rivadeneira", "kylie jenner", "la bella luz", "la charanga habanera", "la india", "la segura", "la única tropical", "lady gaga", "lamine yamal", "laura bozzo", "laura pausini", "laura spoya", "lee jung-jae", "lele pons", "leo dan", "leonardo dicaprio", "leslie moscoso", "leslie shaw", "leslie stewart", "lesly carol", "lesly águila", "león xiv", "liam payne", "lily collins", "lisa", "los 4", "los conquistadores de la salsa", "los mirlos", "lucero", "lucho barrunto", "lucho cuéllar", "luciana fuster", "luis abanto morales", "luis enrique", "luis fonsi", "luis hans", "luis miguel", "lunella torpoco", "mac miller", "macarena gastaldo", "mackeily lujan", "mackeily luján", "mafer neyra", "magaly medina", "magaly solier", "magdyel ugaz", "maisak", "maju mantilla", "maluma", "mamá charo", "manuel mijares", "marc anthony", "marc anthony y nadia ferreira", "marcelo tinelli", "marco antonio solís", "maria grazia polanco", "mariah carey", "mariana de la vega", "marina", "marina yafac", "mario hart", "mario irivarren", "mario kramarenco", "marisol", "maroon 5", "martha ofelia galindo", "maría antonieta de las nieves", "maría pía copello", "maría pía, karina rivera", "massiel", "mathias zevallos", "mauricio mesones", "mayra couto", "mayra goñi", "mayte lascurain", "melissa klug", "melissa klug, jesús barco", "melissa lobatón", "melissa paredes", "meryl streep", "michael hutchence", "michael jackson", "micheille soifer", "michelle trachtenberg", "mick jagger", "miguelito el heredero", "mike bahía", "milagros díaz", "milagros leiva", "milena warthon", "milett figueroa", "miley cyrus", "mimmy succar", "minho, kai", "miss jamaica", "mnzr", "moncho rivera", "mora", "myriam hernández", "nadeska widausky", "natalia salas", "nataly ramírez", "natti natasha", "naty ramirez", "nick jonas", "nicki minaj", "nicki nicole", "nickol sinchi", "nicky jam", "nicol", "nicola porcella", "nicole faverón", "nicole sinchi", "no aplica", "olivia newton-john", "omar courtz", "onelia molina", "orlando bloom", "orquesta son del duke", "oscar del rio", "oscar ibáñez", "oscar junior custodio", "ozuna", "ozzy osbourne", "pablito ruiz", "pablo heredia", "paco bazán", "paco y susana", "paloma fiuza", "pamela franco", "pamela lópez", "paola rubio", "paolo guerrero", "papa francisco", "papa león xiv", "papa león xvi", "papo rosario", "paquita la del barrio", "paquito bazán", "paris jackson", "pati lorena", "patricio parodi", "paul flores", "paul flores garcia", "paul mccartney", "paulito fg", "paulo londra", "peppa pig", "phil collins", "pierina carcelén", "pimpinela", "pink floyd", "pochita", "prince royce", "queen", "rafael ithier", "raphael", "raphy pina", "rauw alejandro", "rebeca escribens", "reimond manco", "renata flores", "renato rossini", "renato tapia", "renzo palacios", "renée león", "ricardo montaner", "richard clayderman", "rihanna", "robbie williams", "robert francis prevost", "robert muñoz", "robert prevost", "roberto carlos", "roberto guizasola", "roberto gómez bolaños", "rocío dúrcal", "rodrigo cuba", "rodrigo de paul", "romeo santos", "rosalía", "rosángela espinoza", "rosé", "roxana molina", "roxy", "rubén blades", "rumi", "rute cardoso", "sabrina carpenter", "said palao", "said palao y alejandra baigorria", "samahara lobatón", "samantha batallanos", "santa rosa de lima", "sasha, milán", "sean lennon", "sebastián britos", "sebastián yaipén", "selena gomez", "sergio romero chechito", "shakira", "shania twain", "shawn mendes", "sheyla rojas", "shirley arica", "snoop dogg", "sofía vergara", "son del duke", "son del duque", "son tentación", "sonia rey", "sor de bogotá", "stephanie cayo", "stephanie lii", "stephanie orué", "steve aoki", "steven franco", "sting", "stiven franco", "suheyn cipriani", "sully saénz", "susana", "susana alvarado", "susana alvarado, paco bazán", "susana baca", "susy díaz", "suu rabanal", "sydney sweeney", "tamara gómez", "tania pinedo", "taylor swift", "tekashi", "thalía", "thamara gómez", "thamara medina", "thamara medina alcalá", "the beatles", "the bee gees", "the police", "the rolling stones", "tiago pzk", "tilsa lozano", "timothée chalamet", "tina turner", "tini", "tini stoessel", "tito 'el bambino' y lennox", "tito nieves", "tom holland", "tony cam", "tony succar", "tony y mimmy succar", "travis kelce", "tula rodríguez", "tyla", "valentina ferrer", "valentino lázaro", "valeria zapata", "vanessa hudgens", "vania bludau", "vasco madueño", "verónica castro", "víctor manuelle", "víctor yaipén", "víctor yaipén uypan", "waldir felipa", "whitney houston", "william levy", "willie colón", "willie gonzalez", "willie gonzález", "wilmer lozano", "wilson manyoma", "wisin", "xiomy kanashiro", "yahaira plasencia", "yailin", "yailin la más viral", "yandel", "yarita lizeth", "yefri chunga", "yiddá eslava", "yoko ono", "yoshimar yotún", "youna", "yrma guerrero", "yuri", "zac efron", "zayn malik", "zion", "zoe saldaña", "zully", "álvaro rod", "ángela aguilar", "ángeles blancos", "óscar carrillo vértiz", "óscar custodio", "óscar d'león", "óscar d’león"]

# ── Predicción ────────────────────────────────────────────────────────
def _ic_radio(radio):
    key = radio.replace("Radio ","").replace("Radio","").strip()
    entry = IC_POR_RADIO.get(key, {})
    ic = entry.get("ic_honesto", IC_GLOBAL_90)
    return float(ic) if ic and not (isinstance(ic,float) and np.isnan(ic)) else IC_GLOBAL_90

def _rango(monto):
    for i in range(len(BINS_RANGO)-1):
        if BINS_RANGO[i] <= monto < BINS_RANGO[i+1]:
            return LABELS_RANGO[i]
    return LABELS_RANGO[-1]

def predecir(post_data):
    df_f      = transform_inputs(post_data, meta, lookups, enc)
    prob      = float(clf.predict_proba(df_f)[:,1][0])
    monto_raw = float(np.clip(inv_boxcox(reg.predict(df_f)[0], LAMBDA_BC), 0, None))
    monto     = prob * monto_raw
    ic        = _ic_radio(str(post_data.get("Radio","")))
    proba_m2  = clf_m2.predict_proba(df_f)[0]
    idx_m2    = int(clf_m2.predict(df_f)[0])
    rango_m1  = _rango(monto)
    rango_m2  = LABELS_RANGO[idx_m2]
    s_prob    = prob * 100
    s_monto   = min(monto/35, 1)*100
    s_ic      = max(0, 1 - ic/50)*100
    s_cons    = 100.0 if rango_m1==rango_m2 else 0.0
    score     = round(0.40*s_prob + 0.25*s_monto + 0.20*s_ic + 0.15*s_cons, 1)
    if score >= 75:   et,em,tip = "Alta probabilidad","🟢","El modelo sugiere publicar este post."
    elif score >= 50: et,em,tip = "Potencial medio","🟡","Considera optimizar el copy o la franja horaria."
    else:             et,em,tip = "Bajo potencial","🔴","Revisa el formato o el tema del contenido."
    return {
        "prob":prob,"monto":round(monto,2),
        "ic_inf":round(max(0,monto-ic),2),"ic_sup":round(monto+ic,2),"ic":round(ic,2),
        "rango_m1":rango_m1,"rango_m2":rango_m2,
        "conf_m2":round(float(proba_m2.max()),4),
        "probs_m2":{LABELS_RANGO[i]:round(float(p),4) for i,p in enumerate(proba_m2)},
        "score":score,"etiqueta":et,"emoji":em,"tip":tip,
        "detalle":{
            "Prob. ganar × 40%":    round(s_prob,1),
            "Monto est. × 25%":     round(s_monto,1),
            "Confianza IC × 20%":   round(s_ic,1),
            "Consistencia × 15%":   round(s_cons,1),
        }
    }

def analizar_timings(post_data):
    """Corre los 24 horarios y devuelve lista ordenada por score."""
    rows = []
    for h in range(24):
        pd_copy = {**post_data, "hora": h}
        try:
            r = predecir(pd_copy)
            rows.append({"hora":h,"franja":franja_horaria(h),"score":r["score"],
                         "prob":r["prob"],"monto":r["monto"],"rango":r["rango_m2"]})
        except:
            rows.append({"hora":h,"franja":franja_horaria(h),"score":0,
                         "prob":0,"monto":0,"rango":"$0"})
    return sorted(rows, key=lambda x: x["score"], reverse=True)

# ── Helpers render ────────────────────────────────────────────────────
BADGE_CLS = {"$0":"b0","$0-$5":"b1","$5-$10":"b2","$10-$35":"b3",">$35":"b4"}

def score_color(s):
    if s >= 75: return "#38A84B"
    if s >= 50: return "#F5B82E"
    return "#E8343A"

def render_resultado(res, hora_sel=None):
    sc   = res["score"]
    col  = score_color(sc)

    # Score hero
    st.markdown(f"""
    <div class="score-hero">
      <div class="score-hero-accent" style="background:{col}"></div>
      <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.12em;color:var(--muted);margin-bottom:0.5rem;">
        Score de publicación
      </div>
      <div class="score-number" style="color:{col}">{sc}</div>
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:{sc}%;background:{col}"></div>
      </div>
      <div style="display:flex;justify-content:space-between;
                  font-size:0.63rem;color:var(--muted2);margin-top:0.2rem;">
        <span>0</span><span>50</span><span>100</span>
      </div>
      <div class="score-label" style="color:{col}">{res['emoji']} {res['etiqueta']}</div>
      <div class="score-tip">{res['tip']}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # Chips
    bc1 = BADGE_CLS.get(res["rango_m1"],"b0")
    bc2 = BADGE_CLS.get(res["rango_m2"],"b0")
    cons_html = (
        '<span style="color:#38A84B;font-size:0.72rem;font-weight:600">✓ Coinciden</span>'
        if res["rango_m1"]==res["rango_m2"] else
        '<span style="color:#F5B82E;font-size:0.72rem;font-weight:600">⚠ Difieren</span>'
    )
    st.markdown(f"""
    <div class="chips-grid">
      <div class="chip">
        <div class="chip-label">Prob. de ganar</div>
        <div class="chip-val">{res['prob']*100:.1f}%</div>
        <div class="chip-sub">Clasificador binario M1</div>
      </div>
      <div class="chip">
        <div class="chip-label">Monto estimado</div>
        <div class="chip-val">${res['monto']:.2f}</div>
        <div class="chip-sub">IC 90%: ${res['ic_inf']} — ${res['ic_sup']}</div>
      </div>
      <div class="chip">
        <div class="chip-label">Rango M1 · M2</div>
        <div class="chip-val" style="font-size:0.85rem;display:flex;gap:0.3rem;align-items:center;">
          <span class="badge {bc1}">{res['rango_m1']}</span>
          <span class="badge {bc2}">{res['rango_m2']}</span>
        </div>
        <div class="chip-sub">{cons_html}</div>
      </div>
      <div class="chip">
        <div class="chip-label">Confianza M2</div>
        <div class="chip-val">{res['conf_m2']*100:.1f}%</div>
        <div class="chip-sub">Clasificador 5 rangos</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Desglose
    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">Desglose del score</div>', unsafe_allow_html=True)
    for lbl, val in res["detalle"].items():
        st.markdown(f"""
        <div class="row-desglose">
          <span style="color:var(--muted)">{lbl}</span>
          <span style="font-weight:700">{val:.1f}<span style="font-weight:400;color:var(--muted2);font-size:0.72rem"> /100</span></span>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Probabilidades por rango (M2)"):
        cols = st.columns(5)
        for i,(r,p) in enumerate(res["probs_m2"].items()):
            bc = BADGE_CLS.get(r,"b0")
            with cols[i]:
                st.markdown(f"""
                <div style="text-align:center;padding:0.5rem 0;">
                  <span class="badge {bc}">{r}</span>
                  <div style="font-weight:700;font-size:1.05rem;margin-top:0.35rem">{p*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)


def render_timing(timings_sorted, hora_sel):
    """Top 5 mejores horarios + posición del horario elegido."""
    st.markdown('<div class="card"><div class="card-title">⏰ Top 5 mejores horarios para publicar</div>', unsafe_allow_html=True)

    top5 = timings_sorted[:5]
    horas_top5 = [t["hora"] for t in top5]
    score_max = timings_sorted[0]["score"] if timings_sorted else 100

    for rank, t in enumerate(top5):
        is_sel = (t["hora"] == hora_sel)
        is_best = (rank == 0)
        row_cls  = "selected" if is_sel else "top"
        pct_bar  = (t["score"] / score_max * 100) if score_max > 0 else 0
        badge_html = ""
        if is_best and is_sel:
            badge_html = '<span class="timing-badge timing-top1">🥇 Mejor · Tu hora</span>'
        elif is_best:
            badge_html = '<span class="timing-badge timing-top1">🥇 Mejor horario</span>'
        elif is_sel:
            badge_html = '<span class="timing-badge timing-sel">✓ Tu horario</span>'

        st.markdown(f"""
        <div class="timing-row {row_cls}">
          <span class="timing-hora">{t['hora']:02d}h</span>
          <span class="timing-franja">{t['franja']}</span>
          <div class="timing-bar-wrap">
            <div class="timing-bar-fill" style="width:{pct_bar:.0f}%"></div>
          </div>
          <span class="timing-score" style="color:{score_color(t['score'])}">{t['score']:.0f}</span>
          {badge_html}
        </div>""", unsafe_allow_html=True)

    # Si la hora seleccionada no está en el top 5, mostrarla aparte
    if hora_sel not in horas_top5:
        t_sel = next((t for t in timings_sorted if t["hora"]==hora_sel), None)
        pos   = next((i+1 for i,t in enumerate(timings_sorted) if t["hora"]==hora_sel), "—")
        if t_sel:
            pct_bar = (t_sel["score"] / score_max * 100) if score_max > 0 else 0
            st.markdown(f"""
            <div style="margin-top:0.5rem;padding-top:0.5rem;border-top:1px dashed var(--border)">
              <div style="font-size:0.65rem;color:var(--muted);margin-bottom:0.3rem;font-weight:600;">
                TU HORARIO ELEGIDO — Posición #{pos} de 24
              </div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="timing-row selected">
              <span class="timing-hora">{t_sel['hora']:02d}h</span>
              <span class="timing-franja">{t_sel['franja']}</span>
              <div class="timing-bar-wrap">
                <div class="timing-bar-fill" style="width:{pct_bar:.0f}%;background:#3DB8D4"></div>
              </div>
              <span class="timing-score" style="color:{score_color(t_sel['score'])}">{t_sel['score']:.0f}</span>
              <span class="timing-badge timing-sel">Tu hora</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── Formulario ────────────────────────────────────────────────────────
def formulario(key_prefix=""):
    with st.form(f"form_{key_prefix}", clear_on_submit=False):
        st.markdown('<div class="card-title">📋 Datos del post</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            radio   = st.selectbox("Radio *", RADIOS)
            formato = st.selectbox("Formato *", FORMATOS)
            fecha   = st.date_input("Fecha *", value=date.today())
            hora    = st.slider("Hora de publicación *", 0, 23, 15, format="%dh")
        with c2:
            cat    = st.selectbox("Categoría *", CATEGORIAS)
            subcat = st.selectbox("Subcategoría", SUBCATS)
            tono   = st.selectbox("Tono emocional *", TONOS_EMO)
            estilo = st.selectbox("Estilo comunicativo *", ESTILOS)

        copy = st.text_area("Copy del post *", placeholder="Escribe o pega el texto del post…", height=90)
        link = st.text_input("Link (opcional)", placeholder="https://…")

        with st.expander("Parámetros adicionales (opcional)"):
            c3, c4, c5 = st.columns(3)
            with c3:
                cta        = st.selectbox("Llamado a la acción", TIPOS_CTA)
                tipo_enlace= st.selectbox("Tipo de enlace", TIPOS_ENLACE)
            with c4:
                situacion  = st.selectbox("Situación / contexto", SITUACIONES)
                celebridad = st.selectbox("Celebridad principal", ["no aplica"] + CELEBRIDADES, key="t1celeb")
            with c5:
                tipo_humor = st.selectbox("Tipo de humor", TIPOS_HUMOR)
                emosec     = st.selectbox("Emoción secundaria", EMOCIONES_SEC)

        submitted = st.form_submit_button("Analizar post →", use_container_width=True)

    if submitted:
        if not copy.strip():
            st.error("El Copy es obligatorio.")
            return None
        return {
            "Radio":radio,"Formato":formato,
            "fecha":pd.Timestamp(fecha),"hora":hora,
            "categoria_contenido":cat,"subcategoria_contenido":subcat,
            "tono_emocional_principal":tono,"estilo_comunicativo":estilo,
            "tipo_llamado_a_la_accion":cta,"tipo_enlace":tipo_enlace,
            "situacion":situacion,
            "celebridad_principal":celebridad.strip() or "no aplica",
            "tipo_humor":tipo_humor,"emocion_secundaria":emosec,
            "Copy":copy,"Link":link.strip() or None,
        }
    return None


# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="crp-topbar">
  <div class="crp-logo">
    <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10
      10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z"/></svg>
  </div>
  <div>
    <div class="crp-title">Score Predictor · Redes Sociales</div>
    <div class="crp-subtitle">CRP Medios y Entretenimiento SAC &nbsp;·&nbsp; Herramienta interna de análisis de contenido</div>
  </div>
  <div class="crp-pill">● Modelo v17 activo</div>
</div>
""", unsafe_allow_html=True)

# ── Pestañas ──────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["  📋  Analizar un post  ", "  📊  Comparar múltiples posts  "])

# ─────────────────────────────── TAB 1 ───────────────────────────────
with tab1:
    cf, cr = st.columns([1.05, 0.95], gap="large")

    with cf:
        post_data = formulario("t1")

    with cr:
        if post_data:
            try:
                with st.spinner("Analizando…"):
                    res      = predecir(post_data)
                    timings  = analizar_timings(post_data)
                st.session_state["res1"]     = res
                st.session_state["timings1"] = timings
                st.session_state["hora1"]    = post_data["hora"]
            except Exception as e:
                st.error(f"Error: {e}")

        if "res1" in st.session_state:
            render_resultado(st.session_state["res1"], st.session_state.get("hora1"))
            st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
            render_timing(st.session_state["timings1"], st.session_state.get("hora1", 0))
        else:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;padding:4rem 2rem;text-align:center;
                        background:var(--surface);border:1px solid var(--border);
                        border-radius:var(--radius);margin-top:2rem;">
              <div style="font-size:2rem;margin-bottom:0.75rem;">📡</div>
              <div style="font-weight:700;font-size:0.95rem;margin-bottom:0.4rem">
                Completa el formulario</div>
              <div style="color:var(--muted);font-size:0.79rem;max-width:240px;line-height:1.6">
                Ingresa los datos del post y haz clic en
                <strong style="color:var(--crp-green)">Analizar post →</strong>
                para ver el score, predicción de ingresos y los mejores horarios.
              </div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────── TAB 2 ───────────────────────────────
with tab2:
    if "cola" not in st.session_state:
        st.session_state["cola"] = []

    ca, cl = st.columns([1.05, 0.95], gap="large")

    with ca:
        st.markdown('<div class="card-title">➕ Agregar post a la cola</div>', unsafe_allow_html=True)
        with st.form("form_t2", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                r2 = st.selectbox("Radio *",   RADIOS,   key="t2r")
                f2 = st.selectbox("Formato *", FORMATOS, key="t2f")
                d2 = st.date_input("Fecha *", value=date.today(), key="t2d")
                h2 = st.slider("Hora *", 0, 23, 15, format="%dh", key="t2h")
            with c2:
                ca2 = st.selectbox("Categoría *",    CATEGORIAS, key="t2ca")
                sb2 = st.selectbox("Subcategoría",   SUBCATS,    key="t2sb")
                to2 = st.selectbox("Tono emocional", TONOS_EMO,  key="t2to")
                es2 = st.selectbox("Estilo",         ESTILOS,    key="t2es")
            cp2 = st.text_area("Copy *", placeholder="Texto del post…", height=75, key="t2cp")
            lk2 = st.text_input("Link (opcional)", key="t2lk")

            with st.expander("Opcionales"):
                cc1,cc2,cc3 = st.columns(3)
                with cc1:
                    cta2 = st.selectbox("CTA",         TIPOS_CTA,    key="t2cta")
                    te2  = st.selectbox("Tipo enlace",  TIPOS_ENLACE, key="t2te")
                with cc2:
                    si2  = st.selectbox("Situación",   SITUACIONES,  key="t2si")
                    ce2  = st.selectbox("Celebridad", ["no aplica"] + CELEBRIDADES, key="t2ce")
                with cc3:
                    th2  = st.selectbox("Tipo humor",  TIPOS_HUMOR,     key="t2th")
                    em2  = st.selectbox("Emoción sec.", EMOCIONES_SEC,  key="t2em")

            agregar = st.form_submit_button("➕  Agregar a la cola", use_container_width=True)

        if agregar:
            if not cp2.strip():
                st.error("El Copy es obligatorio.")
            elif len(st.session_state["cola"]) >= 50:
                st.warning("Máximo 50 posts por sesión.")
            else:
                st.session_state["cola"].append({
                    "Radio":r2,"Formato":f2,"fecha":pd.Timestamp(d2),"hora":h2,
                    "categoria_contenido":ca2,"subcategoria_contenido":sb2,
                    "tono_emocional_principal":to2,"estilo_comunicativo":es2,
                    "tipo_llamado_a_la_accion":cta2,"tipo_enlace":te2,
                    "situacion":si2,
                    "celebridad_principal":ce2.strip() or "no aplica",
                    "tipo_humor":th2,"emocion_secundaria":em2,
                    "Copy":cp2,"Link":lk2.strip() or None,
                    "_prev":cp2[:45]+("…" if len(cp2)>45 else ""),
                })
                st.success(f"Post agregado. Cola: {len(st.session_state['cola'])}/50")
                st.rerun()

    with cl:
        n = len(st.session_state["cola"])
        st.markdown(f'<div class="card-title">🗂️ Cola de posts ({n} / 50)</div>', unsafe_allow_html=True)

        if n == 0:
            st.markdown("""
            <div style="text-align:center;padding:2rem;color:var(--muted);
                        font-size:0.79rem;background:var(--surface2);
                        border:1px solid var(--border);border-radius:var(--radius)">
              Agrega al menos un post para comenzar.
            </div>""", unsafe_allow_html=True)
        else:
            for i, p in enumerate(st.session_state["cola"]):
                ci, cd = st.columns([6, 1])
                with ci:
                    st.markdown(f"""
                    <div class="post-row">
                      <div class="post-row-title">#{i+1} · {p['Radio']} · {p['Formato']}</div>
                      <div class="post-row-meta">{p['_prev']}</div>
                      <div class="post-row-meta">{p['fecha'].strftime('%d/%m/%Y')} · {p['hora']}h · {p['categoria_contenido']}</div>
                    </div>""", unsafe_allow_html=True)
                with cd:
                    if st.button("✕", key=f"del{i}"):
                        st.session_state["cola"].pop(i)
                        if "ranking" in st.session_state: del st.session_state["ranking"]
                        st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            with b1:
                run_all = st.button("🔍  Predecir todos", use_container_width=True)
            with b2:
                if st.button("🗑️  Limpiar", use_container_width=True):
                    st.session_state["cola"] = []
                    if "ranking" in st.session_state: del st.session_state["ranking"]
                    st.rerun()

            if run_all and n > 0:
                rows, prog = [], st.progress(0, text="Analizando…")
                for i, p in enumerate(st.session_state["cola"]):
                    try:
                        r = predecir(p)
                        rows.append({
                            "#":i+1,"Radio":p["Radio"],"Formato":p["Formato"],
                            "Hora":f"{p['hora']}h","Copy":p["_prev"],
                            "Score":r["score"],"Etiqueta":f"{r['emoji']} {r['etiqueta']}",
                            "Prob.":f"{r['prob']*100:.1f}%","Monto":f"${r['monto']:.2f}",
                            "IC 90%":f"${r['ic_inf']}–${r['ic_sup']}",
                            "Rango M1":r["rango_m1"],"Rango M2":r["rango_m2"],
                            "✓":("✓" if r["rango_m1"]==r["rango_m2"] else "⚠"),
                        })
                    except Exception as e:
                        rows.append({"#":i+1,"Radio":p["Radio"],"Score":-1,"Etiqueta":f"❌ {e}"})
                    prog.progress((i+1)/n, text=f"Analizando {i+1}/{n}…")
                prog.empty()
                st.session_state["ranking"] = pd.DataFrame(rows).sort_values("Score", ascending=False)

    if "ranking" in st.session_state:
        st.markdown("---")
        st.markdown('<div class="card-title">🏆 Ranking de posts por score</div>', unsafe_allow_html=True)
        df = st.session_state["ranking"]
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={"Score": st.column_config.ProgressColumn(
                         "Score", min_value=0, max_value=100, format="%.1f")})
        st.download_button("⬇️  Descargar CSV",
                           data=df.to_csv(index=False).encode("utf-8"),
                           file_name="crp_ranking.csv", mime="text/csv")
