"""
utils/helpers.py
Funciones auxiliares reutilizadas en features.py y app.py
"""
import re

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

def tiene_emoji(texto: str) -> int:
    """1 si el texto contiene emoji, 0 si no."""
    return int(bool(_EMOJI_RE.search(str(texto))))


def franja_horaria(hora: int) -> str:
    """
    Convierte hora (0-23) en franja:
      mañana  : 5-9h
      mediodia: 10-13h
      tarde   : 14-20h
      noche   : 21-4h
    """
    if 5 <= hora <= 9:
        return "mañana"
    elif 10 <= hora <= 13:
        return "mediodia"
    elif 14 <= hora <= 20:
        return "tarde"
    else:
        return "noche"
