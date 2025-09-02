# -*- coding: utf-8 -*-
import langid
from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import unicodedata
import html as htmllib

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
try:
    from spacy.lang.es.stop_words import STOP_WORDS as ES_STOP
except Exception:
    # fallback mini-list si no tienes spaCy instalado
    ES_STOP = {
        "a","al","algo","algunas","algunos","ante","antes","como","con","contra","cual","cuando",
        "de","del","desde","donde","durante","e","el","ella","ellas","ellos","en","entre",
        "era","eran","eres","es","esa","esas","ese","eso","esos","esta","estaba","estaban",
        "estás","está","están","este","estos","esto","fue","fueron","fui","fuimos","haber",
        "había","habían","han","hasta","hay","la","las","le","les","lo","los","más","me",
        "mi","mientras","muy","nada","ni","no","nos","nosotros","nuestra","nuestro","o","os",
        "otra","otras","otro","otros","para","pero","poco","por","porque","que","quien",
        "quienes","se","ser","será","si","sin","sobre","sois","somos","son","soy","su","sus",
        "te","tenemos","tener","tengo","ti","tu","tus","un","una","unas","uno","unos","y","ya"
    }

# --- Modelos globales ---
kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

# Traductor ES->EN (solo se usa si el idioma detectado es español)
_tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
_mt = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")

def strip_diacritics(s: str) -> str:
    # NFKD separa base + diacrítico; quitamos diacríticos (acentos, tildes)
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )

# --- util: quitar HTML/URLs de forma segura sin dependencias ---
TAG_SPACERS = re.compile(
    r"</?(?:br|p|div|li|ul|ol|h[1-6]|tr|td|th|blockquote|section|article|span)\b[^>]*>", re.I
)
SCRIPTS = re.compile(r"<script\b[^>]*>.*?</script>", re.I | re.S)
STYLES  = re.compile(r"<style\b[^>]*>.*?</style>",  re.I | re.S)
ALL_TAGS = re.compile(r"<[^>]+>")
URLS = re.compile(r"https?://\S+|www\.\S+", re.I)

def strip_html(text: str) -> str:
    if not text:
        return ""
    # 1) normalizar entidades primero para no perder texto (&amp; -> &)
    text = htmllib.unescape(text)

    # 2) eliminar scripts y styles completos
    text = SCRIPTS.sub(" ", text)
    text = STYLES.sub(" ", text)

    # 3) sustituir tags “de bloque” por espacios para no pegar palabras
    text = TAG_SPACERS.sub(" ", text)

    # 4) quitar cualquier tag restante
    text = ALL_TAGS.sub(" ", text)

    # 5) quitar URLs
    text = URLS.sub(" ", text)

    return text

def clean_lyrics(text: str) -> str:
    if not text:
        return ""

    # A) quitar HTML y urls
    text = strip_html(text)

    # B) bajar a minúsculas y eliminar saltos por espacios
    text = text.lower().replace("\r", " ").replace("\n", " ")

    # C) quitar anotaciones tipo [chorus], [verse], etc.
    text = re.sub(r"\[.*?\]", " ", text)

    # D) quitar diacríticos (á→a, ñ→n, ü→u)
    text = strip_diacritics(text)

    # E) dejar solo letras y espacios (si quieres números, cambia el regex a [a-z0-9\s])
    text = re.sub(r"[^a-z\s]", " ", text)

    # F) ruido típico en canciones
    noise = [
        "la la la", "na na na", "yeah", "ya ya", "oh", "ooh", "uh", "uhh",
        "uh huh", "baby", "mmm", "ay", "ey", "hey", "woah"
    ]
    for p in noise:
        text = text.replace(p, " ")

    # G) colapsar espacios
    text = re.sub(r"\s+", " ", text).strip()

    return text

def detectar_idioma(texto: str) -> str:
    lang, _ = langid.classify(texto or "")
    return "es" if lang.startswith("es") else "en"

def extraer_keywords(texto: str, lang: str, top_n: int = 10, ngrams=(1,2)) -> list[str]:
    # usa stopwords reales según idioma
    stop = list(ES_STOP) if lang == "es" else list(ENGLISH_STOP_WORDS)

    pares = kw_model.extract_keywords(
        texto,
        keyphrase_ngram_range=ngrams,
        stop_words=stop,       # <-- aquí el cambio clave
        top_n=top_n,
        use_mmr=True,
        diversity=0.6,
        nr_candidates=max(30, top_n*4)
    )

    vistos, frases = set(), []
    for frase, _ in pares:
        f = " ".join(frase.split()).strip()
        if f and f not in vistos:
            vistos.add(f)
            frases.append(f)
    return frases

def traducir_es_en(frases_es: list[str]) -> list[str]:
    if not frases_es:
        return []
    # batch simple
    inputs = _tok(frases_es, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = _mt.generate(**inputs, max_length=128)
    traducciones = _tok.batch_decode(outputs, skip_special_tokens=True)
    # normalizar espacios y deduplicar
    vistos, out = set(), []
    for t in traducciones:
        tt = " ".join(t.split()).strip()
        if tt and tt.lower() not in vistos:
            vistos.add(tt.lower())
            out.append(tt)
    return out

def keywords_ingles(texto: str, top_n: int = 10) -> list[str]:
    """
    Devuelve SOLO keywords en inglés.
    - Si el texto está en inglés: devuelve las keywords tal cual.
    - Si está en español: extrae en español y las traduce a inglés.
    """
    lang = detectar_idioma(texto)
    print(f"Detected language: {lang}")
    kws = extraer_keywords(texto, lang=lang, top_n=top_n)
    if lang == "es":
        return traducir_es_en(kws)
    return kws

def keywords_es(texto: str, top_n: int = 10) -> list[str]:
    lang = detectar_idioma(texto)
    # si detecta inglés igual extrae, pero con stopwords EN
    return extraer_keywords(texto, lang=lang if lang in ("es","en") else "es", top_n=top_n)
