# extract_keywords/LLM.py
import os, json, re
from dotenv import load_dotenv
from openai import OpenAI


def get_client():
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=API_KEY)

def llm_keywords_en_simple(lyric: str, n: int = 10, model: str = "gpt-4.1-mini") -> list[str]:
    """
    Envía el lyric a la LLM y devuelve EXACTAMENTE n keywords en inglés (1–3 palabras).
    Sin structured outputs: parseo robusto de JSON devuelto por el modelo.
    """

    if not lyric or not lyric.strip():
        return []

    client = get_client()

    prompt = (
        f"You are a keyword extractor. Return EXACTLY {n + 5} keywords in English "
        "as a JSON array of strings. "
        "If the text is not in English, translate the keyword terms to English before returning. "
        "Be concise (1–3 words per keyword). Avoid stopwords, pronouns, near-duplicates, "
        "and especially avoid proper nouns (names of artists, people, brands, or places). "
        "The FIRST 5 keywords must capture the main themes, mood, or essence of the song. "
        f"The remaining {n} keywords should provide supporting or complementary context. "
        "Return ONLY the JSON array, without any extra text.\n\n"
        "--- TEXT ---\n"
        f"{lyric.strip()}\n"
        "----------------"
    )

    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
    )
    text = resp.output_text.strip()

    # Intentar parsear JSON directo
    def _take_n(arr, n):
        out, seen = [], set()
        for x in arr:
            k = " ".join(str(x).split()).strip()
            lk = k.lower()
            if k and lk not in seen:
                seen.add(lk)
                out.append(k)
            if len(out) == n:
                break
        return out

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return _take_n(data, n)
    except Exception:
        pass

    # Fallback: extraer el primer array [] que aparezca
    m = re.search(r"\[(?:.|\n)*\]", text)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return _take_n(data, n)
        except Exception:
            return []

    return []


def llm_keywords_from_title_artist(
        artist: str,
        title: str,
        n: int = 10,
        model: str = "gpt-4.1-mini"
) -> list[str]:
    """
    Devuelve EXACTAMENTE n+5 keywords en inglés basadas en (artist, title).
    - Las primeras 5: temas/mood/essence más representativos.
    - Las siguientes n: contexto complementario.
    - Evita proper nouns (artists, people, brands, places).
    - 1–3 palabras por keyword, sin duplicados cercanos.
    """
    artist = (artist or "").strip()
    title = (title or "").strip()
    if not artist and not title:
        return []

    total = n + 5
    client = get_client()

    prompt = (
        f"You are a keyword extractor. Using only the song metadata below "
        f"(artist and title; no external browsing), return EXACTLY {total} keywords "
        "in English as a JSON array of strings.\n\n"
        "Rules:\n"
        "• First 5 keywords = the main themes, mood, or essence of the song.\n"
        f"• Remaining {n} keywords = complementary context (style, vibe, topics, setting, emotions).\n"
        "• Be concise: 1–3 words per keyword.\n"
        "• Avoid stopwords, pronouns, and near-duplicates.\n"
        "• Critically: avoid proper nouns (no artist names, people, brands, or place names).\n"
        "• If the metadata is ambiguous, infer plausible common themes for that artist/title, "
        "but still avoid proper nouns.\n"
        "Return ONLY the JSON array, nothing else.\n\n"
        "--- METADATA ---\n"
        f"Artist: {artist}\n"
        f"Title: {title}\n"
        "----------------"
    )

    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
    )
    text = resp.output_text.strip()

    # --- parseo robusto a lista de strings y recorte a total ---
    def _take_clean(arr, k):
        out, seen = [], set()
        for x in arr:
            s = " ".join(str(x).split()).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
            if len(out) == k:
                break
        return out

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return _take_clean(data, total)
    except Exception:
        pass

    m = re.search(r"\[(?:.|\n)*\]", text)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return _take_clean(data, total)
        except Exception:
            return []

    return []
