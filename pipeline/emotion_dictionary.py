import os
import re
import json
import time
from typing import List, Dict, Any, Optional
import libraries.sqlite as db
from dotenv import load_dotenv
from openai import OpenAI


# ---------- Utilidades ----------
def _parse_json_lenient(text: str) -> Any:
    """
    Intenta parsear JSON aunque el modelo devuelva texto extra o ```json fences.
    """
    if text is None:
        raise ValueError("Empty LLM response.")
    cleaned = text.strip()

    # quitar fences ```json ... ```
    cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # intento directo
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # buscar objeto {...}
    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s != -1 and e != -1 and s < e:
        try:
            return json.loads(cleaned[s:e + 1])
        except Exception:
            pass

    # buscar array [...]
    s, e = cleaned.find("["), cleaned.rfind("]")
    if s != -1 and e != -1 and s < e:
        try:
            return json.loads(cleaned[s:e + 1])
        except Exception:
            pass

    raise ValueError(f"Could not parse JSON from LLM output. Got:\n{cleaned[:500]}")


def _canon_norm(label: str) -> str:
    """
    Normaliza etiquetas a Positive|Neutral|Negative (acepta inglés/español/minúsculas).
    """
    if not label:
        return "Neutral"
    l = label.strip().lower()
    if l.startswith(("pos", "poz", "positivo")):
        return "Positive"
    if l.startswith(("neu", "neutro")):
        return "Neutral"
    if l.startswith(("neg", "negativo")):
        return "Negative"
    return "Neutral"


# ---------- LLM ----------
def llm_classify_emotions_with_openai(
        emotions: List[str],
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 3,
        dry_run: bool = False,
        debug_print: bool = False,
) -> List[Dict[str, Any]]:
    load_dotenv()
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not emotions:
        return []
    if dry_run:
        return [{"emotion": e, "normalize": "N", "emoji": "❓"} for e in emotions]

    system_msg = '''
    You are a precise JSON generator.
    For each input emotion, return a JSON array of objects with the following keys:
    - "emotion": the emotion name,
    - "normalize": one of ["FP" (Full Positive), "MP" (Mid Positive), "N" (Neutral), "MN" (Mid Negative), "FN" (Full Negative)],
    - "emoji": a single representative emoji.

    Return ONLY a valid JSON array, with no explanations and no markdown.
    '''
    user_msg = "Emotions:\n" + json.dumps(emotions, ensure_ascii=False)

    last_text: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = _client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=1200,
            )
            text = resp.choices[0].message.content
            last_text = text
            parsed = _parse_json_lenient(text)

            # Acepta lista directa o objeto con "result"
            if isinstance(parsed, dict) and "result" in parsed and isinstance(parsed["result"], list):
                items = parsed["result"]
            elif isinstance(parsed, list):
                items = parsed
            else:
                raise ValueError("Parsed JSON is not a list nor an object with 'result'.")

            out: List[Dict[str, Any]] = []
            for row in items:
                if not isinstance(row, dict):
                    continue
                emo = str(row.get("emotion", "")).strip()
                norm = str(row.get("normalize", "")).strip()
                emoji = str(row.get("emoji", "")).strip()
                if emo and emoji:
                    out.append({"emotion": emo, "normalize": norm, "emoji": emoji})

            if not out:
                raise ValueError("Empty mapping after validation.")
            return out

        except Exception as e:
            if debug_print and last_text:
                print("LLM RAW OUTPUT (truncated):\n", last_text[:800])
            if attempt == max_retries:
                raise
            time.sleep(0.7 * attempt)


# ---------- Orquestación ----------
def build_and_store_emotions_dictionary_with_llm(
        wide_df,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        dry_run_llm: bool = False,
        verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    1) Toma columnas de wide_df (excluye 'track_spotify_id') => emociones.
    2) Llama LLM para clasificar y asignar emoji.
    3) Crea/TRUNCATE emotions_dictionary y guarda.
    4) Devuelve la lista final insertada.
    """
    # 1) Extraer nombres de emociones desde los headers del DF
    all_cols = list(wide_df.columns)
    emotions_in_df = [c for c in all_cols if c.lower() != "track_spotify_id"]
    if verbose:
        print("Emotions from wide_df:", emotions_in_df)

    if not emotions_in_df:
        raise ValueError("No emotion columns found in wide_df (other than 'track_spotify_id').")

    # 2) LLM -> mapping
    mapping = llm_classify_emotions_with_openai(
        emotions=emotions_in_df,
        model=llm_model,
        temperature=temperature,
        dry_run=dry_run_llm,
        debug_print=verbose,
    )

    # 3) Ajustar a headers exactos (por si el LLM cambia mayúsculas/minúsculas)
    by_lower_original = {c.lower(): c for c in emotions_in_df}
    cleaned: List[Dict[str, Any]] = []
    for item in mapping:
        key = item["emotion"].strip().lower()
        if key in by_lower_original:
            cleaned.append({
                "emotion": by_lower_original[key],
                "normalize": item["normalize"],
                "emoji": item["emoji"]
            })

    if not cleaned:
        raise ValueError("LLM produced no valid emotions present in wide_df.")

    # 4) Persistir en DB
    db.create_and_truncate_emotions_dictionary()
    db.insert_emotions_dictionary(cleaned)

    return cleaned


def main(wide_df, set_message=None):
    result = build_and_store_emotions_dictionary_with_llm(
        wide_df,
        llm_model="gpt-4o-mini",
        temperature=0.2,
        dry_run_llm=False,  # True para probar sin consumir API
        verbose=True  # imprime las columnas/RAW si hay errores
    )
    if set_message:
        set_message(f"Inserted {len(result)} emotions into emotions_dictionary")
    print(f"Inserted {len(result)} emotions into emotions_dictionary")

    return result
