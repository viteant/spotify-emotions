import os
from dotenv import load_dotenv
from typing import Optional
import json
import pymysql
from typing import List, Dict

def create_connection():
    """
    Crea una conexión a MySQL usando variables de entorno.
    Requiere archivo .env con:
      DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT
    """
    load_dotenv()  # carga el .env

    conn = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),  # default 3306
        cursorclass=pymysql.cursors.DictCursor
    )
    return conn

def get_lyrics(
        conn,
        track_id: Optional[int] = None
):
    """
    Obtiene los lyrics de la tabla 'tracks' en la BD 'spotify_songs'.
    - Si pasas track_id devuelve solo ese.
    - Si no, devuelve todos los lyrics no nulos.
    """
    with conn.cursor() as cursor:
        if track_id:
            sql = "SELECT spotify_id, lyrics FROM tracks WHERE id = %s"
            cursor.execute(sql, (track_id,))
        else:
            sql = "SELECT spotify_id, lyrics FROM tracks WHERE lyrics IS NOT NULL AND lyrics <> ''"
            cursor.execute(sql)

        rows = cursor.fetchall()
        return {row["spotify_id"]: row["lyrics"] for row in rows}

def save_keywords(conn, keywords_dict: dict):
    """
    keywords_dict: {spotify_id: [kw1, kw2, ...]}
    """
    with conn.cursor() as cursor:
        sql = "UPDATE tracks SET keywords = %s WHERE spotify_id = %s"
        for sid, kws in keywords_dict.items():
            if not isinstance(kws, (list, tuple)):
                raise TypeError(f"keywords debe ser lista, recibido: {type(kws)}")
            cursor.execute(sql, (json.dumps(list(kws)), sid))
            if cursor.rowcount == 0:
                print(f"[WARN] No se encontró spotify_id={sid}; no se actualizó nada.")
    conn.commit()

def get_tracks_missing_lyrics(conn, limit: int = 500, offset: int = 0):
    """
    Devuelve tracks SIN lyrics como lista de dicts:
    {"track_spotify_id": str, "title": str, "artist_name": str}
    """
    sql = """
        SELECT
            t.spotify_id AS track_spotify_id,
            t.name       AS title,
            COALESCE(a.name, '') AS artist_name
        FROM tracks t
        LEFT JOIN artists a
               ON a.spotify_id = t.artist_spotify_id
        WHERE (t.lyrics IS NULL OR TRIM(t.lyrics) = '') AND t.keywords IS NULL
        ORDER BY t.spotify_id
        LIMIT %s OFFSET %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (limit, offset))
        rows = cur.fetchall()

    # Si ya son dicts (DictCursor), regrésalos tal cual
    if rows and isinstance(rows[0], dict):
        return rows

    # Si son tuplas, mapéalas a dicts
    return [
        {"track_spotify_id": r[0], "title": r[1], "artist_name": r[2]}
        for r in rows
    ]



def fetch_tracks_dataset(conn, limit: int | None = None, offset: int = 0) -> list[dict]:
    """
    Devuelve una lista de diccionarios con:
      - track_spotify_id
      - title
      - artist_name
      - keywords (lista de strings parseada desde JSON)
    """
    sql = """
        SELECT
            t.spotify_id   AS track_spotify_id,
            t.name         AS title,
            COALESCE(a.name, '') AS artist_name,
            t.keywords     AS keywords
        FROM tracks t
        LEFT JOIN artists a
               ON a.spotify_id = t.artist_spotify_id
        WHERE t.keywords IS NOT NULL AND TRIM(t.keywords) <> ''
        ORDER BY t.spotify_id
        {limit_clause}
    """
    limit_clause = ""
    params = ()
    if limit is not None:
        limit_clause = "LIMIT %s OFFSET %s"
        params = (limit, offset)
    sql = sql.format(limit_clause=limit_clause)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    # Parsear JSON de keywords
    def _parse_kws(val):
        if not val:
            return []
        if isinstance(val, list):
            return val
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    for r in rows:
        r["keywords"] = _parse_kws(r.get("keywords"))

    return rows



def save_track_emotions(conn, track_spotify_id: str, emotions: List[Dict], perc_from_score: bool = True):
    """
    Save emotions for a track in track_emotions.
    emotions: [{"label": str, "score": float}, ...] (score in 0–1)
    If perc_from_score=True, store percentage (0–100 with 3 decimals).
    Requires tables: emotions(id, label) and track_emotions(track_spotify_id, emotion_id, percentage).
    """
    if not track_spotify_id or not emotions:
        return

    labels = [str(e.get("label", "")).strip().lower() for e in emotions if e.get("label")]
    labels = [l for l in labels if l]
    if not labels:
        return
    unique_labels = sorted(set(labels))

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT IGNORE INTO emotions(label) VALUES (%s)",
            [(lab,) for lab in unique_labels]
        )
        conn.commit()

        # Construye SELECT ... IN (%s, %s, ...)
        placeholders = ",".join(["%s"] * len(unique_labels))
        sql = f"SELECT id, label FROM emotions WHERE label IN ({placeholders})"
        cur.execute(sql, unique_labels)

        # Soporta DictCursor o cursor por tuplas
        rows = cur.fetchall()
        label_to_id = {}
        for r in rows:
            if isinstance(r, dict):
                label_to_id[r["label"]] = r["id"]
            else:
                # r[0]=id, r[1]=label
                label_to_id[r[1]] = r[0]

        # Prepara filas a insertar
        ins_rows = []
        for e in emotions:
            lab = str(e.get("label", "")).strip().lower()
            if not lab or lab not in label_to_id:
                continue
            score = float(e.get("score", 0.0))
            value = round(score * 100.0, 3) if perc_from_score else score
            ins_rows.append((track_spotify_id, label_to_id[lab], value))

        # Reemplaza completamente las emociones del track
        cur.execute("DELETE FROM track_emotions WHERE track_spotify_id = %s", (track_spotify_id,))
        if ins_rows:
            cur.executemany(
                "INSERT INTO track_emotions(track_spotify_id, emotion_id, percentage) VALUES (%s, %s, %s)",
                ins_rows
            )
        conn.commit()




