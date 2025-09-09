import os
from dotenv import load_dotenv
import json
import pymysql
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from pymysql.cursors import DictCursor

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

def ensure_database_and_tables():
    """Create database and tables if they do not exist."""
    ddl = [
        "CREATE DATABASE IF NOT EXISTS spotify_songs CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;",
        "USE spotify_songs;",
        """
        CREATE TABLE IF NOT EXISTS Artists (
          spotify_id VARCHAR(64) NOT NULL,
          name       VARCHAR(255) NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          PRIMARY KEY (spotify_id),
          KEY idx_artists_name (name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS Albums (
          spotify_id         VARCHAR(64) NOT NULL,
          name               VARCHAR(255) NOT NULL,
          image_url          TEXT,
          image_height       INT NULL,
          image_width        INT NULL,
          artist_spotify_id  VARCHAR(64) NOT NULL,
          created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          PRIMARY KEY (spotify_id),
          KEY idx_albums_name (name),
          KEY idx_albums_artist (artist_spotify_id),
          CONSTRAINT fk_albums_artist
            FOREIGN KEY (artist_spotify_id) REFERENCES Artists(spotify_id)
            ON UPDATE CASCADE ON DELETE RESTRICT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
        """
        CREATE TABLE IF NOT EXISTS Tracks (
          spotify_id         VARCHAR(64) NOT NULL,
          artist_spotify_id  VARCHAR(64) NOT NULL,
          album_spotify_id   VARCHAR(64) NOT NULL,
          name               VARCHAR(255) NOT NULL,
          popularity         INT DEFAULT 0,
          href               VARCHAR(255),
          emotions           JSON NULL,
          lyrics             LONGTEXT NULL,
          created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          PRIMARY KEY (spotify_id),
          KEY idx_tracks_name (name),
          KEY idx_tracks_artist (artist_spotify_id),
          KEY idx_tracks_album (album_spotify_id),
          CONSTRAINT fk_tracks_artist
            FOREIGN KEY (artist_spotify_id) REFERENCES Artists(spotify_id)
            ON UPDATE CASCADE ON DELETE RESTRICT,
          CONSTRAINT fk_tracks_album
            FOREIGN KEY (album_spotify_id) REFERENCES Albums(spotify_id)
            ON UPDATE CASCADE ON DELETE RESTRICT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """,
    ]
    conn = get_conn()
    cur = conn.cursor()
    for stmt in ddl:
        cur.execute(stmt)
    cur.close()
    conn.close()

def upsert_artist(conn, spotify_id: str, name: str):
    sql = """
    INSERT INTO Artists (spotify_id, name)
    VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE name = VALUES(name), updated_at = CURRENT_TIMESTAMP;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (spotify_id, name))

def upsert_album(conn, spotify_id: str, name: str, artist_spotify_id: str,
                 image_url, image_height, image_width):
    sql = """
    INSERT INTO Albums (spotify_id, name, image_url, image_height, image_width, artist_spotify_id)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      name=VALUES(name), image_url=VALUES(image_url),
      image_height=VALUES(image_height), image_width=VALUES(image_width),
      artist_spotify_id=VALUES(artist_spotify_id), updated_at=CURRENT_TIMESTAMP;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (spotify_id, name, image_url, image_height, image_width, artist_spotify_id))

def upsert_track(conn, spotify_id: str, name: str, popularity: int, href: str,
                 artist_spotify_id: str, album_spotify_id: str):
    sql = """
    INSERT INTO Tracks (spotify_id, name, popularity, href, artist_spotify_id, album_spotify_id)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      name=VALUES(name), popularity=VALUES(popularity), href=VALUES(href),
      artist_spotify_id=VALUES(artist_spotify_id), album_spotify_id=VALUES(album_spotify_id),
      updated_at=CURRENT_TIMESTAMP;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (spotify_id, name, popularity, href, artist_spotify_id, album_spotify_id))


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

def update_normalized_keywords(conn, df_norm):
    try:
        with conn.cursor() as cursor:
            sql = """
                UPDATE Tracks
                SET normalized_keywords = %s
                WHERE spotify_id = %s
            """
            data = []
            for _, row in df_norm.iterrows():
                norm_kw = json.dumps(row["normalized_keywords"])
                track_id = row["track_spotify_id"]
                data.append((norm_kw, track_id))

            cursor.executemany(sql, data)
        conn.commit()
        print(f"{len(data)} filas actualizadas correctamente.")
    except Exception as e:
        conn.rollback()
        print("Error al actualizar:", e)

def upsert_emotion_wide(conn, table: str, wide_df: pd.DataFrame):
    """
    Bulk UPSERT wide_df into MySQL table:
    - INSERT ... ON DUPLICATE KEY UPDATE for all emotion columns.
    """
    if wide_df.empty:
        return
    cols = list(wide_df.columns)  # first col must be track_spotify_id
    placeholders = ", ".join(["%s"] * len(cols))
    col_list = ", ".join([f"`{c}`" for c in cols])
    update_clause = ", ".join([f"`{c}`=VALUES(`{c}`)" for c in cols[1:]])  # skip PK

    sql = f"""
        INSERT INTO `{table}` ({col_list})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_clause}
    """.strip()

    with conn.cursor() as cur:
        cur.executemany(sql, [tuple(row) for row in wide_df.itertuples(index=False, name=None)])
    conn.commit()


def truncate_track_emotions_wide(conn):
    try:
        with conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE track_emotions_wide;")
        conn.commit()
    except Exception as e:
        conn.rollback()

def update_track_emotions(conn, df):
    """
    Update the 'emotions' field in tracks table.
    - df must have columns: track_spotify_id, emotions
    - emotions is a list[dict] -> will be stored as JSON string
    - match df.track_spotify_id to tracks.spotify_id
    """
    try:
        with conn.cursor() as cur:
            sql = """
                UPDATE tracks
                SET emotions = %s
                WHERE spotify_id = %s
            """
            data = []
            for _, row in df.iterrows():
                spid = str(row["track_spotify_id"]).strip()
                emos = row["emotions"]

                # Ensure JSON string
                emo_json = json.dumps(emos, ensure_ascii=False) if emos is not None else "[]"

                data.append((emo_json, spid))

            cur.executemany(sql, data)

        conn.commit()
        print(f"Updated emotions for {len(data)} tracks.")

    except Exception as e:
        conn.rollback()
        print("Error while updating emotions:", e)
        raise

def build_create_table_sql(table: str, emotion_cols: list[str]) -> str:
    """
    Builds a CREATE TABLE statement with one DECIMAL(6,3) column per emotion (0–100%).
    Primary key: track_spotify_id
    """
    cols_sql = ",\n  ".join([f"`{c}` DECIMAL(6,4) NOT NULL DEFAULT 0" for c in emotion_cols])
    ddl = f"""
CREATE TABLE IF NOT EXISTS `{table}` (
  `track_spotify_id` VARCHAR(64) NOT NULL,
  {cols_sql},
  PRIMARY KEY (`track_spotify_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
""".strip()
    return ddl


def excecute_DDL(conn, ddl):
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


# ---------- DB ----------

def create_and_truncate_emotions_dictionary():
    """
    Crea la tabla emotions_dictionary si no existe y la limpia (TRUNCATE).
    """
    conn = create_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS emotions_dictionary (
                    emotion   VARCHAR(128) PRIMARY KEY,
                    normalize VARCHAR(20) NOT NULL,
                    emoji     VARCHAR(16) NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            cur.execute("TRUNCATE TABLE emotions_dictionary;")
        conn.commit()
    finally:
        try:
            conn.close()
        except:
            pass


def insert_emotions_dictionary(rows: List[Dict[str, Any]]):
    """
    Inserta filas en emotions_dictionary: [{emotion, normalize, emoji}, ...]
    """
    if not rows:
        return
    conn = create_connection()
    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO emotions_dictionary (emotion, normalize, emoji)
                VALUES (%s, %s, %s)
                """,
                [(r["emotion"], r["normalize"], r["emoji"]) for r in rows]
            )
        conn.commit()
    finally:
        try:
            conn.close()
        except:
            pass

def sync_clusters_and_update_tracks(conn, df_named: pd.DataFrame):
    """
    Safe sync without TRUNCATE (FK-friendly) for SQLite.

    Steps:
    1) UPDATE tracks SET cluster_id = NULL
    2) DELETE FROM clusters
    3) Reset AUTOINCREMENT sequence (if exists)
    4) INSERT unique (name, description) from df_named
    5) SELECT id, name FROM clusters -> name→id map
    6) UPDATE tracks.cluster_id by matching spotify_id (df) to tracks.spotify_id (db)
    """

    required_cols = {"cluster_name", "cluster_desc", "track_spotify_id"}
    missing = required_cols - set(df_named.columns)
    if missing:
        raise ValueError(f"df_named is missing required columns: {missing}")

    clusters_df = (
        df_named[["cluster_name", "cluster_desc"]]
        .dropna(subset=["cluster_name"])
        .drop_duplicates(subset=["cluster_name"])
        .rename(columns={"cluster_name": "name", "cluster_desc": "description"})
    )
    track_map_df = df_named.dropna(subset=["cluster_name"])[["track_spotify_id", "cluster_name"]]

    try:
        with conn:  # transaction; commits on success, rollbacks on exception
            conn.execute("PRAGMA foreign_keys = ON;")

            # 1) Break FK references in child table
            conn.execute("UPDATE tracks SET cluster_id = NULL;")

            # 2) Clear parent table
            conn.execute("DELETE FROM clusters;")

            # 3) Reset AUTOINCREMENT sequence (only exists if AUTOINCREMENT was used)
            try:
                conn.execute("DELETE FROM sqlite_sequence WHERE name = 'clusters';")
            except sqlite3.OperationalError:
                pass

            # 4) Insert new clusters
            if not clusters_df.empty:
                payload = [
                    (
                        str(r["name"]).strip(),
                        str(r["description"]).strip() if pd.notna(r["description"]) else ""
                    )
                    for _, r in clusters_df.iterrows()
                ]
                conn.executemany(
                    "INSERT INTO clusters (name, description) VALUES (?, ?)",
                    payload
                )

            # 5) Build name -> id map
            cur = conn.execute("SELECT id, name FROM clusters;")
            name_to_id = {row[1]: row[0] for row in cur.fetchall()}

            # 6) Update tracks with cluster_id
            data = []
            for _, r in track_map_df.iterrows():
                name = str(r["cluster_name"]).strip()
                track_id = str(r["track_spotify_id"]).strip()
                cid = name_to_id.get(name)
                if cid is not None and track_id:
                    data.append((cid, track_id))

            if data:
                conn.executemany(
                    "UPDATE tracks SET cluster_id = ? WHERE spotify_id = ?",
                    data
                )

        print(f"Inserted clusters: {len(clusters_df)} | Updated tracks: {len(data) if 'data' in locals() else 0}")

    except Exception:
        # 'with conn:' auto-rollbacks on exception
        raise

from pymysql.cursors import DictCursor


def get_emotions_from_dictionary() -> list[str]:
    """
    Step 1: Fetch all emotions from emotions_dictionary.
    """
    conn = create_connection()
    try:
        with conn.cursor(DictCursor) as cur:
            cur.execute("SELECT emotion FROM emotions_dictionary")
            emotions = [r["emotion"] for r in cur.fetchall()]
    finally:
        try:
            conn.close()
        except:
            pass

    if not emotions:
        raise ValueError("No emotions found in emotions_dictionary")

    return emotions


def build_avg_emotions_sql(emotions: list[str]) -> str:
    """
    Step 2: Build a dynamic SQL query with AVG(col) for each emotion.
    Note: we use aliases equal to the original emotion names.
    """
    avg_exprs = [f"AVG(`{emo}`) AS `{emo}`" for emo in emotions]
    sql = f"SELECT {', '.join(avg_exprs)} FROM track_emotions_wide"
    return sql


def execute_avg_emotions_query(sql: str) -> dict:
    """
    Step 3: Execute the generated SQL query and return the results
    with keys = original emotion names.
    """
    conn = create_connection()
    try:
        with conn.cursor(DictCursor) as cur:
            cur.execute(sql)
            row = cur.fetchone()
    finally:
        try:
            conn.close()
        except:
            pass

    return {k: float(v) if v is not None else None for k, v in row.items()}


def fetch_avg_per_emotion() -> dict:
    """
    Orchestrates the 3 steps:
    1. Fetch emotions from the dictionary,
    2. Build the dynamic SQL query,
    3. Execute the query and return the dict with averages.
    """
    emotions = get_emotions_from_dictionary()
    sql = build_avg_emotions_sql(emotions)
    result = execute_avg_emotions_query(sql)
    return result


def fetch_emotion_normalize_mapping() -> Dict[str, str]:
    """
    Step 1: Query emotions_dictionary and return a mapping:
      { emotion_lower: normalize_code }

    Works case-insensitive by lowering all emotion names.
    """
    conn = create_connection()
    try:
        with conn.cursor(DictCursor) as cur:
            cur.execute("SELECT emotion, normalize FROM emotions_dictionary")
            rows = cur.fetchall()
    finally:
        try:
            conn.close()
        except:
            pass

    if not rows:
        raise ValueError("emotions_dictionary is empty or missing.")

    # Build a dict where keys are lowercase emotion names and values are normalize codes
    return {r["emotion"].strip().lower(): str(r["normalize"]).strip() for r in rows}


def group_avg_emotions_by_normalize(avg_per_emotion: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Step 2: Receive a dict with per-emotion averages, e.g.:
      {'admiration': 3.2, 'anger': 1.8, ...}

    Group those averages by their normalize category from emotions_dictionary.

    Returns a list of dicts with:
      - normalize: category (FP/MP/N/MN/FN or Positive/Neutral/Negative)
      - sum_avg: sum of averages of all emotions in this group
      - mean_avg: mean of those averages (sum_avg / emotion_count)
      - emotion_count: how many emotions were grouped
    """
    if not isinstance(avg_per_emotion, dict) or not avg_per_emotion:
        raise ValueError("avg_per_emotion must be a non-empty dict like {'admiration': 3.2, ...}")

    # Get mapping from emotions to normalize codes
    mapping = fetch_emotion_normalize_mapping()  # {emotion_lower: normalize}
    groups: Dict[str, List[float]] = {}

    # Match emotions to normalize groups
    for emo_name, avg_val in avg_per_emotion.items():
        if avg_val is None:
            continue
        try:
            v = float(avg_val)
        except (TypeError, ValueError):
            continue
        norm = mapping.get(str(emo_name).strip().lower())
        if not norm:
            # Emotion not found in dictionary → skip
            continue
        groups.setdefault(norm, []).append(v)

    # Build final result list
    results: List[Dict[str, Any]] = []
    # Preferred order (if you are using 5-level system); fallback: alphabetical
    preferred_order = ["FP", "MP", "N", "MN", "FN", "Positive", "Neutral", "Negative"]
    ordered_norms = [n for n in preferred_order if n in groups] + \
                    [n for n in sorted(groups.keys()) if n not in preferred_order]

    for norm in ordered_norms:
        vals = groups[norm]
        s = sum(vals)
        c = len(vals)
        results.append({
            "normalize": norm,
            "sum_avg": s,
            "emotion_count": c,
        })

    return results

def fetch_emotions_dictionary() -> List[Dict[str, Any]]:
    """
    Fetch all rows from emotions_dictionary.
    Returns a list of dicts with keys: emotion, normalize, emoji.
    """
    sql = "SELECT emotion, emoji FROM emotions_dictionary ORDER BY emotion"

    conn = create_connection()
    try:
        with conn.cursor(DictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    finally:
        try: conn.close()
        except: pass

    return rows


# ---------- Helpers (reuse-safe) ----------

def _get_existing_tew_columns() -> List[str]:
    """
    Return the list of existing columns in track_emotions_wide
    (excluding track_spotify_id), preserving table order.
    """
    conn = create_connection()
    sql = """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'track_emotions_wide'
        ORDER BY ORDINAL_POSITION
    """
    try:
        with conn.cursor(DictCursor) as cur:
            cur.execute(sql)
            cols = [r["COLUMN_NAME"] for r in cur.fetchall()]
    finally:
        try:
            conn.close()
        except:
            pass

    return [c for c in cols if c and c.lower() != "track_spotify_id"]


def get_emotions_from_dictionary() -> List[str]:
    """
    Step 1 for dynamic AVG building: fetch emotion column names from emotions_dictionary.
    """
    conn = create_connection()
    try:
        with conn.cursor(DictCursor) as cur:
            cur.execute("SELECT emotion FROM emotions_dictionary")
            emotions = [r["emotion"] for r in cur.fetchall()]
    finally:
        try:
            conn.close()
        except:
            pass

    if not emotions:
        raise ValueError("No emotions found in emotions_dictionary")
    return emotions


# ---------- 1) All tracks with their clusters, artist name, and album title ----------

def fetch_all_tracks_with_clusters() -> List[Dict[str, Any]]:
    """
    Return all tracks (expected ~990) with:
      - track_spotify_id
      - cluster_id, cluster_name
      - artist_spotify_id, artist_name
      - album_spotify_id, album_name
    """
    sql = """
        SELECT
          t.spotify_id          AS track_spotify_id,
          t.cluster_id          AS cluster_id,
          t.name                AS track_name,
          c.name                AS cluster_name,
          ar.spotify_id         AS artist_spotify_id,
          ar.name               AS artist_name,
          a.spotify_id          AS album_spotify_id,
          a.name                AS album_name
        FROM tracks t
        LEFT JOIN clusters c ON c.id = t.cluster_id
        LEFT JOIN artists  ar ON ar.spotify_id = t.artist_spotify_id
        LEFT JOIN albums   a  ON a.spotify_id  = t.album_spotify_id
        ORDER BY ar.name, a.name, t.spotify_id
    """
    conn = create_connection()
    try:
        with conn.cursor(DictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    finally:
        try:
            conn.close()
        except:
            pass

    return rows

def fetch_clusters() -> List[Dict[str, str]]:
    """
    Return all clusters as a list of dicts with id, name, description.
    """
    conn = create_connection()
    try:
        cur = conn.cursor(dictionary=True)  # Dict-style rows
        cur.execute("SELECT id, name, description FROM clusters ORDER BY id")
        return cur.fetchall()
    finally:
        conn.close()


# ---------- 2) All tracks for a given cluster (with artist and album names) ----------

def fetch_tracks_by_cluster(cluster_id: int) -> List[Dict[str, Any]]:
    """
    Return tracks for a specific cluster with the same fields as above.
    """
    sql = """
        SELECT
          t.spotify_id          AS track_spotify_id,
          t.cluster_id          AS cluster_id,
          t.name                AS track_name,
          c.name                AS cluster_name,
          ar.spotify_id         AS artist_spotify_id,
          ar.name               AS artist_name,
          a.spotify_id          AS album_spotify_id,
          a.name                AS album_name
        FROM tracks t
        JOIN clusters c  ON c.id = t.cluster_id
        LEFT JOIN artists ar ON ar.spotify_id = t.artist_spotify_id
        LEFT JOIN albums  a  ON a.spotify_id  = t.album_spotify_id
        WHERE t.cluster_id = %s
        ORDER BY ar.name, a.name, t.spotify_id
    """
    conn = create_connection()
    try:
        with conn.cursor(DictCursor) as cur:
            cur.execute(sql, (cluster_id,))
            rows = cur.fetchall()
    finally:
        try:
            conn.close()
        except:
            pass

    return rows


# ---------- 4) Per-cluster averages for every emotion column ----------

def build_cluster_avg_emotions_sql(emotions: List[str]) -> str:
    """
    Build a dynamic SQL SELECT that computes AVG(col) AS `col` for all given emotions,
    restricted to a cluster via WHERE t.cluster_id = %s.
    The SELECT joins track_emotions_wide with tracks by spotify_id.
    """
    # Intersect requested emotions with actual columns in track_emotions_wide
    existing = set(_get_existing_tew_columns())
    valid_emotions = [e for e in emotions if e in existing]
    if not valid_emotions:
        raise ValueError("None of the emotions exist in track_emotions_wide")

    avg_exprs = [f"AVG(tew.`{emo}`) AS `{emo}`" for emo in valid_emotions]
    sql = f"""
        SELECT {', '.join(avg_exprs)}
        FROM track_emotions_wide AS tew
        JOIN tracks AS t ON t.spotify_id = tew.track_spotify_id
        WHERE t.cluster_id = %s
    """
    return sql


def fetch_cluster_avg_emotions(cluster_id: int) -> Dict[str, Optional[float]]:
    """
    Compute emotion-wise averages for a given cluster.
    - Uses emotions from emotions_dictionary
    - Validates columns against track_emotions_wide
    - Returns a dict: { 'admiration': 3.2, 'anger': 1.8, ... }
    """
    emotions = get_emotions_from_dictionary()
    sql = build_cluster_avg_emotions_sql(emotions)

    conn = create_connection()
    try:
        with conn.cursor(DictCursor) as cur:
            cur.execute(sql, (cluster_id,))
            row = cur.fetchone()  # one row with AVG per emotion
    finally:
        try:
            conn.close()
        except:
            pass

    # Convert Decimal/None to float/None
    return {k: (float(v) if v is not None else None) for k, v in row.items()}


# ---------- Optional convenience: also return track_count for the cluster ----------

def fetch_cluster_avg_emotions_with_count(cluster_id: int) -> Dict[str, Any]:
    """
    Same as fetch_cluster_avg_emotions, but also returns 'track_count' for the cluster.
    """
    # Averages
    emo_avgs = fetch_cluster_avg_emotions(cluster_id)

    # Track count
    conn = create_connection()
    try:
        with conn.cursor(DictCursor) as cur:
            cur.execute("SELECT COUNT(*) AS n FROM tracks WHERE cluster_id = %s", (cluster_id,))
            n = cur.fetchone()["n"]
    finally:
        try:
            conn.close()
        except:
            pass

    out: Dict[str, Any] = {"cluster_id": cluster_id, "track_count": int(n)}
    out.update(emo_avgs)
    return out


def fetch_all_artists() -> List[Dict[str, Any]]:
    """
    Return all artists with their spotify_id and name.
    """
    sql = "SELECT spotify_id, name FROM artists ORDER BY name"
    conn = create_connection()
    try:
        with conn.cursor(DictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    finally:
        try:
            conn.close()
        except:
            pass
    return rows


def fetch_artist_by_id(artist_spotify_id: str) -> Optional[Dict[str, Any]]:
    """
    Return a single artist by its spotify_id.
    """
    sql = "SELECT spotify_id, name FROM artists WHERE spotify_id = %s"
    conn = create_connection()
    try:
        with conn.cursor(DictCursor) as cur:
            cur.execute(sql, (artist_spotify_id,))
            row = cur.fetchone()
    finally:
        try:
            conn.close()
        except:
            pass
    return row


# ------------------- Albums -------------------

def fetch_all_albums() -> List[Dict[str, Any]]:
    """
    Return all albums with their spotify_id and name.
    """
    sql = "SELECT spotify_id, name FROM albums ORDER BY name"
    conn = create_connection()
    try:
        with conn.cursor(DictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    finally:
        try:
            conn.close()
        except:
            pass
    return rows


def fetch_album_by_id(album_spotify_id: str) -> Optional[Dict[str, Any]]:
    """
    Return a single album by its spotify_id.
    """
    sql = "SELECT spotify_id, name FROM albums WHERE spotify_id = %s"
    conn = create_connection()
    try:
        with conn.cursor(DictCursor) as cur:
            cur.execute(sql, (album_spotify_id,))
            row = cur.fetchone()
    finally:
        try:
            conn.close()
        except:
            pass
    return row

# mysql_queries.py
# MySQL version of the dynamic bucket query against track_emotions_wide.
# Comments in English only.



BUCKET_MAP = {
    "FP": "full_positive",
    "MP": "positive",
    "N":  "neutral",
    "MN": "negative",
    "FN": "full_negative",
}

ALLOWED_SORT_KEYS = ["track", "artist", "album"] + list(BUCKET_MAP.values())

def _get_bucket_columns(conn) -> Dict[str, List[str]]:
    """
    Discover which columns from track_emotions_wide belong to each bucket code
    using emotions_dictionary (emotion -> normalize[FP/MP/N/MN/FN]).
    Only keep emotions that exist as columns in track_emotions_wide.
    Returns: { 'FP': ['admiration', ...], 'MP': [...], ... }
    """
    cur = conn.cursor()
    # 1) dictionary rows
    cur.execute("SELECT emotion, normalize FROM emotions_dictionary")
    dic = cur.fetchall()

    # 2) existing wide columns from information_schema
    cur.execute(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'track_emotions_wide'
        """
    )
    cols = {r[0] if not isinstance(r, dict) else r["COLUMN_NAME"] for r in cur.fetchall()}
    if "track_spotify_id" in cols:
        cols.remove("track_spotify_id")

    mapping: Dict[str, List[str]] = {"FP": [], "MP": [], "N": [], "MN": [], "FN": []}
    for row in dic:
        emo = (row[0] if not isinstance(row, dict) else row["emotion"]).strip()
        code = (row[1] if not isinstance(row, dict) else row["normalize"]).strip().upper()
        if emo in cols and code in mapping:
            mapping[code].append(emo)

    cur.close()
    return mapping

def _build_bucket_sum_exprs(bucket_cols: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Build SUM expressions per bucket using COALESCE for missing values.
    Returns dict with SQL snippets keyed by pretty bucket names.
    """
    out: Dict[str, str] = {}
    for code, pretty in BUCKET_MAP.items():
        cols = bucket_cols.get(code, [])
        if not cols:
            out[pretty] = "0.0"
            continue
        pieces = [f"COALESCE(tew.`{c}`, 0.0)" for c in cols]
        out[pretty] = "(" + " + ".join(pieces) + ")"
    return out

def fetch_tracks_with_buckets_paginated(
    conn,
    page: int = 1,
    page_size: int = 50,
    track: Optional[str] = None,
    artist: Optional[str] = None,
    album: Optional[str] = None,
    sort_by: str = "track",      # track | artist | album | full_positive | positive | neutral | negative | full_negative
    sort_dir: str = "asc",       # asc | desc
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Return paginated rows with: track, artist, album and 5 bucket sums.
    Uses emotions_dictionary to decide which wide columns belong to each bucket.
    Pagination and ordering happen in SQL (MySQL).
    """
    if page < 1 or page_size < 1:
        raise ValueError("page and page_size must be >= 1")

    sort_by = sort_by if sort_by in ALLOWED_SORT_KEYS else "track"
    sort_dir_sql = "DESC" if str(sort_dir).lower() == "desc" else "ASC"

    # Filters (case-insensitive via LOWER)
    wheres: List[str] = []
    params: List[Any] = []
    if track:
        wheres.append("LOWER(t.`name`) LIKE LOWER(%s)")
        params.append(f"%{track}%")
    if artist:
        wheres.append("LOWER(a.`name`) LIKE LOWER(%s)")
        params.append(f"%{artist}%")
    if album:
        wheres.append("LOWER(al.`name`) LIKE LOWER(%s)")
        params.append(f"%{album}%")
    where_sql = f"WHERE {' AND '.join(wheres)}" if wheres else ""

    cur = conn.cursor()

    # Total rows (filtered)
    total_sql = f"""
        SELECT COUNT(1)
        FROM `Tracks` t
        LEFT JOIN `Artists` a ON a.`spotify_id` = t.`artist_spotify_id`
        LEFT JOIN `Albums`  al ON al.`spotify_id` = t.`album_spotify_id`
        {where_sql}
    """
    cur.execute(total_sql, params)
    total = int(cur.fetchone()[0])

    # Bucket expressions built dynamically
    bucket_cols = _get_bucket_columns(conn)
    bucket_exprs = _build_bucket_sum_exprs(bucket_cols)
    select_buckets = ",\n          ".join([f"{expr} AS `{name}`" for name, expr in bucket_exprs.items()])

    # Safe ORDER BY (whitelisted)
    if sort_by in ["track", "artist", "album"]:
        order_sql = f"ORDER BY `{sort_by}` {sort_dir_sql}, `track` ASC, `artist` ASC, `album` ASC"
    else:
        order_sql = f"ORDER BY `{sort_by}` {sort_dir_sql}, `track` ASC, `artist` ASC, `album` ASC"

    # Paged query
    page_sql = f"""
        SELECT
          t.`name`                     AS `track`,
          COALESCE(a.`name`, '')       AS `artist`,
          COALESCE(al.`name`, '')      AS `album`,
          {select_buckets},
          t.`emotions`
        FROM `Tracks` t
        LEFT JOIN `Artists` a ON a.`spotify_id` = t.`artist_spotify_id`
        LEFT JOIN `Albums`  al ON al.`spotify_id` = t.`album_spotify_id`
        LEFT JOIN `track_emotions_wide` tew ON tew.`track_spotify_id` = t.`spotify_id`
        {where_sql}
        {order_sql}
        LIMIT %s OFFSET %s
    """
    cur.execute(page_sql, params + [page_size, (page - 1) * page_size])

    # Return rows as dicts regardless of cursor type
    desc = [d[0] for d in cur.description]
    fetched = cur.fetchall()
    if fetched and isinstance(fetched[0], dict):
        rows = fetched  # DictCursor already
    else:
        rows = [dict(zip(desc, r)) for r in fetched]

    cur.close()
    return rows, total



