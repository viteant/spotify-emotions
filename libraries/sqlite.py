import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple
import json
import sqlite3
from pathlib import Path
import pandas as pd

# -----------------------------
# SQLite connection
# -----------------------------
def create_connection():
    """
    Create a connection to SQLite using environment variables.
    Requires in .env (optional):
      SQLITE_DB_PATH=<path_to_db_file>
    If not set, defaults to app.db in the current directory.
    """
    load_dotenv()
    db_path = os.getenv("SQLITE_DB_PATH")
    if not db_path:
        db_path = Path(__file__).resolve().parent / "../app.db"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # access rows as dict-like objects
    conn.execute("PRAGMA foreign_keys = ON;")  # enable foreign keys
    return conn


# -----------------------------
# Tables Spotify
# -----------------------------
def ensure_database_and_tables(conn: sqlite3.Connection):
    """Create tables if they do not exist (SQLite compatible schema)."""
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS Artists (
          spotify_id TEXT PRIMARY KEY,
          name       TEXT NOT NULL,
          created_at TEXT DEFAULT (datetime('now')),
          updated_at TEXT DEFAULT (datetime('now'))
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS Albums (
          spotify_id         TEXT PRIMARY KEY,
          name               TEXT NOT NULL,
          image_url          TEXT,
          image_height       INTEGER,
          image_width        INTEGER,
          artist_spotify_id  TEXT NOT NULL,
          created_at         TEXT DEFAULT (datetime('now')),
          updated_at         TEXT DEFAULT (datetime('now')),
          FOREIGN KEY (artist_spotify_id) REFERENCES Artists(spotify_id) ON UPDATE CASCADE ON DELETE RESTRICT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS Tracks (
          spotify_id         TEXT PRIMARY KEY,
          artist_spotify_id  TEXT NOT NULL,
          album_spotify_id   TEXT NOT NULL,
          name               TEXT NOT NULL,
          popularity         INTEGER DEFAULT 0,
          href               TEXT,
          emotions           TEXT,
          lyrics             TEXT,
          keywords           TEXT,
          created_at         TEXT DEFAULT (datetime('now')),
          updated_at         TEXT DEFAULT (datetime('now')),
          FOREIGN KEY (artist_spotify_id) REFERENCES Artists(spotify_id) ON UPDATE CASCADE ON DELETE RESTRICT,
          FOREIGN KEY (album_spotify_id)  REFERENCES Albums(spotify_id)  ON UPDATE CASCADE ON DELETE RESTRICT
        );
        """,
        # Trigger to simulate ON UPDATE CURRENT_TIMESTAMP for Tracks.updated_at
        """
        CREATE TRIGGER IF NOT EXISTS trg_tracks_updated_at
        AFTER UPDATE ON Tracks
        FOR EACH ROW
        BEGIN
          UPDATE Tracks SET updated_at = datetime('now') WHERE spotify_id = NEW.spotify_id;
        END;
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_albums_updated_at
        AFTER UPDATE ON Albums
        FOR EACH ROW
        BEGIN
          UPDATE Albums SET updated_at = datetime('now') WHERE spotify_id = NEW.spotify_id;
        END;
        """,
        """
        CREATE TRIGGER IF NOT EXISTS trg_artists_updated_at
        AFTER UPDATE ON Artists
        FOR EACH ROW
        BEGIN
          UPDATE Artists SET updated_at = datetime('now') WHERE spotify_id = NEW.spotify_id;
        END;
        """
    ]
    cur = conn.cursor()
    for stmt in ddl:
        cur.execute(stmt)
    cur.close()
    conn.commit()

# -----------------------------
# Set Artist Albums and Track
# -----------------------------
def upsert_artist(conn: sqlite3.Connection, spotify_id: str, name: str):
    sql = """
    INSERT INTO Artists(spotify_id, name)
    VALUES (?, ?)
    ON CONFLICT(spotify_id) DO UPDATE SET
      name=excluded.name,
      updated_at=datetime('now');
    """
    conn.execute(sql, (spotify_id, name))
    conn.commit()

def upsert_album(conn: sqlite3.Connection, spotify_id: str, name: str, artist_spotify_id: str,
                 image_url, image_height, image_width):
    sql = """
    INSERT INTO Albums(spotify_id, name, image_url, image_height, image_width, artist_spotify_id)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(spotify_id) DO UPDATE SET
      name=excluded.name,
      image_url=excluded.image_url,
      image_height=excluded.image_height,
      image_width=excluded.image_width,
      artist_spotify_id=excluded.artist_spotify_id,
      updated_at=datetime('now');
    """
    conn.execute(sql, (spotify_id, name, image_url, image_height, image_width, artist_spotify_id))
    conn.commit()

def upsert_track(conn: sqlite3.Connection, spotify_id: str, name: str, popularity: int, href: str,
                 artist_spotify_id: str, album_spotify_id: str):
    sql = """
    INSERT INTO Tracks(spotify_id, name, popularity, href, artist_spotify_id, album_spotify_id)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(spotify_id) DO UPDATE SET
      name=excluded.name,
      popularity=excluded.popularity,
      href=excluded.href,
      artist_spotify_id=excluded.artist_spotify_id,
      album_spotify_id=excluded.album_spotify_id,
      updated_at=datetime('now');
    """
    conn.execute(sql, (spotify_id, name, popularity, href, artist_spotify_id, album_spotify_id))
    conn.commit()

# -----------------------------
# Read lyrics
# -----------------------------
def get_lyrics(conn, track_id: Optional[int] = None):
    """
    Get lyrics from the 'tracks' table.
    - If track_id is provided, return only that record.
    - Otherwise, return all non-null lyrics.
    """
    with conn:
        cur = conn.cursor()
        if track_id:
            sql = "SELECT spotify_id, lyrics FROM tracks WHERE id = ?"
            cur.execute(sql, (track_id,))
        else:
            sql = "SELECT spotify_id, lyrics FROM tracks WHERE lyrics IS NOT NULL AND TRIM(lyrics) <> ''"
            cur.execute(sql)

        rows = cur.fetchall()
        return {row["spotify_id"]: row["lyrics"] for row in rows}

# -----------------------------
# Save keywords (stored as JSON in TEXT)
# -----------------------------
def save_keywords(conn, keywords_dict: dict):
    """
    keywords_dict: {spotify_id: [kw1, kw2, ...]}
    """
    with conn:
        cur = conn.cursor()
        sql = "UPDATE tracks SET keywords = ? WHERE spotify_id = ?"
        for sid, kws in keywords_dict.items():
            if not isinstance(kws, (list, tuple)):
                raise TypeError(f"keywords must be list, received: {type(kws)}")
            cur.execute(sql, (json.dumps(list(kws), ensure_ascii=False), sid))
            if cur.rowcount == 0:
                print(f"[WARN] spotify_id={sid} not found; no update performed.")

# -----------------------------
# Tracks without lyrics
# -----------------------------
def get_tracks_missing_lyrics(conn, limit: int = 500, offset: int = 0):
    """
    Return tracks without lyrics as a list of dicts:
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
        WHERE (t.lyrics IS NULL OR TRIM(t.lyrics) = '') AND (t.keywords IS NULL OR TRIM(t.keywords) = '')
        ORDER BY t.spotify_id
        LIMIT ? OFFSET ?
    """
    with conn:
        cur = conn.cursor()
        cur.execute(sql, (limit, offset))
        rows = cur.fetchall()

    return [
        {"track_spotify_id": r["track_spotify_id"], "title": r["title"], "artist_name": r["artist_name"]}
        for r in rows
    ]

# -----------------------------
# Tracks dataset with keywords
# -----------------------------
def fetch_tracks_dataset(conn, limit: int | None = None, offset: int = 0) -> List[Dict]:
    """
    Return a list of dicts with:
      - track_spotify_id
      - title
      - artist_name
      - keywords (parsed from JSON string to list)
    """
    base_sql = """
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
    """

    params: tuple = ()
    if limit is not None:
        base_sql += " LIMIT ? OFFSET ?"
        params = (limit, offset)

    with conn:
        cur = conn.cursor()
        cur.execute(base_sql, params)
        rows = [dict(r) for r in cur.fetchall()]

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

# -----------------------------
# Save track emotions
# -----------------------------
def save_track_emotions(conn, track_spotify_id: str, emotions: List[Dict], perc_from_score: bool = True):
    """
    Save emotions for a track into track_emotions table.
    emotions: [{"label": str, "score": float}, ...] (score in 0–1 range)
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

    with conn:
        cur = conn.cursor()

        # INSERT OR IGNORE is the SQLite equivalent of MySQL's INSERT IGNORE
        cur.executemany(
            "INSERT OR IGNORE INTO emotions(label) VALUES (?)",
            [(lab,) for lab in unique_labels]
        )

        # Fetch ids for the given labels
        placeholders = ",".join(["?"] * len(unique_labels))
        sql = f"SELECT id, label FROM emotions WHERE label IN ({placeholders})"
        cur.execute(sql, unique_labels)

        rows = cur.fetchall()
        label_to_id = {r["label"]: r["id"] for r in rows}

        # Prepare rows to insert
        ins_rows = []
        for e in emotions:
            lab = str(e.get("label", "")).strip().lower()
            if not lab or lab not in label_to_id:
                continue
            score = float(e.get("score", 0.0))
            value = round(score * 100.0, 3) if perc_from_score else score
            ins_rows.append((track_spotify_id, label_to_id[lab], value))

        # Replace existing emotions for this track
        cur.execute("DELETE FROM track_emotions WHERE track_spotify_id = ?", (track_spotify_id,))
        if ins_rows:
            cur.executemany(
                "INSERT INTO track_emotions(track_spotify_id, emotion_id, percentage) VALUES (?, ?, ?)",
                ins_rows
            )

def ensure_normalized_keywords_column(conn):
    """
    Ensure the 'normalized_keywords' column exists in the Tracks table.
    If it does not exist, create it as TEXT (to hold JSON strings).
    """
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(Tracks)")
    cols = [row[1] for row in cur.fetchall()]  # row[1] is column name
    if "normalized_keywords" not in cols:
        cur.execute("ALTER TABLE Tracks ADD COLUMN normalized_keywords TEXT")
        conn.commit()
        print("Column 'normalized_keywords' created.")
    else:
        print("Column 'normalized_keywords' already exists.")
    cur.close()

def update_normalized_keywords(conn, df_norm):
    # df_norm must have columns: track_spotify_id, normalized_keywords
    sql = """
        UPDATE Tracks
        SET normalized_keywords = ?
        WHERE spotify_id = ?
    """
    data = [
        (json.dumps(row["normalized_keywords"], ensure_ascii=False), row["track_spotify_id"])
        for _, row in df_norm.iterrows()
    ]

    cur = conn.cursor()
    try:
        cur.executemany(sql, data)
        conn.commit()
        print(f"{cur.rowcount} rows updated.")
    except Exception as e:
        conn.rollback()
        print("Update failed:", e)
    finally:
        cur.close()

def upsert_emotion_wide(conn: sqlite3.Connection, table: str, wide_df: pd.DataFrame, pk="track_spotify_id"):
    """
    Bulk UPSERT wide_df into SQLite table using ON CONFLICT DO UPDATE.
    - pk can be a string (single PK) or a list/tuple (composite key).
    - Placeholders are '?' (SQLite).
    """
    if wide_df.empty:
        return

    cols = list(wide_df.columns)
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join([f'"{c}"' for c in cols])

    # Build conflict target and update set
    if isinstance(pk, (list, tuple)):
        conflict_target = ", ".join([f'"{c}"' for c in pk])
        update_cols = [c for c in cols if c not in pk]
    else:
        conflict_target = f'"{pk}"'
        update_cols = [c for c in cols if c != pk]

    update_clause = ", ".join([f'"{c}"=excluded."{c}"' for c in update_cols])

    sql = f'''
        INSERT INTO "{table}" ({col_list})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_target})
        DO UPDATE SET {update_clause};
    '''.strip()

    try:
        with conn:  # transaction context
            conn.executemany(sql, [tuple(row) for row in wide_df.itertuples(index=False, name=None)])
    except Exception as e:
        # If this crashes, your key likely isn't UNIQUE/PK or column names mismatch
        raise


def truncate_track_emotions_wide(conn: sqlite3.Connection, table: str = "track_emotions_wide", reset_autoincrement: bool = True):
    """
    Emulates TRUNCATE for SQLite:
    - DELETE all rows.
    - Optionally reset AUTOINCREMENT by clearing sqlite_sequence (if table used it).
    """
    try:
        with conn:
            conn.execute(f'DELETE FROM "{table}";')
            if reset_autoincrement:
                # This table exists only if AUTOINCREMENT was used; ignore if not present
                try:
                    conn.execute('DELETE FROM sqlite_sequence WHERE name = ?;', (table,))
                except sqlite3.OperationalError:
                    pass
    except Exception:
        conn.rollback()
        raise


def update_track_emotions(conn: sqlite3.Connection, df: pd.DataFrame):
    """
    Update the 'emotions' TEXT (JSON) field in tracks table on SQLite.
    - df must have columns: track_spotify_id, emotions
    - emotions is list[dict] or any JSON-serializable object -> stored as JSON string
    - matches df.track_spotify_id to tracks.spotify_id
    """
    if df.empty:
        print("No rows to update.")
        return

    # Build payload once; JSON-serialize here to keep executemany lean
    payload = []
    for _, row in df.iterrows():
        spid = str(row["track_spotify_id"]).strip()
        emos = row["emotions"]
        emo_json = json.dumps(emos, ensure_ascii=False) if emos is not None else "[]"
        payload.append((emo_json, spid))

    sql = """
        UPDATE tracks
        SET emotions = ?
        WHERE spotify_id = ?
    """

    try:
        with conn:  # transaction
            conn.executemany(sql, payload)
        print(f"Updated emotions for {len(payload)} tracks.")
    except Exception as e:
        # 'with conn:' will rollback on exception automatically
        print("Error while updating emotions:", e)
        raise
def build_create_table_sql(table: str, emotion_cols: list[str]) -> str:
    """
    Builds a CREATE TABLE statement for SQLite.
    - track_spotify_id is the PRIMARY KEY (TEXT)
    - one REAL column per emotion (default 0)
    """
    cols_sql = ",\n  ".join([f'"{c}" REAL NOT NULL DEFAULT 0' for c in emotion_cols])
    ddl = f"""
CREATE TABLE IF NOT EXISTS "{table}" (
  "track_spotify_id" TEXT NOT NULL,
  {cols_sql},
  PRIMARY KEY ("track_spotify_id")
);
""".strip()
    return ddl

def execute_DDL(conn: sqlite3.Connection, ddl: str):
    """
    Ejecuta un DDL (CREATE, DROP, ALTER...) en SQLite usando with conn.
    - Usa conn como contexto para que maneje commit/rollback automáticamente.
    """
    try:
        with conn:  # abre transacción
            conn.execute(ddl)
        print("DDL ejecutado con éxito.")
    except Exception as e:
        print("Error al ejecutar DDL:", e)
        raise

def create_and_truncate_emotions_dictionary():
    """
    Crea la tabla emotions_dictionary si no existe y la limpia (DELETE).
    """
    conn = create_connection()
    try:
        with conn:  # maneja commit/rollback automáticamente
            conn.execute("""
                CREATE TABLE IF NOT EXISTS emotions_dictionary (
                    emotion   TEXT PRIMARY KEY,
                    normalize TEXT NOT NULL,
                    emoji     TEXT NOT NULL
                );
            """)
            conn.execute("DELETE FROM emotions_dictionary;")
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
        with conn:  # transacción
            conn.executemany(
                """
                INSERT INTO emotions_dictionary (emotion, normalize, emoji)
                VALUES (?, ?, ?)
                """,
                [(r["emotion"], r["normalize"], r["emoji"]) for r in rows]
            )
    finally:
        try:
            conn.close()
        except:
            pass


def sync_clusters_and_update_tracks(conn, df_named: pd.DataFrame):
    """
    SQLite-safe sync:
    - Ensure clusters table and tracks.cluster_id column exist
    - Clear clusters and reset sequence
    - Insert unique (name, description) from df_named (UPSERT to avoid rollback on dups)
    - Map clusters to tracks by spotify_id
    Returns (inserted_clusters, updated_tracks)
    """

    # Validate input columns
    required_cols = {"cluster_name", "cluster_desc", "track_spotify_id"}
    missing = required_cols - set(df_named.columns)
    if missing:
        raise ValueError(f"df_named is missing required columns: {missing}")

    # Prepare cluster rows: strip, drop empties, deduplicate by name
    clusters_df = (
        df_named[["cluster_name", "cluster_desc"]]
        .assign(
            cluster_name=lambda d: d["cluster_name"].astype(str).str.strip(),
            cluster_desc=lambda d: d["cluster_desc"].fillna("").astype(str).str.strip()
        )
    )
    clusters_df = clusters_df[clusters_df["cluster_name"] != ""]
    clusters_df = clusters_df.drop_duplicates(subset=["cluster_name"])
    clusters_df = clusters_df.rename(columns={"cluster_name": "name", "cluster_desc": "description"})

    # Prepare track map (only rows with a non-empty cluster name)
    track_map_df = (
        df_named.assign(cluster_name=lambda d: d["cluster_name"].astype(str).str.strip())
               .query("cluster_name != ''")[["track_spotify_id", "cluster_name"]]
    )

    inserted = 0
    updated = 0

    try:
        with conn:  # atomic transaction
            conn.execute("PRAGMA foreign_keys = ON;")

            # 1) Ensure clusters table exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL DEFAULT ''
                );
            """)

            # 2) Ensure tracks.cluster_id exists (FK to clusters.id)
            cols = {row[1] for row in conn.execute("PRAGMA table_info('tracks');").fetchall()}
            if "cluster_id" not in cols:
                conn.execute("""
                    ALTER TABLE tracks
                    ADD COLUMN cluster_id INTEGER
                        REFERENCES clusters(id)
                        ON UPDATE CASCADE
                        ON DELETE SET NULL
                """)

            # 3) Break FK references in child table
            conn.execute("UPDATE tracks SET cluster_id = NULL;")

            # 4) Clear parent table
            conn.execute("DELETE FROM clusters;")

            # 5) Reset AUTOINCREMENT sequence (only if AUTOINCREMENT is used)
            try:
                conn.execute("DELETE FROM sqlite_sequence WHERE name = 'clusters';")
            except sqlite3.OperationalError:
                pass

            # 6) Insert clusters; use UPSERT to avoid rollback on accidental duplicates
            if not clusters_df.empty:
                payload = [
                    (str(r["name"]), str(r["description"]) if pd.notna(r["description"]) else "")
                    for _, r in clusters_df.iterrows()
                ]
                conn.executemany(
                    """
                    INSERT INTO clusters (name, description)
                    VALUES (?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        description=excluded.description
                    """,
                    payload
                )
                # Count rows after insert
                inserted = conn.execute("SELECT COUNT(*) FROM clusters;").fetchone()[0]
            else:
                inserted = 0  # explicit for clarity

            # 7) Build name -> id map
            name_to_id = {name: cid for cid, name in conn.execute("SELECT id, name FROM clusters;").fetchall()}

            # 8) Update tracks with cluster_id
            if not track_map_df.empty and name_to_id:
                update_payload = []
                for _, r in track_map_df.iterrows():
                    name = str(r["cluster_name"])
                    track_id = str(r["track_spotify_id"]).strip()
                    cid = name_to_id.get(name)
                    if cid is not None and track_id:
                        update_payload.append((cid, track_id))

                if update_payload:
                    conn.executemany(
                        "UPDATE tracks SET cluster_id = ? WHERE spotify_id = ?",
                        update_payload
                    )
                    updated = len(update_payload)

        print(f"Inserted clusters: {inserted} | Updated tracks: {updated}")
        return inserted, updated

    except Exception as e:
        # with conn: auto-rollback on exception
        # Surface a hint if cluster_name values are empty/whitespace
        raise

def get_emotions_from_dictionary() -> List[str]:
    """
    Step 1: Fetch all emotions from emotions_dictionary (SQLite).
    """
    conn = create_connection()
    try:
        # enable dict-like rows
        try:
            conn.row_factory = sqlite3.Row
        except Exception:
            pass
        with conn:
            cur = conn.execute("SELECT emotion FROM emotions_dictionary")
            emotions = [row["emotion"] for row in cur.fetchall()]
    finally:
        try:
            conn.close()
        except:
            pass

    if not emotions:
        raise ValueError("No emotions found in emotions_dictionary")
    return emotions


def build_avg_emotions_sql(emotions: List[str]) -> str:
    """
    Step 2: Build a dynamic SQL query with AVG(col) for each emotion (SQLite).
    Uses backticks to quote identifiers (accepted by SQLite).
    """
    if not emotions:
        raise ValueError("emotions list is empty")
    avg_exprs = [f"AVG(`{emo}`) AS `{emo}`" for emo in emotions]
    sql = f"SELECT {', '.join(avg_exprs)} FROM track_emotions_wide"
    return sql


def execute_avg_emotions_query(sql: str) -> Dict[str, Any]:
    """
    Step 3: Execute the generated SQL query and return the results
    with keys = original emotion names (SQLite).
    """
    conn = create_connection()
    try:
        try:
            conn.row_factory = sqlite3.Row
        except Exception:
            pass
        with conn:
            cur = conn.execute(sql)
            row = cur.fetchone()
    finally:
        try:
            conn.close()
        except:
            pass

    if row is None:
        return {}
    # Convert SQLite Row to plain dict with float-or-None values
    return {k: (float(v) if v is not None else None) for k, v in dict(row).items()}


def fetch_avg_per_emotion() -> Dict[str, Any]:
    """
    Orchestrates the 3 steps (SQLite):
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
      { emotion_lower: normalize_code }  (SQLite)
    """
    conn = create_connection()
    try:
        try:
            conn.row_factory = sqlite3.Row
        except Exception:
            pass
        with conn:
            cur = conn.execute("SELECT emotion, normalize FROM emotions_dictionary")
            rows = cur.fetchall()
    finally:
        try:
            conn.close()
        except:
            pass

    if not rows:
        raise ValueError("emotions_dictionary is empty or missing.")

    return {str(r["emotion"]).strip().lower(): str(r["normalize"]).strip() for r in rows}


def group_avg_emotions_by_normalize(avg_per_emotion: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Step 2: Receive a dict with per-emotion averages, e.g.:
      {'admiration': 3.2, 'anger': 1.8, ...}

    Group those averages by their normalize category from emotions_dictionary.

    Returns a list of dicts with:
      - normalize: category (FP/MP/N/MN/FN or Positive/Neutral/Negative)
      - sum_avg: sum of averages of all emotions in this group
      - emotion_count: how many emotions were grouped
    """
    if not isinstance(avg_per_emotion, dict) or not avg_per_emotion:
        raise ValueError("avg_per_emotion must be a non-empty dict like {'admiration': 3.2, ...}")

    mapping = fetch_emotion_normalize_mapping()  # {emotion_lower: normalize}
    groups: Dict[str, List[float]] = {}

    for emo_name, avg_val in avg_per_emotion.items():
        if avg_val is None:
            continue
        try:
            v = float(avg_val)
        except (TypeError, ValueError):
            continue
        norm = mapping.get(str(emo_name).strip().lower())
        if not norm:
            continue
        groups.setdefault(norm, []).append(v)

    results: List[Dict[str, Any]] = []
    preferred_order = ["FP", "MP", "N", "MN", "FN", "Positive", "Neutral", "Negative"]
    ordered_norms = [n for n in preferred_order if n in groups] + \
                    [n for n in sorted(groups.keys()) if n not in preferred_order]

    for norm in ordered_norms:
        vals = groups[norm]
        results.append({
            "normalize": norm,
            "sum_avg": sum(vals),
            "emotion_count": len(vals),
        })

    return results


def fetch_emotions_dictionary() -> List[Dict[str, Any]]:
    """
    Fetch all rows from emotions_dictionary.
    Returns a list of dicts with keys: emotion, normalize, emoji.
    """
    sql = "SELECT emotion, normalize, emoji FROM emotions_dictionary ORDER BY emotion"

    conn = create_connection()
    try:
        conn.row_factory = sqlite3.Row  # dict-like rows
        cur = conn.execute(sql)
        rows = cur.fetchall()
        result = [
            {
                "emotion":   (str(r["emotion"]).strip() if r["emotion"] is not None else ""),
                "normalize": (str(r["normalize"]).strip() if r["normalize"] is not None else ""),
                "emoji":     (str(r["emoji"]).strip() if r["emoji"] is not None else "")
            }
            for r in rows
        ]
    finally:
        try:
            conn.close()
        except:
            pass

    return result

def _get_existing_tew_columns() -> List[str]:
    """
    Return the list of existing columns in track_emotions_wide
    (excluding track_spotify_id), preserving table order.
    """
    conn = create_connection()
    try:
        conn.row_factory = sqlite3.Row
        # PRAGMA returns columns in table order via 'cid'
        rows = conn.execute("PRAGMA table_info('track_emotions_wide');").fetchall()
        cols = [r["name"] for r in rows]
    finally:
        try:
            conn.close()
        except:
            pass

    return [c for c in cols if c and c.strip().lower() != "track_spotify_id"]


def get_emotions_from_dictionary() -> List[str]:
    """
    Step 1 for dynamic AVG building: fetch emotion column names from emotions_dictionary.
    """
    conn = create_connection()
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT emotion FROM emotions_dictionary")
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
    Return all tracks (expected ~1000) with:
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
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]
    finally:
        try:
            conn.close()
        except:
            pass

def fetch_clusters() -> List[Dict[str, str]]:
    """
    Return all clusters as a list of dicts with id, name, description.
    """
    conn = create_connection()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, name, description FROM clusters ORDER BY id")
        return [dict(row) for row in cur.fetchall()]
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
        WHERE t.cluster_id = ?
        ORDER BY ar.name, a.name, t.spotify_id
    """
    conn = create_connection()
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, (cluster_id,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        try:
            conn.close()
        except:
            pass


# ---------- 4) Per-cluster averages for every emotion column ----------

def build_cluster_avg_emotions_sql(emotions: List[str]) -> str:
    """
    Build a dynamic SQL SELECT that computes AVG(col) AS `col` for all given emotions,
    restricted to a cluster via WHERE t.cluster_id = ?.
    The SELECT joins track_emotions_wide with tracks by spotify_id. (SQLite)
    """
    existing = set(_get_existing_tew_columns())
    valid_emotions = [e for e in emotions if e in existing]
    if not valid_emotions:
        raise ValueError("None of the emotions exist in track_emotions_wide")

    # Backticks are accepted by SQLite for quoting identifiers
    avg_exprs = [f"AVG(tew.`{emo}`) AS `{emo}`" for emo in valid_emotions]
    sql = f"""
        SELECT {', '.join(avg_exprs)}
        FROM track_emotions_wide AS tew
        JOIN tracks AS t ON t.spotify_id = tew.track_spotify_id
        WHERE t.cluster_id = ?
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
        conn.row_factory = sqlite3.Row
        row = conn.execute(sql, (cluster_id,)).fetchone()
    finally:
        try:
            conn.close()
        except:
            pass

    if row is None:
        return {e: None for e in emotions}

    return {k: (float(v) if v is not None else None) for k, v in dict(row).items()}


# ---------- Optional convenience: also return track_count for the cluster ----------

def fetch_cluster_avg_emotions_with_count(cluster_id: int) -> Dict[str, Any]:
    """
    Same as fetch_cluster_avg_emotions, but also returns 'track_count' for the cluster.
    """
    emo_avgs = fetch_cluster_avg_emotions(cluster_id)

    conn = create_connection()
    try:
        conn.row_factory = sqlite3.Row
        n = conn.execute("SELECT COUNT(*) AS n FROM tracks WHERE cluster_id = ?", (cluster_id,)).fetchone()["n"]
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
    Return all artists with their spotify_id and name. (SQLite)
    """
    sql = "SELECT spotify_id, name FROM artists ORDER BY name"
    conn = create_connection()
    try:
        conn.row_factory = sqlite3.Row  # dict-like rows
        rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]
    finally:
        try:
            conn.close()
        except:
            pass


def fetch_artist_by_id(artist_spotify_id: str) -> Optional[Dict[str, Any]]:
    """
    Return a single artist by its spotify_id. (SQLite)
    """
    sql = "SELECT spotify_id, name FROM artists WHERE spotify_id = ?"
    conn = create_connection()
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(sql, (artist_spotify_id,)).fetchone()
        return dict(row) if row else None
    finally:
        try:
            conn.close()
        except:
            pass


# ------------------- Albums -------------------

def fetch_all_albums() -> List[Dict[str, Any]]:
    """
    Return all albums with their spotify_id and name. (SQLite)
    """
    sql = "SELECT spotify_id, name FROM albums ORDER BY name"
    conn = create_connection()
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]
    finally:
        try:
            conn.close()
        except:
            pass


def fetch_album_by_id(album_spotify_id: str) -> Optional[Dict[str, Any]]:
    """
    Return a single album by its spotify_id. (SQLite)
    """
    sql = "SELECT spotify_id, name FROM albums WHERE spotify_id = ?"
    conn = create_connection()
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(sql, (album_spotify_id,)).fetchone()
        return dict(row) if row else None
    finally:
        try:
            conn.close()
        except:
            pass



# --- Buckets + pagination over track_emotions_wide (dictionary-driven).
BUCKET_MAP = {
    "FP": "full_positive",
    "MP": "positive",
    "N":  "neutral",
    "MN": "negative",
    "FN": "full_negative",
}

ALLOWED_SORT_KEYS = ["track", "artist", "album"] + list(BUCKET_MAP.values())

def _get_bucket_columns(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    """
    Discover which columns from track_emotions_wide belong to each bucket code
    using emotions_dictionary (emotion -> normalize[FP/MP/N/MN/FN]).
    Only keep emotions that exist as columns in track_emotions_wide.
    Returns: { 'FP': ['admiration', ...], 'MP': [...], ... }
    """
    # dictionary
    dic = conn.execute("SELECT emotion, normalize FROM emotions_dictionary").fetchall()
    mapping: Dict[str, List[str]] = {"FP": [], "MP": [], "N": [], "MN": [], "FN": []}
    if dic:
        # existing wide columns
        cols = [r["name"] for r in conn.execute("PRAGMA table_info('track_emotions_wide')").fetchall()]
        colset = {c for c in cols if c and c.lower() != "track_spotify_id"}
        for row in dic:
            emo = str(row["emotion"]).strip()
            code = str(row["normalize"]).strip().upper()
            if emo in colset and code in mapping:
                mapping[code].append(emo)
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
        # COALESCE(tew.`col`, 0) + ...
        pieces = [f"COALESCE(tew.`{c}` , 0.0)" for c in cols]
        out[pretty] = "(" + " + ".join(pieces) + ")"
    return out

def fetch_tracks_with_buckets_paginated(
    conn,
    page: int = 1,
    page_size: int = 10,
    track: Optional[str] = None,
    artist: Optional[str] = None,
    album: Optional[str] = None,
    sort_by: str = "track",      # track | artist | album | full_positive | positive | neutral | negative | full_negative
    sort_dir: str = "asc",       # asc | desc
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Return paginated rows with: track, artist, album and 5 bucket sums.
    Uses emotions_dictionary to decide which wide columns belong to each bucket.
    Pagination and ordering happen in SQL.
    """
    if page < 1 or page_size < 1:
        raise ValueError("page and page_size must be >= 1")

    sort_by = sort_by if sort_by in ALLOWED_SORT_KEYS else "track"
    sort_dir_sql = "DESC" if str(sort_dir).lower() == "desc" else "ASC"

    # Build filters
    wheres: List[str] = []
    params: List[Any] = []
    if track:
        wheres.append("t.name LIKE ?")
        params.append(f"%{track}%")
    if artist:
        wheres.append("a.name LIKE ?")
        params.append(f"%{artist}%")
    if album:
        wheres.append("al.name LIKE ?")
        params.append(f"%{album}%")
    where_sql = f"WHERE {' AND '.join(wheres)}" if wheres else ""

    # Count total tracks that match filters (no need to touch TEW)
    total_sql = f"""
        SELECT COUNT(1)
        FROM Tracks t
        LEFT JOIN Artists a ON a.spotify_id = t.artist_spotify_id
        LEFT JOIN Albums  al ON al.spotify_id = t.album_spotify_id
        {where_sql}
    """
    cur = conn.cursor()
    cur.execute(total_sql, params)
    total = int(cur.fetchone()[0])

    # Bucket expressions
    bucket_cols = _get_bucket_columns(conn)
    bucket_exprs = _build_bucket_sum_exprs(bucket_cols)  # {pretty: "SQL"}
    select_buckets = ",\n          ".join([f"{expr} AS {name}" for name, expr in bucket_exprs.items()])

    # Safe ORDER BY: whitelist column names
    if sort_by in ["track", "artist", "album"]:
        order_sql = f"ORDER BY {sort_by} {sort_dir_sql}, track ASC, artist ASC, album ASC"
    else:
        order_sql = f"ORDER BY {sort_by} {sort_dir_sql}, track ASC, artist ASC, album ASC"

    # Page query
    page_sql = f"""
        SELECT
          t.name               AS track,
          COALESCE(a.name,'')  AS artist,
          COALESCE(al.name,'') AS album,
          {select_buckets},
          t.emotions
        FROM Tracks t
        LEFT JOIN Artists a ON a.spotify_id = t.artist_spotify_id
        LEFT JOIN Albums  al ON al.spotify_id = t.album_spotify_id
        LEFT JOIN track_emotions_wide tew ON tew.track_spotify_id = t.spotify_id
        {where_sql}
        COLLATE NOCASE
        {order_sql}
        LIMIT ? OFFSET ?
    """
    cur.execute(page_sql, params + [page_size, (page - 1) * page_size])
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    return rows, total

def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """
    Return True if 'column' exists in 'table' (SQLite).
    """
    rows = conn.execute(f"PRAGMA table_info('{table}');").fetchall()
    # row_factory suele ser sqlite3.Row en tu create_connection()
    for r in rows:
        name = r["name"] if isinstance(r, sqlite3.Row) else r[1]
        if str(name).strip().lower() == column.strip().lower():
            return True
    return False


def has_null_keywords(conn: sqlite3.Connection | None = None) -> bool:
    """
    True si existe al menos un registro en Tracks con keywords IS NULL.
    """
    own = False
    if conn is None:
        conn = create_connection()
        own = True
    try:
        row = conn.execute("SELECT 1 FROM Tracks WHERE keywords IS NULL LIMIT 1;").fetchone()
        return row is not None
    finally:
        if own:
            try: conn.close()
            except: pass


def has_null_emotions(conn: sqlite3.Connection | None = None) -> bool:
    """
    True si existe al menos un registro en Tracks con emotions IS NULL.
    """
    own = False
    if conn is None:
        conn = create_connection()
        own = True
    try:
        row = conn.execute("SELECT 1 FROM Tracks WHERE emotions IS NULL LIMIT 1;").fetchone()
        return row is not None
    finally:
        if own:
            try: conn.close()
            except: pass


def has_null_cluster_id(conn: sqlite3.Connection | None = None) -> bool:
    """
    True si existe al menos un registro en Tracks con cluster_id IS NULL.
    Si la columna 'cluster_id' no existe, devuelve False (no hay nulos que revisar).
    """
    own = False
    if conn is None:
        conn = create_connection()
        own = True
    try:
        if not _column_exists(conn, "Tracks", "cluster_id"):
            return False  # la columna aún no ha sido creada en este esquema
        row = conn.execute("SELECT 1 FROM Tracks WHERE cluster_id IS NULL LIMIT 1;").fetchone()
        return row is not None
    finally:
        if own:
            try: conn.close()
            except: pass
