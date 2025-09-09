import libraries.sqlite as db
import pandas as pd
import itertools
import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import os
from transformers import pipeline
import ast
from collections import defaultdict
import re

def apply_normalization_array(df: pd.DataFrame, norm_dict: dict) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["normalized_keywords"] = df_copy["keywords"].apply(
        lambda kws: list({
            norm_dict.get(str(k).lower().strip(), str(k).lower().strip())
            for k in kws if isinstance(k, str) and k.strip()
        })
    )
    return df_copy

def top_normalized_keywords(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Returns the top N most frequent normalized keywords across all tracks.
    """
    all_kws = list(itertools.chain.from_iterable(df["normalized_keywords"]))
    counter = Counter(all_kws)
    top = counter.most_common(top_n)
    return pd.DataFrame(top, columns=["keyword", "count"])

def load_df_if_needed(df: pd.DataFrame | None = None,
                      csv_path: str = "tracks_with_normalized_keywords.csv") -> pd.DataFrame:
    """
    Returns a DataFrame ready for processing.
    If df is None OR df does not contain 'normalized_keywords', tries to load from CSV.
    """
    if df is None or "normalized_keywords" not in df.columns:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"DataFrame is missing and '{csv_path}' was not found. "
                "Provide a DataFrame with columns: track_spotify_id, artist_name, title, keywords (list)."
            )
        df = pd.read_csv(csv_path)
    return df


def ensure_keywords_list(df: pd.DataFrame, col: str = "keywords") -> pd.DataFrame:
    """
    Robustly converts the column `col` into a Python list[str] per row.
    Handles:
      - JSON lists: ["love","party"]
      - Python repr lists: ['love', 'party']
      - Double-escaped JSON (e.g., '"[\"love\",\"party\"]"')
      - Comma-separated fallbacks: love, party
      - NaN/None → []
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")

    def _strip_outer_quotes(s: str) -> str:
        # Remove a single pair of wrapping quotes if present: '"[...]' or "'[...]'"
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            return s[1:-1]
        return s

    def _parse(x):
        # Already a list
        if isinstance(x, list):
            return [str(t).strip() for t in x if isinstance(t, (str, int, float)) and str(t).strip()]

        # Missing
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []

        # String-like
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return []

            # Try to un-wrap if double-escaped: e.g., '"[\"love\",\"party\"]"'
            s_unwrapped = _strip_outer_quotes(s)

            # 1) Try JSON
            for candidate in (s, s_unwrapped):
                if candidate.startswith("[") and candidate.endswith("]"):
                    try:
                        v = json.loads(candidate)
                        if isinstance(v, list):
                            return [str(t).strip() for t in v if str(t).strip()]
                    except Exception:
                        pass

            # 2) Try Python literal (handles single quotes lists)
            for candidate in (s, s_unwrapped):
                if candidate.startswith("[") and candidate.endswith("]"):
                    try:
                        v = ast.literal_eval(candidate)
                        if isinstance(v, list):
                            return [str(t).strip() for t in v if str(t).strip()]
                    except Exception:
                        pass

            # 3) Fallback: comma-separated
            parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
            return [p for p in parts if p]

        # Anything else → try stringify
        try:
            s = str(x).strip()
            if s.startswith("[") and s.endswith("]"):
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [str(t).strip() for t in v if str(t).strip()]
        except Exception:
            pass
        return []

    out = df.copy()
    out[col] = out[col].apply(_parse)
    return out


# -------------------------
# Emotion analysis (model-based, no lexicon)
# -------------------------
# Load a pre-trained emotion classifier (GoEmotions fine-tuned DistilRoBERTa)
# 1) Multi-label emotion pipeline (GoEmotions)
_emotion_pipe = None


def get_emotion_pipeline_multilabel():
    """
    Multi-label emotion classifier based on GoEmotions.
    Uses sigmoid (not softmax) under the hood.
    """
    global _emotion_pipe
    if _emotion_pipe is None:
        _emotion_pipe = pipeline(
            "text-classification",
            model="joeddav/distilbert-base-uncased-go-emotions-student",
            return_all_scores=True,
            top_k=None,  # return all labels with scores
            truncation=True
        )
    return _emotion_pipe


# 2) Utility: classify a list of keywords IN BATCH, then aggregate per label
def analyze_emotion_from_keywords_multilabel(
        keywords: list[str],
        min_score_threshold: float = 0.15,  # keep labels with mean score >= threshold
        top_k: int = 5,  # return top-k emotions after threshold
        batch_size: int = 32,
        max_keywords: int | None = 100  # cap to avoid very long batches
) -> list[dict]:
    """
    Multi-label emotion analysis:
      - Classifies each keyword separately (batched).
      - Averages scores per emotion across keywords.
      - Filters by threshold and returns top_k labels.

    Returns a list of dicts: [{"label": "...", "score": 0.xx}, ...]
    """
    if not keywords:
        return []

    # Clean and cap keywords
    kws = [str(k).strip() for k in keywords if isinstance(k, str) and str(k).strip()]
    if not kws:
        return []
    if max_keywords is not None:
        kws = kws[:max_keywords]

    pipe = get_emotion_pipeline_multilabel()

    # Batch inference
    all_scores = pipe(kws, batch_size=batch_size)  # list of list[{"label","score"}]

    # Aggregate scores per label (mean over keywords)
    # Initialize label space from the first result
    if not all_scores or not all_scores[0]:
        return []

    label_scores = defaultdict(list)
    labels = [d["label"] for d in all_scores[0]]
    for per_kw in all_scores:
        for d in per_kw:
            label_scores[d["label"]].append(float(d["score"]))

    mean_scores = {lab: (sum(vals) / len(vals)) for lab, vals in label_scores.items()}

    # Threshold and sort
    filtered = [{"label": lab, "score": sc} for lab, sc in mean_scores.items() if sc >= min_score_threshold]
    filtered.sort(key=lambda x: x["score"], reverse=True)

    # Take top_k (if threshold filters too much and nothing remains, fall back to top_k without threshold)
    if not filtered:
        fallback = [{"label": lab, "score": sc} for lab, sc in mean_scores.items()]
        fallback.sort(key=lambda x: x["score"], reverse=True)
        return fallback[:top_k]
    return filtered[:top_k]


# 3) Attach emotions to your DataFrame using the ORIGINAL 'keywords' column
def attach_emotions_multilabel(df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Adds 'emotions' column with the aggregated multi-label results
    computed from the 'keywords' column for each row.
    """
    df_out = df.copy()
    df_out["emotions"] = df_out["keywords"].apply(
        lambda kws: analyze_emotion_from_keywords_multilabel(
            kws,
            min_score_threshold=0.15,
            top_k=top_k
        )
    )
    return df_out

def all_emotions_from_df(df):
    """
    Returns a sorted list of unique emotion labels present in df['emotions'].
    """
    labels = set()
    for lst in df["emotions"]:
        if isinstance(lst, list):
            for d in lst:
                lab = str(d.get("label", "")).strip().lower()
                if lab:
                    labels.add(lab)
    return sorted(labels)


def to_snake(s: str) -> str:
    """
    Safe snake_case for MySQL column names.
    """
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def df_to_emotion_wide(df: pd.DataFrame, emotions: list[str]) -> pd.DataFrame:
    """
    Returns a wide DataFrame with one column per emotion (percentage 0–100),
    filling 0 where the emotion is absent.
    """
    cols = ["track_spotify_id"] + [to_snake(e) for e in emotions]
    out_rows = []
    for _, row in df.iterrows():
        base = {c: 0.0 for c in cols}
        base["track_spotify_id"] = str(row["track_spotify_id"]).strip()
        lst = row["emotions"] if isinstance(row["emotions"], list) else []
        for d in lst:
            lab = str(d.get("label", "")).strip().lower()
            score = round(float(d.get("score", 0.0) * 100), 4)
            col = to_snake(lab)
            if col in base:
                val = score  # store as percentage
                # if multiple entries for the same label, keep max
                base[col] = max(base[col], val)
        out_rows.append(base)
    wide = pd.DataFrame(out_rows, columns=cols)
    return wide

def main(set_message=None):
    conn = db.create_connection()
    tracks = db.fetch_tracks_dataset(conn)
    conn.close()
    df = pd.DataFrame(tracks, columns=["track_spotify_id", "artist_name", "title", "keywords"])

    # ========= 1) Create a unique keywords list =========
    all_kws = list(itertools.chain.from_iterable(df['keywords']))
    unique_kws = sorted({
        " ".join(str(k).lower().split())
        for k in all_kws
        if isinstance(k, str) and k.strip()
    })

    if set_message:
        set_message(f"Total unique keywords: {len(unique_kws)}")

    print(f"Total unique keywords: {len(unique_kws)}")

    # ========= 2) Embeddings =========
    model = SentenceTransformer('all-MiniLM-L6-v2')
    emb = model.encode(unique_kws, normalize_embeddings=True)

    # ========= 3) Hierarchical cluster =========
    clu = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage='average',
        distance_threshold=0.4  # ajusta el threshold: más pequeño = clusters más finos
    )
    labels = clu.fit_predict(emb)

    # ========= 4) Build the clusters =========
    clusters = {}
    for kw, lab in zip(unique_kws, labels):
        clusters.setdefault(lab, []).append(kw)

    # ========= 5) Choose canonical representative per cluster =========
    def pick_rep(words):
        # heurística: palabra con menos tokens, si empata, la más corta
        return sorted(words, key=lambda w: (len(w.split()), len(w)))[0]

    suggested_norm = {}
    for words in clusters.values():
        rep = pick_rep(words)
        for w in words:
            suggested_norm[w] = rep

    if set_message:
        set_message(f"Suggested keywords: {len(clusters)}")

    # ========= 6) Save suggested dictionary =========
    with open("keyword_norm_suggested.json", "w", encoding="utf-8") as f:
        json.dump(suggested_norm, f, indent=2, ensure_ascii=False)

    unique_norm = set(suggested_norm.values())
    if set_message:
        set_message(f"{len(unique_kws)} unique words - suggested words {len(unique_norm)}")

    print(f"{len(unique_kws)} unique words - suggested words {len(unique_norm)}")

    df_norm = apply_normalization_array(df, suggested_norm)

    top_normalized_keywords(df_norm, top_n=20)

    df_norm.to_csv("tracks_with_normalized_keywords.csv", index=False, encoding="utf-8")

    conn = db.create_connection()
    db.ensure_normalized_keywords_column(conn)
    db.update_normalized_keywords(conn, df_norm)
    conn.close()

    if set_message:
        set_message(f"Keywords normalized and saved to DB")

    # If you already have a DataFrame named `df`, pass it directly to attach_emotions(df, top_k=3).
    # Otherwise, load from CSV (expects at least: track_spotify_id, artist_name, title, keywords):
    try:
        df = load_df_if_needed(df=None, csv_path="tracks_with_normalized_keywords.csv")
    except FileNotFoundError as e:
        print(e)
        raise

    # Ensure keywords are lists, then attach emotions
    df = ensure_keywords_list(df, col="keywords")
    df = attach_emotions_multilabel(df, top_k=28)  # Top K depends on how many emotions the dataset can describe.

    # Get the first non-empty emotions entry
    first_non_empty = df.loc[df["emotions"].map(lambda x: isinstance(x, list) and len(x) > 0), "emotions"].iloc[0]
    second = df.loc[df["emotions"].map(lambda x: isinstance(x, list) and len(x) > 0), "emotions"].iloc[1]

    # Pretty print as JSON for readability
    print(json.dumps(first_non_empty, indent=2))
    print(json.dumps(second, indent=2))

    emotions = all_emotions_from_df(df)
    emotion_cols = [to_snake(e) for e in emotions]


    if set_message:
        set_message(f"{len(emotion_cols)} Emotions discovered")

    print(len(emotion_cols), "emotions discovered")

    # Example: run once in MySQL
    table_name = "track_emotions_wide"
    ddl = db.build_create_table_sql(table_name, emotion_cols)

    wide_df = df_to_emotion_wide(df, emotions)

    if set_message:
        set_message(f"Value: {wide_df.iloc[0, 1:].sum()}")

    print(f"Value: {wide_df.iloc[0, 1:].sum()}")  # The value needs to be ~= 100

    conn = db.create_connection()
    try:
        db.truncate_track_emotions_wide(conn)
    except:
        print("Tabla no truncada")

    if set_message:
        set_message("Saving Sentimental Analysis to DB")

    db.execute_DDL(conn, ddl)
    db.upsert_emotion_wide(conn, table_name, wide_df)
    conn.close()

    if set_message:
        set_message("Data was successfully upserted")

    conn = db.create_connection()
    db.update_track_emotions(conn, df)
    conn.close()

    if set_message:
        set_message("Data was successfully upserted")

    return wide_df, df











