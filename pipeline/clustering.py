import ast
import pandas as pd
import os, json, re
from openai import OpenAI
import numpy as np
from typing import List
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import libraries.sqlite as db


def ensure_list_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Ensures column `col` is a Python list[str].
    Handles:
      - Python repr lists with single quotes: "['a', 'b']"
      - JSON lists: ["a","b"]
      - Comma-separated fallback: a, b
    """

    def _parse(x):
        if isinstance(x, list):
            return [str(t).strip() for t in x if str(t).strip()]
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return []
            # 1) Try Python literal (handles single quotes)
            if s.startswith("[") and s.endswith("]"):
                try:
                    v = ast.literal_eval(s)
                    if isinstance(v, list):
                        return [str(t).strip() for t in v if str(t).strip()]
                except Exception:
                    pass
                # 2) Try JSON (double quotes)
                try:
                    v = json.loads(s)
                    if isinstance(v, list):
                        return [str(t).strip() for t in v if str(t).strip()]
                except Exception:
                    pass
                # 3) Fallback: split inner by comma
                inner = s[1:-1]
                parts = [p.strip().strip("'").strip('"') for p in inner.split(",")]
                return [p for p in parts if p]
            # 4) Plain comma-separated
            return [t.strip() for t in s.split(",") if t.strip()]
        # last resort
        return [str(x).strip()] if str(x).strip() else []

    out = df.copy()
    out[col] = out[col].apply(_parse)
    return out


# ---------------------------
# 1) Build embeddings per track
# ---------------------------
def compute_track_embeddings(
        df: pd.DataFrame,
        keywords_col: str = "normalized_keywords",
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = True,
        max_keywords: int | None = None
) -> tuple[np.ndarray, List[int], SentenceTransformer]:
    """
    Computes a dense embedding per track by averaging the embeddings of its normalized keywords.
    Returns:
      - X: np.ndarray of shape (n_tracks, dim)
      - valid_idx: indices of rows that have at least 1 keyword (used to subset df later)
      - model: the SentenceTransformer model used
    """
    model = SentenceTransformer(model_name)
    vectors = []
    valid_idx = []

    for i, kws in enumerate(df[keywords_col].tolist()):
        if not isinstance(kws, list) or len(kws) == 0:
            continue
        toks = [str(k).strip() for k in kws if isinstance(k, str) and str(k).strip()]
        if not toks:
            continue
        if max_keywords is not None:
            toks = toks[:max_keywords]

        kw_emb = model.encode(toks, normalize_embeddings=normalize)
        if isinstance(kw_emb, list):
            kw_emb = np.asarray(kw_emb)
        track_vec = kw_emb.mean(axis=0)  # mean pooling over keywords
        vectors.append(track_vec)
        valid_idx.append(i)

    if not vectors:
        raise ValueError("No tracks produced embeddings. Check 'normalized_keywords' column.")
    X = np.vstack(vectors)
    return X, valid_idx, model


# ---------------------------
# 2) KMeans clustering
# ---------------------------
def cluster_tracks_embeddings(
        X: np.ndarray,
        k: int = 8,
        random_state: int = 42
) -> tuple[np.ndarray, KMeans, float]:
    """
    Clusters the embedding matrix X with KMeans.
    Returns:
      - labels: cluster assignment per row in X
      - kmeans: fitted model
      - sil: silhouette score (for quick quality check)
    """
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan
    return labels, km, sil


# ---------------------------
# 3) Attach clusters back to df
# ---------------------------
def attach_clusters_to_df(
        df: pd.DataFrame,
        labels: np.ndarray,
        valid_idx: List[int],
        cluster_col: str = "cluster_emb"
) -> pd.DataFrame:
    """
    Adds the cluster labels to a copy of df at the indices used for embeddings.
    Rows without embeddings remain with NaN clusters.
    """
    out = df.copy()
    out[cluster_col] = np.nan
    out.loc[out.index[valid_idx], cluster_col] = labels
    if out[cluster_col].isna().any():
        out[cluster_col] = out[cluster_col].astype("Int64")
    return out


# ---------------------------
# 4) Cluster summaries (human-readable)
# ---------------------------
def top_keywords_per_cluster(
        df_with_clusters: pd.DataFrame,
        cluster_col: str = "cluster_emb",
        keywords_col: str = "normalized_keywords",
        top_n: int = 12
) -> dict[int, List[tuple[str, int]]]:
    """
    For interpretability: returns top-N most frequent normalized keywords per cluster.
    """
    summary = {}
    for c in sorted(df_with_clusters[cluster_col].dropna().unique()):
        sub = df_with_clusters[df_with_clusters[cluster_col] == c]
        cnt = Counter()
        for kws in sub[keywords_col]:
            if isinstance(kws, list):
                cnt.update([k for k in kws if isinstance(k, str) and k.strip()])
        summary[int(c)] = cnt.most_common(top_n)
    return summary


def sample_titles_per_cluster(
        df_with_clusters: pd.DataFrame,
        cluster_col: str = "cluster_emb",
        n_samples: int = 5
) -> dict[int, List[str]]:
    """
    Returns up to n_samples "Artist — Title" examples per cluster.
    """
    examples = {}
    for c in sorted(df_with_clusters[cluster_col].dropna().unique()):
        sub = df_with_clusters[df_with_clusters[cluster_col] == c].head(n_samples)
        examples[int(c)] = [
            f"{row['artist_name']} — {row['title']}"
            for _, row in sub.iterrows()
        ]
    return examples


def build_cluster_payload(df_with_clusters: pd.DataFrame,
                          cluster_col: str = "cluster_emb",
                          keywords_col: str = "normalized_keywords",
                          title_cols=("artist_name", "title"),
                          top_k_keywords: int = 15,
                          max_examples: int = 5) -> dict:
    """
    Prepares a payload per cluster:
      {cluster_id: {"keywords": [...], "examples": ["Artist — Title", ...]}}
    """
    payload = {}
    for cid in sorted(df_with_clusters[cluster_col].dropna().unique()):
        sub = df_with_clusters[df_with_clusters[cluster_col] == cid]
        # top keywords by simple frequency inside the cluster
        kw_series = sub[keywords_col].explode()
        kw_series = kw_series[kw_series.notna()].astype(str).str.strip().str.lower()
        top = (kw_series.value_counts().head(top_k_keywords).index.tolist()
               if not kw_series.empty else [])
        payload[int(cid)] = {"keywords": top}
    return payload

import json, re

def parse_json_maybe(s: str) -> dict:
    # Normalize input
    if s is None:
        return {}
    s = s.strip().lstrip("\ufeff")

    # 1) Strip code fences if the whole thing is fence-wrapped
    s_nofence = re.sub(r"^\s*```(?:json|js|javascript)?\s*|\s*```\s*$", "", s, flags=re.I|re.S)
    try:
        return json.loads(s_nofence)
    except Exception:
        pass

    # 2) Extract JSON from a fenced block if present
    m = re.search(r"```(?:json|js|javascript)?\s*(\{.*?\})\s*```", s, flags=re.I|re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 3) Raw-decode from first '{' that yields valid JSON
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, end = dec.raw_decode(s[i:])
                return obj
            except Exception:
                continue

    # 4) Very last resort: grab a curly block and try fixing trailing commas
    m2 = re.search(r"\{.*\}", s, flags=re.S)
    if m2:
        frag = m2.group(0)
        # remove trailing commas before } or ]
        frag = re.sub(r",\s*([}\]])", r"\1", frag)
        try:
            return json.loads(frag)
        except Exception:
            pass

    return {}


def name_clusters_with_llm(cluster_payload: dict,
                           model: str = "gpt-4.1-mini",
                           temperature: float = 0.2) -> dict:
    """
    Sends cluster summaries to the LLM and returns:
      {cluster_id: {"name": str, "description": str}}
    Robust JSON parsing with fallback.
    """
    # Build prompt
    items = []
    for cid, data in cluster_payload.items():
        kws = ", ".join(data["keywords"])
        items.append(
            f"Cluster {cid}:\n"
            f"- Top keywords: {kws or '(none)'}\n"
        )
    clusters_block = "\n\n".join(items)

    prompt = (
        "You are an expert music curator. Name each cluster of songs concisely.\n"
        "Rules:\n"
        "- Provide a short, human-friendly name (≤ 4 words) and a 1–2 sentence description.\n"
        "- Avoid artist names and proper nouns; focus on themes/moods.\n"
        "- Prefer genre/vibe/emotion terms (e.g., 'Dancefloor Energy', 'Melancholic Love').\n"
        "- Keep names in Title Case, English only.\n"
        "Return ONLY a JSON object mapping cluster id to an object with fields 'name' and 'description'.\n\n"
        "Example output format:\n"
        "{\n"
        "  \"0\": {\"name\": \"Late-Night Romance\", \"description\": \"Sensual, intimate themes with slow dance vibes.\"},\n"
        "  \"1\": {\"name\": \"Party Anthems\", \"description\": \"High-energy club tracks centered on dancing and celebration.\"}\n"
        "}\n\n"
        "Clusters to label:\n"
        f"{clusters_block}\n\n"
        "Now return ONLY the JSON."
    )

    print("Prompt:",prompt)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    resp = client.responses.create(model=model, input=prompt, temperature=temperature)

    text = resp.output_text.strip()
    print("Resp:", text)

    data = parse_json_maybe(text)
    # Normalize keys to ints if possible
    out = {}
    for k, v in data.items():
        try:
            cid = int(k)
        except Exception:
            cid = k
        name = (v or {}).get("name", "").strip()
        desc = (v or {}).get("description", "").strip()
        if name:
            out[cid] = {"name": name, "description": desc}
    return out


def attach_cluster_names(df_with_clusters: pd.DataFrame,
                         labels_map: dict,
                         cluster_col: str = "cluster_emb") -> pd.DataFrame:
    """
    Adds 'cluster_name' and 'cluster_desc' columns using labels_map from LLM.
    """
    out = df_with_clusters.copy()
    out["cluster_name"] = out[cluster_col].map(
        lambda c: labels_map.get(int(c), {}).get("name") if pd.notna(c) else None)
    out["cluster_desc"] = out[cluster_col].map(
        lambda c: labels_map.get(int(c), {}).get("description") if pd.notna(c) else None)
    return out


def main(df, set_message=None):
    print("CLUSTERING************")

    if set_message:
        set_message("Creating clusters...")

    # Convert the column from string to list[str]
    df = ensure_list_column(df, col="normalized_keywords")

    # Sanity check
    print(type(df.loc[df.index[0], "normalized_keywords"]), df.loc[df.index[0], "normalized_keywords"][:5])

    # Build embeddings
    X, valid_idx, st_model = compute_track_embeddings(
        df, keywords_col="normalized_keywords", model_name="all-MiniLM-L6-v2", normalize=True
    )

    # Cluster (choose k based on your dataset size; start with 6–12)
    labels, kmeans, sil = cluster_tracks_embeddings(X, k=15, random_state=42)
    print("Silhouette:", sil)

    # Attach cluster labels back to df
    df_emb = attach_clusters_to_df(df, labels, valid_idx, cluster_col="cluster_emb")

    # Inspect
    df_emb[["track_spotify_id", "artist_name", "title", "cluster_emb"]].head(10)

    # Top keywords per cluster (for interpretation)
    cluster_keywords = top_keywords_per_cluster(df_emb, cluster_col="cluster_emb", keywords_col="normalized_keywords",
                                                top_n=20)
    for cid, tops in cluster_keywords.items():
        print(f"\nCluster {cid} top keywords:")
        print(", ".join([f"{w}({c})" for w, c in tops]))

    # Optional: sample titles per cluster
    examples = sample_titles_per_cluster(df_emb, cluster_col="cluster_emb", n_samples=10)
    for cid, ex in examples.items():
        print(f"\nCluster {cid} examples:")
        for s in ex:
            print(" -", s)

    if set_message:
        set_message("Naming clusters...")

    # 1) Build payload from your clustered DataFrame (assumes 'cluster_emb' and 'normalized_keywords')
    payload = build_cluster_payload(df_emb, cluster_col="cluster_emb", keywords_col="normalized_keywords",
                                    top_k_keywords=20, max_examples=5)
    print("\nPayload:")
    print(payload)
    print("\n")

    # 2) Ask the LLM for names/descriptions
    labels_map = name_clusters_with_llm(payload, model="gpt-4.1-mini", temperature=0.2)
    print("\nLabels:")
    print(labels_map)
    print("\n")

    if set_message:
        set_message("Saving clusters...")

    # 3) Attach names back to your DataFrame
    df_named = attach_cluster_names(df_emb, labels_map, cluster_col="cluster_emb")
    df_named[["cluster_emb", "cluster_name", "cluster_desc"]].drop_duplicates().sort_values("cluster_emb").head(20)

    df_named.to_csv("tracks_with_normalized_keywords.csv", index=False, encoding="utf-8")

    conn = db.create_connection()
    db.sync_clusters_and_update_tracks(conn, df_named)
    conn.close()

    return df, df_emb, df_named
