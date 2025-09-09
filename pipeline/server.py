# server.py
import os
import time
import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount

import libraries.sqlite as db
from pydantic import BaseModel
from libraries.spotify import get_spotify_auth, get_spotify_client, upsert_playlist_by_name

# Spotify config (env-driven; must match your app's settings)
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "ec6c43ef03034d7393a4906cd1df5f06")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI", "https://spotify-auth.viant.dev/callback")
SPOTIPY_CACHE_PATH = os.getenv("SPOTIPY_CACHE_PATH", "../.cache")
# For playlist ops we need these scopes; add others as required
SPOTIPY_SCOPE = os.getenv("SPOTIPY_SCOPE", "user-library-read playlist-modify-public playlist-modify-private")



# =========================
# FastAPI app
# =========================
app = FastAPI()

# CORS (relax as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# API router under /api
# =========================
api = APIRouter(prefix="/api")

@api.get("/health")
def health():
    return {"status": "ok"}

@api.get("/playlist-emotions")
def getNormalizeEmotions():
    avgEmotions = db.group_avg_emotions_by_normalize(db.fetch_avg_per_emotion())
    return {"emotions": avgEmotions}

@api.get("/playlist-emotions/all")
def getAllEmotions():
    avgEmotions = db.fetch_avg_per_emotion()
    return {"emotions": avgEmotions, "dictionary": db.fetch_emotions_dictionary()}

@api.get("/dictionary/emotions")
def get_emotions_dic():
    return db.fetch_emotions_dictionary()

@app.get("/callback")
def spotify_callback(code: str | None = None, error: str | None = None):
    if error:
        raise HTTPException(status_code=400, detail=f"Spotify error: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    auth = get_spotify_auth(
        client_id=SPOTIPY_CLIENT_ID,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope=SPOTIPY_SCOPE,
        cache_path=SPOTIPY_CACHE_PATH,
        open_browser=True,
    )
    # Intercambia el code y guarda el token en .cache
    auth.get_access_token(code, check_cache=True)
    return {"status": "ok", "message": "Auth OK. You can close this tab."}

@api.get("/emotion-tracks")
def list_tracks(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=200),
    track: str | None = Query(None, description="Filter by track name (case-insensitive)"),
    artist: str | None = Query(None, description="Filter by artist name (case-insensitive)"),
    album: str | None = Query(None, description="Filter by album name (case-insensitive)"),
    sort_by: str = Query("track", description="Sort key"),
    sort_dir: str = Query("asc", description="asc or desc"),
):
    sort_by = sort_by if sort_by in db.ALLOWED_SORT_KEYS else "track"
    sort_dir = "desc" if sort_dir.lower() == "desc" else "asc"

    try:
        conn = db.create_connection()
        items, total = db.fetch_tracks_with_buckets_paginated(
            conn=conn,
            page=page,
            page_size=page_size,
            track=track,
            artist=artist,
            album=album,
            sort_by=sort_by,
            sort_dir=sort_dir,
        )
    except Exception as e:
        parts = [str(a) for a in getattr(e, "args", []) if a]
        detail = " ".join(parts) if parts else str(e)
        raise HTTPException(status_code=500, detail=f"Database error: {detail}".strip())
    finally:
        try:
            conn.close()
        except:
            pass

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "sort_by": sort_by,
        "sort_dir": sort_dir,
        "items": items,
    }

@api.get("/playlist-cluster")
def get_playlist_cluster():
    return db.fetch_all_tracks_with_clusters()

@api.get("/playlist-cluster/{cluster_id}")
def get_playlist_cluster(
        cluster_id: int,
):
    return db.fetch_tracks_by_cluster(cluster_id)

@api.get("/clusters")
def get_playlist_cluster_all():
    return db.fetch_clusters()

@app.get("/spotify/callback")
def spotify_callback(code: str | None = None, error: str | None = None):
    """
    PKCE callback: exchange 'code' for tokens and persist them in the cache.
    """
    if error:
        raise HTTPException(status_code=400, detail=f"Spotify error: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    auth = get_spotify_auth(
        client_id=SPOTIPY_CLIENT_ID,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope=SPOTIPY_SCOPE,
        cache_path=SPOTIPY_CACHE_PATH,
        open_browser=True,
    )
    # Store tokens in cache
    auth.get_access_token(code, check_cache=True)
    return {"status": "ok", "message": "Auth OK. You can close this tab."}

class ClusterPlaylistBody(BaseModel):
    description: str = "Generated from cluster"
    public: bool = False
    replace: bool = True  # replace items instead of append

@api.post("/clusters/{cluster_id}/playlist")
def create_or_update_cluster_playlist(cluster_id: int, body: ClusterPlaylistBody):
    """
    Build a Spotify client from cached PKCE tokens, fetch cluster tracks from SQLite,
    and upsert a playlist named after the cluster.
    """
    # 1) Create Spotipy client with playlist scopes using your existing helper
    sp = get_spotify_client(
        client_id=SPOTIPY_CLIENT_ID,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope=SPOTIPY_SCOPE,
        cache_path=SPOTIPY_CACHE_PATH,
        open_browser=False,
    )

    # 2) Pull tracks for cluster
    rows = db.fetch_tracks_by_cluster(cluster_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No tracks found for cluster_id={cluster_id}")

    cluster_name = rows[0].get("cluster_name") or f"Cluster {cluster_id}"
    track_ids = [r["track_spotify_id"] for r in rows if r.get("track_spotify_id")]

    print(f"Tracks found: {len(track_ids)}")
    print(track_ids)

    if not track_ids:
        raise HTTPException(status_code=404, detail=f"No track_spotify_id values for cluster_id={cluster_id}")

    # 3) Create/update playlist by name
    pid, url = upsert_playlist_by_name(
        sp=sp,
        name=cluster_name,
        description=body.description,
        track_ids_or_urls=track_ids,
        public=body.public,
        replace=body.replace,
    )

    # Comprueba propiedad del playlist (debe ser tu usuario actual)
    owner = sp.playlist(pid, fields="owner.id").get("owner", {}).get("id")
    current = sp.current_user().get("id")
    if owner != current:
        raise HTTPException(status_code=403, detail=f"Playlist owner is {owner}, not {current}. You cannot modify it.")

    # Lee total de items tras la operación
    total_after = sp.playlist_items(pid, fields="total").get("total")
    print(f"Total tracks: {total_after}")

    snapshots = []  # captura lo que retorne upsert internamente si lo propagas
    pl_info = sp.playlist_items(pid, fields="total")
    return {
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "requested_tracks": len(track_ids),
        "playlist_id": pid,
        "playlist_url": url,
        "playlist_items_after": pl_info.get("total"),
        "snapshots": snapshots,  # útil para confirmar mutaciones
    }


app.include_router(api)


# =========================
# SPA static mounting with history fallback
# =========================
class SPAStaticFiles(StaticFiles):
    """
    Static files with history-fallback: if path not found, serve index.html.
    This enables Vue Router (history mode) deep links.
    """
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 404:
            return await super().get_response("index.html", scope)
        return response


def initialize_frontend(
    frontend_dir: Optional[str] = None,
    mount_path: str = "/",
    env_var: str = "FRONTEND_DIR",
    default_rel: str = "../frontend/dist",
) -> str:
    """
    Mount a built Vue app (SPA) at `mount_path`, with history fallback.
      - If `frontend_dir` is None, uses $FRONTEND_DIR or `default_rel`.
      - Idempotent: skips if already mounted at `mount_path`.
      - Returns absolute path used.
    """
    base = frontend_dir or os.getenv(env_var, default_rel)
    abs_dir = os.path.abspath(base)

    # Avoid duplicate mounts
    for r in app.routes:
        if isinstance(r, Mount) and r.path == mount_path:
            print(f"[init] SPA already mounted at {mount_path} → {abs_dir}")
            return abs_dir

    index_path = os.path.join(abs_dir, "index.html")
    if not os.path.exists(index_path):
        print(f"[warn] {abs_dir} does not contain index.html. Did you run `npm run build`?")

    app.mount(mount_path, SPAStaticFiles(directory=abs_dir, html=True), name="spa")
    print(f"[init] SPA mounted at {mount_path} from {abs_dir}  (API under /api)")
    return abs_dir


# =========================
# Server controls
# =========================
SERVER_STATE = {"server": None, "thread": None}

def start_server(host="127.0.0.1", port=8080):
    """
    Start Uvicorn in a background thread.
    """
    if SERVER_STATE["server"] is not None:
        print(f"⚠️ Server already running at http://{host}:{port}")
        return

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    time.sleep(0.5)
    SERVER_STATE.update({"server": server, "thread": thread})
    print(f"🚀 Server running at http://{host}:{port}  (API under /api)")

def stop_server():
    """
    Stop the background Uvicorn server.
    """
    server = SERVER_STATE.get("server")
    thread = SERVER_STATE.get("thread")
    if server is None:
        print("ℹ️ No server is currently running.")
        return
    server.should_exit = True
    if thread and thread.is_alive():
        thread.join(timeout=3)
    SERVER_STATE.update({"server": None, "thread": None})
    print("🛑 Server stopped.")

def main(frontend_dir:str = None):
    # Mount frontend (uses FRONTEND_DIR env var or ../frontend/dist)
    initialize_frontend(frontend_dir=frontend_dir, mount_path="/")
    # Start server
    start_server(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8080")))
