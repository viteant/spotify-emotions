import os
import time
import threading
import webbrowser
from typing import Optional, Dict, Any

from flask import Flask, request

# Spotify logic (PKCE-only helpers and iterators)
from libraries.spotify import get_spotify_client, iter_saved_tracks

# SQLite backend
from libraries.sqlite import (
    create_connection,
    ensure_database_and_tables,
    upsert_artist,
    upsert_album,
    upsert_track,
)

# =========================
# Configuration (edit here)
# =========================
CLIENT_ID: str = "ec6c43ef03034d7393a4906cd1df5f06"         # do not commit real values to public repos
REDIRECT_URI: str = "https://spotify-auth.viant.dev/callback"  # must match Spotify dashboard
SCOPE: str = "user-library-read playlist-modify-public playlist-modify-private"
CACHE_PATH: str = "../.cache"                        # per-user cache file
LOCAL_CALLBACK_PORT: int = 8080                  # must match your Vercel relay target (127.0.0.1:<PORT>)

# =========================
# Server state (global)
# =========================
SERVER_STATE: Dict[str, Optional[Any]] = {"server": None, "thread": None}


def build_spotify_pkce(
    client_id: str,
    redirect_uri: str,
    scope: str,
    cache_path: str,
    open_browser: bool = False,
):
    """Return a PKCE auth manager (Spotipy SpotifyPKCE)."""
    from spotipy.oauth2 import SpotifyPKCE  # PKCE class (no client_secret)
    return SpotifyPKCE(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        open_browser=open_browser,  # we manually open the browser
        cache_path=cache_path,
    )


def wait_for_token(auth_mgr, timeout: int = 300, interval: float = 1.0) -> bool:
    """Poll the cache until a valid token exists or timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            token = auth_mgr.get_cached_token()
            if token and token.get("access_token"):
                return True
        except Exception:
            # Some spotipy versions may raise if cache is empty; ignore and keep polling
            pass
        time.sleep(interval)
    return False


def run_server(app: Flask) -> None:
    """Run Flask using a Werkzeug WSGI server we can stop programmatically."""
    from werkzeug.serving import make_server
    httpd = make_server("127.0.0.1", LOCAL_CALLBACK_PORT, app)
    SERVER_STATE["server"] = httpd
    # Blocks current thread; intended to be called from a background thread.
    httpd.serve_forever()


def stop_server() -> None:
    """Stop the local auth callback server if it is running."""
    server = SERVER_STATE.get("server")
    thread = SERVER_STATE.get("thread")

    if server is None:
        print("ℹ️ No server is currently running.")
        return

    try:
        server.shutdown()  # Properly shut down the Werkzeug server
    except Exception as e:
        print(f"⚠️ Failed to shutdown server cleanly: {e}")

    if thread and thread.is_alive():
        thread.join(timeout=3)

    SERVER_STATE.update({"server": None, "thread": None})
    print("🛑 Server stopped.")


def ensure_pkce_authorized(client_id: str, redirect_uri: str, scope: str, cache_path: str) -> None:
    """
    Ensure PKCE flow is completed.
    If a valid token is already cached, nothing happens.
    Otherwise:
      - starts a local Flask server on 127.0.0.1:<PORT>/callback
      - opens the Spotify consent page
      - exchanges the 'code' and stores tokens in cache
      - blocks until the token is available or timeout
    """
    auth = build_spotify_pkce(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        cache_path=cache_path,
        open_browser=False,
    )

    # If token already cached, skip auth
    try:
        token = auth.get_cached_token()
        if token and token.get("access_token"):
            return
    except Exception:
        pass

    app = Flask(__name__)

    @app.route("/callback")
    def callback():
        code = request.args.get("code")
        if not code:
            return "Missing authorization code.", 400
        # Complete token exchange and persist to cache
        auth.get_access_token(code, check_cache=False)
        return "Auth OK. You can close this tab."

    # Start local receiver with a stoppable server and open consent page
    t = threading.Thread(target=run_server, args=(app,), daemon=True)
    t.start()
    SERVER_STATE["thread"] = t

    auth_url = auth.get_authorize_url()
    webbrowser.open_new_tab(auth_url)
    print(f"Open this URL if the browser did not pop up:\n{auth_url}")
    print(f"Waiting for callback on http://127.0.0.1:{LOCAL_CALLBACK_PORT}/callback ...")

    # Block until token is present (or timeout)
    ok = wait_for_token(auth, timeout=300, interval=1.0)

    # Always stop the local server to free the port
    stop_server()

    if not ok:
        raise TimeoutError("PKCE authorization timed out. Try again or check redirect/callback setup.")


def run_import(client_id: str, redirect_uri: str, scope: str, cache_path: str) -> None:
    """Fetch saved tracks from Spotify and persist them using selected backend."""
    # Build a Spotify client using the same config used for PKCE cache
    sp = get_spotify_client(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        cache_path=cache_path,
        open_browser=False,   # no need; token already cached
    )

    conn = create_connection()
    ensure_database_and_tables(conn)

    count = 0
    for row in iter_saved_tracks(sp, page_limit=50):
        a = row["artist"]
        al = row["album"]
        t = row["track"]

        upsert_artist(conn, a["spotify_id"], a["name"])
        upsert_album(
            conn,
            al["spotify_id"],
            al["name"],
            al["artist_spotify_id"],
            al["image_url"],
            al["image_height"],
            al["image_width"],
        )
        upsert_track(
            conn,
            t["spotify_id"],
            t["name"],
            t["popularity"],
            t["href"],
            t["artist_spotify_id"],
            t["album_spotify_id"],
        )
        count += 1

    conn.close()
    print(f"Imported or updated {count} tracks.")


def main():
    print("Authorizing...")

    # 1) Complete PKCE (opens browser if cache is empty, waits until token is stored)
    ensure_pkce_authorized(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_path=CACHE_PATH,
    )

    print("Start to import tracks")

    # 2) Run the import using the cached token
    run_import(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_path=CACHE_PATH,
    )
