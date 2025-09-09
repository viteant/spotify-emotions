# libraries/spotify.py
# PKCE-only helpers so notebooks can pass client_id explicitly.
# Comments in English only.

from typing import Dict, Any, Generator, Tuple, Optional
from spotipy import Spotify
from spotipy.oauth2 import SpotifyPKCE
from typing import Iterable, List, Optional, Tuple
import re

_SPOTIFY_TRACK_URL_RE = re.compile(r"open\.spotify\.com/track/([A-Za-z0-9]+)")


def get_spotify_auth(
    client_id: str,
    redirect_uri: str = "https://spotify-auth.viant.dev/callback",
    scope: str = "user-library-read",
    cache_path: str = "../.cache",
    open_browser: bool = True,
) -> SpotifyPKCE:
    """Return a PKCE auth manager configured from function arguments."""
    return SpotifyPKCE(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        open_browser=open_browser,
        cache_path=cache_path,
    )

def get_spotify_client(
    client_id: str,
    redirect_uri: str = "https://auth.viant.dev/callback",
    scope: str = "user-library-read",
    cache_path: str = "../.cache",
    open_browser: bool = True,
) -> Spotify:
    """Build a Spotify client using PKCE with the provided configuration."""
    auth = get_spotify_auth(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        cache_path=cache_path,
        open_browser=open_browser,
    )
    return Spotify(auth_manager=auth)

def pick_best_image(images) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """Return url, height, width of the first image if available."""
    if images and isinstance(images, list):
        img = images[0]
        return img.get("url"), img.get("height"), img.get("width")
    return None, None, None

def iter_saved_tracks(sp: Spotify, page_limit: int = 50) -> Generator[Dict[str, Any], None, None]:
    """
    Yield normalized dicts for user's saved tracks.
    Fields:
      - artist: {spotify_id, name}
      - album:  {spotify_id, name, image_url, image_height, image_width}
      - track:  {spotify_id, name, popularity, href}
    """
    offset = 0
    total = None
    while True:
        page = sp.current_user_saved_tracks(limit=page_limit, offset=offset)
        if total is None:
            total = page.get("total", 0)

        items = page.get("items", [])
        if not items:
            break

        for it in items:
            track = it.get("track") or {}
            if not track:
                continue

            t_id   = track.get("id")
            t_name = track.get("name", "")
            t_pop  = int(track.get("popularity", 0) or 0)
            t_href = track.get("href", "")

            artists = track.get("artists", []) or []
            if not artists:
                continue
            a = artists[0]
            a_id   = a.get("id")
            a_name = a.get("name", "")

            album = track.get("album", {}) or {}
            al_id   = album.get("id")
            al_name = album.get("name", "")
            img_url, img_h, img_w = pick_best_image(album.get("images", []))

            if not (t_id, a_id, al_id):
                continue

            yield {
                "artist": {"spotify_id": a_id, "name": a_name},
                "album": {
                    "spotify_id": al_id, "name": al_name,
                    "image_url": img_url, "image_height": img_h, "image_width": img_w,
                    "artist_spotify_id": a_id
                },
                "track": {
                    "spotify_id": t_id, "name": t_name, "popularity": t_pop,
                    "href": t_href, "artist_spotify_id": a_id, "album_spotify_id": al_id
                },
            }
        offset += page_limit
        if total is not None and offset >= total:
            break


# --- Playlist helpers (PKCE; works with get_spotify_client) ---

def _to_track_uris(track_ids_or_urls: Iterable[str]) -> List[str]:
    """Accept raw IDs, URIs, or URLs; normalize to spotify:track:<id> URIs."""
    out: List[str] = []
    for s in track_ids_or_urls:
        if not s:
            continue
        if s.startswith("spotify:track:"):
            out.append(s)
            continue
        m = _SPOTIFY_TRACK_URL_RE.search(s)
        if m:
            out.append(f"spotify:track:{m.group(1)}")
        else:
            out.append(f"spotify:track:{s}")
    return out

def _chunked(seq: List[str], size: int = 100):
    """Yield chunks of 'size' elements."""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def get_current_user_id(sp: Spotify) -> str:
    """Return the current user's Spotify ID."""
    return sp.current_user()["id"]

def find_playlist_by_name(sp: Spotify, name: str) -> Optional[dict]:
    """Return the first playlist matching 'name' (case-sensitive) among the user's playlists."""
    limit, offset = 50, 0
    while True:
        page = sp.current_user_playlists(limit=limit, offset=offset)
        for it in page.get("items", []):
            if it.get("name") == name:
                return it
        if page.get("next"):
            offset += limit
        else:
            return None

def create_playlist(sp: Spotify, name: str, description: str = "", public: bool = False) -> Tuple[str, str]:
    """Create a playlist and return (playlist_id, playlist_url)."""
    uid = get_current_user_id(sp)
    pl = sp.user_playlist_create(user=uid, name=name, public=public, description=description)
    return pl["id"], pl["external_urls"]["spotify"]

def add_tracks(sp: Spotify, playlist_id: str, track_ids_or_urls: List[str]) -> List[str]:
    uris = _to_track_uris(track_ids_or_urls)
    if not uris:
        print("[add_tracks] nothing to add; _to_track_uris returned 0 items")
        return []
    print(f"[add_tracks] adding {len(uris)} URIs")
    snapshots: List[str] = []
    for batch in _chunked(uris, 100):
        resp = sp.playlist_add_items(playlist_id, batch)
        snapshots.append(resp.get("snapshot_id"))
    return snapshots

def replace_tracks(sp: Spotify, playlist_id: str, track_ids_or_urls: List[str]) -> List[str]:
    uris = _to_track_uris(track_ids_or_urls)
    if not uris:
        print("[replace_tracks] nothing to replace; _to_track_uris returned 0 items")
        return []
    print(f"[replace_tracks] replacing with {len(uris)} URIs total")
    snapshots: List[str] = []
    if len(uris) <= 100:
        resp = sp.playlist_replace_items(playlist_id, uris)
        snapshots.append(resp.get("snapshot_id"))
    else:
        resp = sp.playlist_replace_items(playlist_id, uris[:100])
        snapshots.append(resp.get("snapshot_id"))
        for batch in _chunked(uris[100:], 100):
            resp = sp.playlist_add_items(playlist_id, batch)
            snapshots.append(resp.get("snapshot_id"))
    return snapshots

def upsert_playlist_by_name(
    sp: Spotify,
    name: str,
    description: str,
    track_ids_or_urls: List[str],
    public: bool = False,
    replace: bool = True,
) -> Tuple[str, str]:
    """
    Create or update a playlist by 'name' and return (playlist_id, playlist_url).
    - replace=True: overwrite items with provided list
    - replace=False: append items
    """
    existing = find_playlist_by_name(sp, name)
    if existing:
        print("[upsert] playlist exists -> updating items")
        pid = existing["id"]
        url = existing["external_urls"]["spotify"]
        sp.playlist_change_details(pid, name=name, public=public, description=description)
        if track_ids_or_urls:
            if replace:
                snapshots = replace_tracks(sp, pid, track_ids_or_urls)
            else:
                snapshots = add_tracks(sp, pid, track_ids_or_urls)
            print(f"[upsert] snapshots: {snapshots}")
        else:
            print("[upsert] no tracks provided; leaving items as-is")
        return pid, url

    # Not found: create and THEN add tracks (tu bug estaba aquí)
    print("[upsert] playlist not found -> creating and adding items")
    pid, url = create_playlist(sp, name, description=description, public=public)
    if track_ids_or_urls:
        snapshots = add_tracks(sp, pid, track_ids_or_urls)
        print(f"[upsert] snapshots: {snapshots}")
    else:
        print("[upsert] created empty playlist (no tracks provided)")
    return pid, url