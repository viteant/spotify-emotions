# libraries/spotify.py
# PKCE-only helpers so notebooks can pass client_id explicitly.
# Comments in English only.

from typing import Dict, Any, Generator, Tuple, Optional
from spotipy import Spotify
from spotipy.oauth2 import SpotifyPKCE

def get_spotify_auth(
    client_id: str,
    redirect_uri: str = "https://spotify-auth.viant.dev/callback",
    scope: str = "user-library-read",
    cache_path: str = ".cache",
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
    cache_path: str = ".cache",
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
