#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
from typing import Dict, Any, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

import mysql.connector
from mysql.connector import errorcode
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth

# =========================
# CONFIG
# =========================
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "")
DB_NAME = os.getenv("DB_NAME", "spotify_songs")

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:3000/callback")
SPOTIFY_SCOPE = "user-library-read"

LIMIT = 50  # Spotify permite 50 por página en saved tracks

# =========================
# DB HELPERS
# =========================
def get_mysql_conn(database: Optional[str] = None):
    cfg = {
        "host": DB_HOST,
        "port": DB_PORT,
        "user": DB_USER,
        "password": DB_PASS,
        "charset": "utf8mb4",
        "autocommit": True
    }
    if database:
        cfg["database"] = database
    return mysql.connector.connect(**cfg)

def ensure_database_and_tables():
    # Crea BD si no existe y tablas
    ddl = [
        "CREATE DATABASE IF NOT EXISTS spotify_songs CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;",
        "USE spotify_songs;",
        # Artists
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
        # Albums
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
        # Tracks
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
        """
    ]

    conn = get_mysql_conn()
    cur = conn.cursor()
    for stmt in ddl:
        cur.execute(stmt)
    cur.close()
    conn.close()

def upsert_artist(conn, spotify_id: str, name: str):
    sql = """
    INSERT INTO Artists (spotify_id, name)
    VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE
      name = VALUES(name),
      updated_at = CURRENT_TIMESTAMP;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (spotify_id, name))

def upsert_album(conn, spotify_id: str, name: str, artist_spotify_id: str,
                 image_url: Optional[str], image_height: Optional[int], image_width: Optional[int]):
    sql = """
    INSERT INTO Albums (spotify_id, name, image_url, image_height, image_width, artist_spotify_id)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      name = VALUES(name),
      image_url = VALUES(image_url),
      image_height = VALUES(image_height),
      image_width = VALUES(image_width),
      artist_spotify_id = VALUES(artist_spotify_id),
      updated_at = CURRENT_TIMESTAMP;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (spotify_id, name, image_url, image_height, image_width, artist_spotify_id))

def upsert_track(conn, spotify_id: str, name: str, popularity: int, href: str,
                 artist_spotify_id: str, album_spotify_id: str):
    sql = """
    INSERT INTO Tracks (spotify_id, name, popularity, href, artist_spotify_id, album_spotify_id)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      name = VALUES(name),
      popularity = VALUES(popularity),
      href = VALUES(href),
      artist_spotify_id = VALUES(artist_spotify_id),
      album_spotify_id = VALUES(album_spotify_id),
      updated_at = CURRENT_TIMESTAMP;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (spotify_id, name, popularity, href, artist_spotify_id, album_spotify_id))

# =========================
# SPOTIFY HELPERS
# =========================
def get_spotify_client() -> Spotify:
    auth = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SPOTIFY_SCOPE,
        cache_path="../.cache-spotify"
    )
    return Spotify(auth_manager=auth)

def pick_best_image(images) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Devuelve la primera imagen si existe (Spotify ordena de mayor a menor).
    """
    if images and isinstance(images, list) and len(images) > 0:
        img = images[0]
        return img.get("url"), img.get("height"), img.get("width")
    return None, None, None

# =========================
# MAIN
# =========================
def main():
    # 1) Garantizar BD y tablas
    ensure_database_and_tables()
    conn = get_mysql_conn(DB_NAME)

    # 2) Cliente Spotify
    sp = get_spotify_client()

    # 3) Paginación saved tracks de 50 en 50
    offset = 0
    total = None

    while True:
        page = sp.current_user_saved_tracks(limit=LIMIT, offset=offset)
        if total is None:
            total = page.get("total", 0)
            print(f"Total saved tracks reportado por Spotify: {total}")

        items = page.get("items", [])
        if not items:
            break

        for it in items:
            track = it.get("track", {})
            if not track:
                continue

            t_id   = track.get("id")
            t_name = track.get("name", "")
            t_pop  = int(track.get("popularity", 0) or 0)
            t_href = track.get("href", "")

            # Artista principal (primero de la lista)
            artists = track.get("artists", []) or []
            if not artists:
                # sin artista no podemos normalizar bien — lo omitimos
                continue
            a = artists[0]
            a_id   = a.get("id")
            a_name = a.get("name", "")

            # Álbum
            album = track.get("album", {}) or {}
            al_id   = album.get("id")
            al_name = album.get("name", "")
            img_url, img_h, img_w = pick_best_image(album.get("images", []))

            # UPSERT Artist
            if a_id:
                upsert_artist(conn, a_id, a_name)

            # UPSERT Album (asociado al artista principal)
            if al_id and a_id:
                upsert_album(conn, al_id, al_name, a_id, img_url, img_h, img_w)

            # UPSERT Track (con FKs a artista y álbum)
            if t_id and a_id and al_id:
                upsert_track(conn, t_id, t_name, t_pop, t_href, a_id, al_id)

        offset += LIMIT
        if offset >= total:
            break

    conn.close()
    print("¡Listo! Guardados Artists, Albums y Tracks de tus Saved Tracks.")

if __name__ == "__main__":
    # Validación básica
    missing = [k for k in ["SPOTIFY_CLIENT_ID","SPOTIFY_CLIENT_SECRET","SPOTIFY_REDIRECT_URI"] if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Faltan variables de entorno: {missing}. Configura tu .env primero.")
    main()
