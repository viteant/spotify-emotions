import libraries.sqlite as db
import extract_keywords.LLM as LLM

'''
        print(f"Processing {id}")
        cleanLyric = kwUtils.clean_lyrics(lyric)
        print(f"Cleaned {cleanLyric[0:30]}...")
        keywords = LLM.llm_keywords_en_simple(cleanLyric, 20)
        print(keywords)
        db.save_keywords(conn, {spotify_id: keywords})
        print("Next...")
'''

def extract_keywords_from_title_artist():
    conn = db.create_connection()
    songs = db.get_tracks_missing_lyrics(conn, 1000, 0)
    i = 1
    total = len(songs)
    for song in songs:
        track_id, title, artist = (
            song["track_spotify_id"],
            song["title"],
            song["artist_name"]
        )
        print(f"Processing {artist}-{title} - {i}/{total}")
        keywords = LLM.llm_keywords_from_title_artist(artist, title, 20)
        print(keywords)
        db.save_keywords(conn, {track_id: keywords})
        i += 1
    
