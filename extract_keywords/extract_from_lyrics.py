import extract_keywords.keywords as kwUtils
import libraries.db as db
import extract_keywords.LLM as LLM

if __name__ == "__main__":
    '''
    Vamos a seguir los siguientes pasos:
    1. Vamos a obtener los keywords de todas nuestros tracks y enviarlos a un LLM para obtener los keywords
    2. Guardamos los datos dentro de la base de datos.
    '''

    conn = db.create_connection()
    lyrics = db.get_lyrics(conn)

    try:
        for spotify_id, lyric in lyrics.items():
            print(f"Processing {spotify_id}")
            cleanLyric = kwUtils.clean_lyrics(lyric)
            print(f"Cleaned {cleanLyric[0:30]}...")
            keywords = LLM.llm_keywords_en_simple(cleanLyric,20)
            print(keywords)
            db.save_keywords(conn, {spotify_id: keywords})
            print("Next...")

    except Exception as e:
        print("Error:", e)

    finally:
        conn.close()
