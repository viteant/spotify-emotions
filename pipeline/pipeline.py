# main.py
import threading

from prompt_toolkit.shortcuts import progress_dialog

import utils
import libraries.sqlite as db
import data_extraction
import extract_keywords.extract_from_title_artist as keywords
import sentimental_analysis
import emotion_dictionary
import clustering

from progress_ui import ProgressUI  # si pusiste el helper en otro archivo
from progress_ui import show_server_dialog

def run_pipeline(ui: ProgressUI):
    try:
        ui.set_message("Extracting data...")
        data_extraction.main()

        ui.set_message("Ensuring OpenAI API key...")
        utils.ensure_openai_api_key(env_path="../.env", parent=ui.parent)

        # Keywords
        if db.has_null_keywords():
            ui.set_message("Extracting keywords (title & artist)...")
            keywords.extract_keywords_from_title_artist(1000, ui.set_message)

        # Emotions
        df = None
        if db.has_null_emotions():
            ui.set_message("Sentimental Analysis...")
            wide_df, df = sentimental_analysis.main(set_message=ui.set_message)

            ui.set_message("Emotion Dictionary...")
            emotion_dictionary.main(wide_df, ui.set_message)

        # Clustering
        if db.has_null_cluster_id():
            ui.set_message("Clustering...")
            # si df es None porque no corriste sentimental antes, tu función clustering.main debería poder manejarlo
            df_emb, df_emb, df_named = clustering.main(df, ui.set_message)

        ui.set_message("Listo.")

        # Mostrar el diálogo de opciones sobre la ventana actual (en el hilo de Tk)
        ui.parent.after(0, lambda: show_server_dialog(ui.parent))

    except Exception as e:
        # Si algo peta, muestra error y sal
        from tkinter import messagebox
        ui.parent.after(0, lambda: messagebox.showerror("Error", str(e), parent=ui.parent))
        ui.parent.after(0, ui.close)


if __name__ == "__main__":
    # Create UI on MAIN thread (macOS requires it)
    ui = ProgressUI(title="Pipeline", initial_message="Starting...", width=560)

    # Worker thread for your pipeline
    worker = threading.Thread(target=run_pipeline, args=(ui,), daemon=True)
    worker.start()

    # Block here in Tk mainloop; UI will be closed by worker via ui.close()
    ui.run()
