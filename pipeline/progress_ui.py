# progress_ui.py
import queue
from typing import Optional, Tuple
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser

class ProgressUI:
    """
    Tkinter UI that MUST be created and run on the main thread (macOS-safe).
    Thread-safe API for workers:
      - set_message(text)
      - set_progress(0..100 or None for indeterminate)
      - set_indeterminate(True/False)
      - close()

    Usage:
        ui = ProgressUI(title="Pipeline", initial_message="Starting...")
        # start worker thread...
        ui.run()  # blocks main thread in the Tk event loop until close()
    """
    def __init__(self, title: str = "Working...", initial_message: str = "Please wait...", width: int = 500):
        self._title = title
        self._initial_message = initial_message
        self._width = width

        self._cmds: "queue.Queue[Tuple[str, Optional[object]]]" = queue.Queue()

        # Build UI on MAIN thread
        self._root = tk.Tk()
        self._root.title(self._title)
        # Enough height for message + progress bar
        self._root.geometry(f"{self._width}x140")
        self._root.resizable(False, False)
        try:
            self._root.attributes("-topmost", True)
        except Exception:
            pass

        container = ttk.Frame(self._root, padding=16)
        container.pack(fill="both", expand=True)

        self._label = ttk.Label(
            container,
            text=self._initial_message,
            anchor="w",
            wraplength=self._width - 40,
            justify="left",
        )
        self._label.pack(fill="x", pady=(0, 12))

        # Wider, more visible progress bar
        self._bar = ttk.Progressbar(
            container,
            mode="determinate",       # start in determinate; switch when needed
            maximum=100.0,
            length=self._width - 40,  # make it wide and obvious
        )
        self._bar.pack(fill="x")
        self._bar["value"] = 0.0

        # Internal state
        self._indeterminate = False
        self._running = True

        # Close behavior
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Kick off command polling
        self._root.after(50, self._process_commands)

    # ---------- Public API ----------
    @property
    def parent(self) -> tk.Misc:
        """Public Tk parent for dialogs/windows (root)."""
        return self._root

    def set_message(self, text: str) -> None:
        """Update status label (thread-safe)."""
        self._cmds.put(("message", text))

    def set_progress(self, value: Optional[float]) -> None:
        """
        Update progress:
          - value in 0..100 -> determinate mode with that value
          - value is None   -> switch to indeterminate spinner
        """
        if value is None:
            self.set_indeterminate(True)
            return
        # Switch to determinate and send clamped value
        try:
            v = float(value)
        except Exception:
            v = 0.0
        v = max(0.0, min(100.0, v))
        self._cmds.put(("progress", v))

    def set_indeterminate(self, enabled: bool = True) -> None:
        """Toggle indeterminate (animated) mode."""
        self._cmds.put(("indeterminate", bool(enabled)))

    def close(self) -> None:
        """Close window and end mainloop (thread-safe)."""
        self._cmds.put(("close", None))

    # ---------- Mainloop control (call on main thread) ----------
    def run(self) -> None:
        """Block in Tk mainloop until close() is called."""
        self._root.mainloop()

    # ---------- Internals ----------
    def _process_commands(self):
        try:
            while True:
                cmd, payload = self._cmds.get_nowait()
                if cmd == "message":
                    self._label.config(text=str(payload))

                elif cmd == "progress":
                    # Switch to determinate and update value
                    if self._indeterminate:
                        self._bar.stop()
                        self._indeterminate = False
                    self._bar.config(mode="determinate")
                    self._bar["value"] = float(payload)

                elif cmd == "indeterminate":
                    enable = bool(payload)
                    if enable and not self._indeterminate:
                        # Faster, smoother marquee (smaller interval = faster)
                        self._bar.config(mode="indeterminate")
                        self._bar.start(8)  # 8ms per step -> very visible
                        self._indeterminate = True
                    elif not enable and self._indeterminate:
                        self._bar.stop()
                        self._bar.config(mode="determinate")
                        self._indeterminate = False

                elif cmd == "close":
                    return self._safe_destroy()
        except queue.Empty:
            pass

        if self._running:
            self._root.after(50, self._process_commands)

    def _on_close(self):
        self.close()

    def _safe_destroy(self):
        self._running = False
        try:
            self._bar.stop()
        except Exception:
            pass
        try:
            self._root.destroy()
        except Exception:
            pass

def show_server_dialog(root: tk.Misc, open_url: str = "http://127.0.0.1:8000"):
    """
    Muestra un diálogo modal con dos opciones:
      - Abrir servidor: inicia el server y deshabilita el botón
      - Terminar sesión: detiene el server (si está corriendo), cierra la app y sale

    root: ventana principal (usa ui.parent de tu ProgressUI)
    """
    dlg = tk.Toplevel(root)
    dlg.title("Servidor")
    dlg.transient(root)   # asociar al root
    dlg.grab_set()        # modal
    try:
        dlg.attributes("-topmost", True)
    except Exception:
        pass

    # Layout simple
    frame = ttk.Frame(dlg, padding=16)
    frame.pack(fill="both", expand=True)

    title = ttk.Label(frame, text="¿Qué deseas hacer ahora?", font=("", 12, "bold"))
    title.pack(anchor="w", pady=(0, 8))

    status_var = tk.StringVar(value="Servidor inactivo.")
    status = ttk.Label(frame, textvariable=status_var)
    status.pack(anchor="w", pady=(0, 12))

    btns = ttk.Frame(frame)
    btns.pack(fill="x")

    # Estado interno
    server_running = {"value": False}

    def on_open():
        if server_running["value"]:
            return
        try:
            import server as srv
            srv.main("../playlist-emotions/dist")
            server_running["value"] = True
            status_var.set(f"Servidor en ejecución en {open_url}")
            open_btn.config(state="disabled")
            # opcional: abrir navegador
            try:
                webbrowser.open_new_tab(open_url)
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Error al iniciar", str(e), parent=dlg)

    def on_quit():
        try:
            import server as srv
            if server_running["value"]:
                srv.stop_server()
        except Exception:
            pass
        try:
            root.destroy()
        except Exception:
            pass
        sys.exit(0)

    open_btn = ttk.Button(btns, text="Abrir servidor", command=on_open)
    open_btn.pack(side="left", padx=(0, 8))

    quit_btn = ttk.Button(btns, text="Terminar sesión", command=on_quit)
    quit_btn.pack(side="left")

    # Cerrar la ventana = terminar sesión
    dlg.protocol("WM_DELETE_WINDOW", on_quit)

    # Tamaño mínimo y centrado suave
    dlg.update_idletasks()
    w, h = 380, 160
    dlg.minsize(w, h)
    try:
        sw = dlg.winfo_screenwidth()
        sh = dlg.winfo_screenheight()
        x = int((sw - w) / 2)
        y = int((sh - h) / 3)
        dlg.geometry(f"{w}x{h}+{x}+{y}")
    except Exception:
        pass

    # Devolver el control al mainloop (modal con grab_set ya hecho)
    dlg.wait_window()