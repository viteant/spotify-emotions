import os
import sys
import pathlib
import threading
import tkinter as tk
from tkinter import simpledialog, messagebox
from typing import Optional, Tuple


def ensure_openai_api_key(env_path: str = "../.env", parent: Optional[tk.Misc] = None) -> str:
    """
    Ensure OPENAI_API_KEY exists.
      1) If present in os.environ or .env -> return it.
      2) Else show a modal input (masked) over `parent` without hiding the main window.
         - Cancel -> exit(1)
         - OK     -> write to .env and return the value

    Notes:
    - `parent` MUST be a Tk root/Toplevel running on the main thread.
    - If called from a worker thread, this function marshals UI calls to the Tk main thread.
    """

    KEY = "OPENAI_API_KEY"

    # 1) Environment
    val = os.environ.get(KEY)
    if val:
        return val

    # 2) .env file
    def _parse_env_line(line: str) -> Tuple[Optional[str], Optional[str]]:
        s = line.strip()
        if not s or s.startswith("#"):
            return None, None
        if s.startswith("export "):
            s = s[len("export "):].lstrip()
        if "=" not in s:
            return None, None
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        return k, v

    def _read_from_env_file(path: str, key: str) -> Optional[str]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    k, v = _parse_env_line(raw)
                    if k == key:
                        return v
        except Exception:
            return None
        return None

    val = _read_from_env_file(env_path, KEY)
    if val:
        os.environ[KEY] = val
        return val

    # 3) Prompt GUI
    if parent is None:
        raise RuntimeError(
            "ensure_openai_api_key needs a Tk parent. Pass parent=ui.parent (your root)."
        )

    # Run a Tk call on the main thread when invoked from a worker
    def _on_main(fn, *args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            return fn(*args, **kwargs)
        result = {}
        done = threading.Event()
        def wrapper():
            try:
                result["v"] = fn(*args, **kwargs)
            finally:
                done.set()
        parent.after(0, wrapper)
        done.wait()
        return result.get("v")

    # Ask for the key using a hidden Toplevel so the main window stays visible
    def _ask_secret_modal(owner: tk.Misc, title: str, prompt: str) -> Optional[str]:
        # Create a tiny hidden toplevel as the dialog's parent
        tl = tk.Toplevel(owner)
        tl.withdraw()
        tl.transient(owner)
        tl.grab_set()  # modal
        try:
            tl.attributes("-topmost", True)
        except Exception:
            pass
        try:
            value = simpledialog.askstring(title=title, prompt=prompt, show="*", parent=tl)
        finally:
            try:
                tl.destroy()
            except Exception:
                pass
        return value

    val = _on_main(_ask_secret_modal, parent, "OpenAI API Key Required", "Enter your OPENAI_API_KEY:")

    if not val:
        _on_main(messagebox.showwarning,
                 "Missing Key",
                 "OPENAI_API_KEY was not provided. Exiting.",
                 parent=parent)
        sys.exit(1)

    # Write/update .env
    def _write_key(path: str, key: str, value: str) -> None:
        lines = []
        found = False
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for i, raw in enumerate(lines):
                s = raw.strip()
                s_noexp = s[len("export "):].lstrip() if s.startswith("export ") else s
                if s_noexp.startswith(f"{key}="):
                    lines[i] = f"{key}={value}\n"
                    found = True
                    break
        if not found:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] = lines[-1] + "\n"
            lines.append(f"{key}={value}\n")
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    _write_key(env_path, KEY, val)
    os.environ[KEY] = val

    _on_main(messagebox.showinfo,
             "Saved",
             f"OPENAI_API_KEY saved to {env_path}.",
             parent=parent)

    return val
