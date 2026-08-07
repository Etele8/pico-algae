"""
Pico-Algae Counter - screen-capture control.

Runs the local web app together with a small always-on-top window that has a
"Capture" button. Point the Olympus software at an image, click Capture, and the
model runs on a screenshot of the region you picked. All captures collect in one
browser tab where you inspect, correct and save each one (screenshot + annotated
image + a running counts.csv) to your chosen output folder.

No microscopy file needs to be exported first - it reads straight off the screen.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox

# Make screen coordinates match real pixels on high-DPI monitors. Must run
# before Tk starts and before any screen grab so selection and capture agree.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor aware
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()    # system aware (older Windows)
    except Exception:
        pass

import numpy as np
from PIL import ImageGrab

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server as S  # Flask app + add_capture() + OUTPUT_DIR

HOST, PORT = "127.0.0.1", 5000
BASE_URL = f"http://{HOST}:{PORT}"
CAPTURES_URL = f"{BASE_URL}/captures"
# Config (region + output folder) lives in server.cfg_load/cfg_set so the
# capture window and the browser folder box share one source of truth.


def virtual_screen():
    """(x, y, w, h) of the whole virtual desktop spanning all monitors."""
    try:
        u = ctypes.windll.user32
        return (u.GetSystemMetrics(76), u.GetSystemMetrics(77),
                u.GetSystemMetrics(78), u.GetSystemMetrics(79))
    except Exception:
        return (0, 0, 1920, 1080)


def start_server():
    try:
        S.get_counter()  # preload the model so the first capture is fast
    except Exception as e:  # noqa: BLE001
        print("[pico] model load error:", e)
    S.app.run(host=HOST, port=PORT, debug=False, use_reloader=False, threaded=True)


class CaptureApp:
    def __init__(self):
        region = S.cfg_load().get("region")
        self.region = tuple(region) if region else None
        # server already restored the saved output folder on import

        self.root = tk.Tk()
        self.root.title("Pico Capture")
        self.root.attributes("-topmost", True)
        self.root.geometry("380x300+40+40")
        self.root.configure(bg="#0f172a")
        self._build()

    def _build(self):
        tk.Label(self.root, text="🦠  Pico-Algae Capture", bg="#0f172a", fg="#e2e8f0",
                 font=("Segoe UI", 14, "bold")).pack(pady=(14, 2))
        self.status = tk.Label(self.root, text="", bg="#0f172a", fg="#94a3b8",
                               font=("Segoe UI", 9), wraplength=350, justify="center")
        self.status.pack(pady=(0, 6))

        self.cap_btn = tk.Button(self.root, text="📸  Capture & Review", command=self.capture,
                                 bg="#38bdf8", fg="#04222e", font=("Segoe UI", 13, "bold"),
                                 relief="flat", activebackground="#22d3ee", cursor="hand2", bd=0)
        self.cap_btn.pack(fill="x", padx=16, pady=(4, 8), ipady=6)

        def sb(parent, text, cmd):
            return tk.Button(parent, text=text, command=cmd, relief="flat", bd=0, bg="#1e293b",
                             fg="#e2e8f0", activebackground="#334155", cursor="hand2")

        r1 = tk.Frame(self.root, bg="#0f172a"); r1.pack(fill="x", padx=16, pady=3)
        sb(r1, "◻ Select region", self.select_region).pack(side="left", expand=True, fill="x", padx=(0, 4), ipady=4)
        sb(r1, "🗂 Save folder…", self.choose_folder).pack(side="left", expand=True, fill="x", padx=(4, 0), ipady=4)

        r2 = tk.Frame(self.root, bg="#0f172a"); r2.pack(fill="x", padx=16, pady=3)
        sb(r2, "🔍 Open review", lambda: webbrowser.open(CAPTURES_URL)).pack(side="left", expand=True, fill="x", padx=(0, 4), ipady=4)
        sb(r2, "🌐 Upload UI", lambda: webbrowser.open(BASE_URL)).pack(side="left", expand=True, fill="x", padx=(4, 0), ipady=4)

        sb(self.root, "📂 Open saves folder", self.open_folder).pack(fill="x", padx=16, pady=3, ipady=4)
        self._refresh_status()

    def _refresh_status(self):
        parts = []
        if self.region:
            x1, y1, x2, y2 = self.region
            parts.append(f"Region {x2 - x1}×{y2 - y1}px")
            self.cap_btn.config(state="normal")
        else:
            parts.append("No region yet — click 'Select region'")
            self.cap_btn.config(state="disabled")
        parts.append(f"Saves → {S.OUTPUT_DIR}")
        self.status.config(text="   •   ".join(parts))

    def choose_folder(self):
        d = filedialog.askdirectory(title="Choose where captures are saved",
                                    initialdir=str(S.OUTPUT_DIR.parent if not S.OUTPUT_DIR.exists() else S.OUTPUT_DIR))
        if d:
            S.OUTPUT_DIR = Path(d)
            S.cfg_set("output_dir", d)
            self._refresh_status()

    def open_folder(self):
        S.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(S.OUTPUT_DIR))  # noqa: S606
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Pico", str(e))

    def select_region(self):
        vx, vy, vw, vh = virtual_screen()
        top = tk.Toplevel(self.root)
        top.overrideredirect(True)
        top.geometry("%dx%d+%d+%d" % (vw, vh, vx, vy))
        top.attributes("-topmost", True)
        try:
            top.attributes("-alpha", 0.30)
        except Exception:
            pass
        cv = tk.Canvas(top, bg="#1e293b", highlightthickness=0, cursor="crosshair")
        cv.pack(fill="both", expand=True)
        cv.create_text(vw // 2, 44, fill="#e2e8f0", font=("Segoe UI", 18, "bold"),
                       text="Drag a box over the microscopy image, then release.   (Esc to cancel)")
        st = {"x": 0, "y": 0, "r": None}

        def down(e):
            st["x"], st["y"] = e.x, e.y

        def move(e):
            if st["r"]:
                cv.delete(st["r"])
            st["r"] = cv.create_rectangle(st["x"], st["y"], e.x, e.y, outline="#38bdf8", width=2)

        def up(e):
            x1, y1 = min(st["x"], e.x) + vx, min(st["y"], e.y) + vy
            x2, y2 = max(st["x"], e.x) + vx, max(st["y"], e.y) + vy
            top.destroy()
            if abs(x2 - x1) > 20 and abs(y2 - y1) > 20:
                self.region = (x1, y1, x2, y2)
                S.cfg_set("region", [x1, y1, x2, y2])
                self._refresh_status()

        cv.bind("<Button-1>", down)
        cv.bind("<B1-Motion>", move)
        cv.bind("<ButtonRelease-1>", up)
        top.bind("<Escape>", lambda e: top.destroy())
        top.focus_force()

    def capture(self):
        if not self.region:
            return
        self.status.config(text="Capturing…")
        self.root.update()
        self.root.withdraw()          # hide so our window isn't in the screenshot
        self.root.update()
        time.sleep(0.18)
        img = None
        try:
            img = ImageGrab.grab(bbox=self.region, all_screens=True)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Pico", f"Screenshot failed:\n{e}")
        finally:
            self.root.deiconify()
        if img is None:
            self._refresh_status()
            return
        try:
            S.add_capture(np.array(img.convert("RGB")))
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Pico", f"Detection failed:\n{e}")
            self._refresh_status()
            return
        self.status.config(text="✔ Captured — see the review tab in your browser.")
        self.root.after(2200, self._refresh_status)

    def run(self):
        self.root.mainloop()


def main():
    threading.Thread(target=start_server, daemon=True).start()
    time.sleep(0.9)
    webbrowser.open(CAPTURES_URL)   # one tab collects every capture
    CaptureApp().run()


if __name__ == "__main__":
    main()
