# Pico-Algae Counter — Desktop UI

A small local web app for counting pico-algae cells in microscopy images.
Colleagues open it in a browser, drop in images, and get per-class counts
(EUK, FE, FC) plus an annotated preview and a CSV export. It runs entirely
on the local PC — no internet connection is needed once set up.

## For everyday users

1. Double-click **`Start Pico-Algae Counter.bat`** (in the project's top folder).
2. A browser tab opens automatically at `http://127.0.0.1:5000`.
3. Drag your microscopy images onto the page (or click to browse) and press **Analyze**.
4. Read the counts, download the CSV, and/or save the annotated previews.
5. When finished, close the small black window to stop the program.

Notes:
- Upload the **brightfield/overlay images** (e.g. `*_og.png`). Files ending in
  `_red` are the paired fluorescence images and are skipped automatically —
  the model only uses the main image.
- You can upload many images at once.
- The **confidence threshold** (default `0.50`) controls how sure the model must
  be before it counts a cell. Raise it for fewer, higher-confidence detections;
  lower it if faint cells are being missed.

## One-time setup (for whoever installs it)

Works with **Python 3.10–3.14** (3.13 recommended — it has a normal Windows
installer at [python.org](https://www.python.org/downloads/); tick *"Add
python.exe to PATH"*). The easiest way is to **double-click `Install.bat`** —
it finds an installed Python automatically and asks you to choose CPU or GPU:

- **CPU only** — works on any PC, no graphics card needed.
- **NVIDIA GPU (CUDA)** — a few times faster, if the PC has an NVIDIA card.
  Pick the CUDA version that matches the installed NVIDIA driver (12.6 is a safe
  default; 12.8 / 13.0 for newer drivers).

The app then uses the GPU automatically whenever a GPU build is installed and a
card is present — otherwise it falls back to CPU. The badge on the web page and
the console line show which one is active. To force a device, set the
`PICO_DEVICE` environment variable to `cpu` or `cuda`.

### Manual install (equivalent to Install.bat)

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
REM CPU build:
.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
REM ...or NVIDIA GPU build (swap cu126 for cu128 / cu130 to match the driver):
.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The launcher uses `.venv\Scripts\python.exe` directly.

## What model it uses

- Checkpoint: `runs/tuning/train/best_train_model.pt` — a 3-channel Faster R-CNN
  (ResNet50-FPN), 4 trained classes (`EUK`, `FE`, `FC`, `colony`).
- The app reads the checkpoint's own tuned settings (anchor sizes, NMS,
  validation confidence threshold `0.50`, and the counted classes `EUK/FE/FC`).
- `colony` detections are drawn on the preview but not included in the totals,
  matching how this checkpoint was validated.

To serve a different checkpoint, change `DEFAULT_CKPT` near the top of
[`app/server.py`](server.py). Only 3-channel checkpoints are supported by this UI.

## Files

- `Install.bat` — interactive installer (choose CPU or GPU).
- `Start Pico-Algae Counter.bat` — double-click launcher.
- `app/server.py` — the Flask web app (UI + routes).
- `app/pico_counter.py` — offline model loading, device selection, inference.
