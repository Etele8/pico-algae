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
- **Drop images or a folder** — you can drag them in one or a few at a time and
  they **add up** (nothing is lost), or use *choose a folder*. Images are listed
  and analyzed in **alphabetical order**.
- **Red pairs**: if you include both an image and its partner (either `*_og`/`*_red`,
  or the next consecutive number, e.g. `image_3012` + `image_3013`), they're paired
  automatically — only the base image gets a tile, and inside the editor **`q`**
  toggles the background between the two (boxes stay put) for easy validation.
- The **confidence threshold** (default `0.50`) controls how sure the model must
  be before it counts a cell. Raise it for fewer detections; lower it for more.
- **Save folder**: the box at the top of the results sets where server-side outputs go
  (**Save counts CSV**, the training export, and captures). Type a path or click **📂 Browse…**
  to pick one from the local file tree; it's remembered between runs.

### In the editor (click any image)

- **Left-drag = draw a box** (only inside the image). **Right-drag empty = pan**, **wheel = zoom**, **Fit** resets.
- **Right-click a box to select** it (edges and near-edges count too); **Shift/Ctrl + right-click** for several;
  **right-drag** a box to move it; **drag its white handles** to resize.
- **1–4** reclassify the selection (buttons show their number); **Del** removes.
- Select a **colony** to type how many cells it holds and their class.
- **`q`** swaps between the og and red image. The **name** (top-left) is editable.

### Export corrected boxes as training data

**⬇ Export for training** (top of the results) writes every reviewed image plus
your corrected boxes into a **`training_export`** folder inside your output folder:
`images_og/`, `images_red/` (for paired images), `labels/` (one `.txt` per image,
`raw_class x1 y1 x2 y2` in pixels) and an `index.csv`. That folder feeds both the
og and the 6-channel trainers directly — so pre-labelling with the model and
correcting here turns validation into training data. Merge it with last year's
dataset and retrain to include new images.

## Capture straight off the screen (Olympus workflow)

To skip exporting files, double-click **`Capture from Olympus.bat`**. A small
always-on-top **Pico Capture** window appears alongside the web app:

1. Click **Select region** once and drag a box over the microscopy image area in
   the Olympus software (remembered for next time). Overshoot is fine — the
   dark/gray border is auto-cropped away.
2. Click **📸 Capture & Review** — it screenshots that region and runs the model.
   All captures collect in **one review tab** (no new tab per shot); a new capture
   opens automatically unless you're mid-edit.
3. Correct the detections, optionally rename it, then click **💾 Save to disk**.

Choose where output goes with **🗂 Save folder…** (capture window) or the **Save
folder** box at the top of the results page — it works in the upload flow too and
is remembered between runs. Each save writes the raw screenshot and/or annotated
image (**checkboxes** let you pick) plus a row in a running `counts.csv`, named
`<yourname>_<serial>`. The screenshot reads the
*displayed* image, so detection can be slightly less precise than the original
file — the correction step covers the difference.

## One-time setup (for whoever installs it)

**Easiest — the Setup.exe wizard.** Run `Pico-Algae-Counter-Setup.exe` and click
through the normal install wizard (it installs per-user, so no admin rights are
needed and it avoids the write-protected `C:\Program Files`). It creates Start
Menu and Desktop shortcuts and an entry in *Add or remove programs*. Everything
below happens automatically during the wizard.

**Or the zip** — double-click **`Install.bat`**. Either way the setup:

- **installs Python for you** if none is present (downloads the official
  Python 3.13 and adds it to PATH). Python 3.10–3.14 all work.
- **auto-detects the hardware** — it reads the NVIDIA driver via `nvidia-smi`
  and installs the matching CUDA build of PyTorch (cu126 / cu128 / cu130), or
  the CPU build if there's no NVIDIA card. Nobody has to know their CUDA version.

The detection logic lives in [`tools/detect_cuda.py`](../tools/detect_cuda.py) and
the whole flow in [`tools/setup_env.ps1`](../tools/setup_env.ps1); re-running
`Install.bat` (or *Repair setup* in the Start Menu) repeats it safely. The badge
on the web page and the console line show whether CPU or GPU is active. To force
a device, set the `PICO_DEVICE` environment variable to `cpu` or `cuda`.

### Building the installer (for the maintainer)

Needs [Inno Setup 6](https://jrsoftware.org/isdl.php)
(`winget install JRSoftware.InnoSetup`). From the repo root, after pushing your
changes to GitHub `main`:

```powershell
powershell -ExecutionPolicy Bypass -File installer\build.ps1
```

It stamps `app/VERSION.txt` from the current commit (the baseline for the update
check — so build from a **pushed** commit) and compiles
[`installer/pico.iss`](../installer/pico.iss) into `Pico-Algae-Counter-Setup.exe`.

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

## Updating (remote, no reinstall)

To ship a new version to everyone: commit and push the code to
`https://github.com/Etele8/pico-algae` (main). On each colleague's PC, the web
page then shows a green **"A new version is available"** bar with an **Update
now** button (it checks GitHub via `/check_update` on load and compares the
pushed commit against `app/VERSION.txt`). Clicking it launches `Update.bat`;
they can also double-click **`Update.bat`** directly. It downloads the latest
code from GitHub and overwrites only the program files. It **does not** touch the `.venv`, the model
(`runs/…`), saved captures, or the user's settings, so updates are small and
fast even though the model is large. Any new Python packages in
`requirements.txt` are installed automatically (PyTorch is left as-is).

`Update.bat` writes the installed commit id and date to `app/VERSION.txt`, shown
at the end of the update. If there's no internet the current version keeps
working. The model itself is *not* in the repo, so retraining a model is
delivered by replacing `runs/tuning/train/best_train_model.pt` (share the new
file directly), not through `Update.bat`.

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

- `Pico-Algae-Counter-Setup.exe` — the wizard installer (built from `installer/`).
- `Install.bat` — auto installer (installs Python + auto-detects CPU/GPU).
- `Update.bat` — pulls the latest code from GitHub (keeps model/venv/data).
- `tools/detect_cuda.py`, `tools/setup_env.ps1` — auto Python + CUDA setup.
- `Start Pico-Algae Counter.bat` — double-click launcher.
- `app/server.py` — the Flask web app (UI + routes).
- `app/pico_counter.py` — offline model loading, device selection, inference.
