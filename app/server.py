"""
Pico-Algae Counter - simple local web UI.

Run this, then open the printed http://127.0.0.1:5000 address in a browser.
Colleagues upload microscopy images and get per-class cell counts plus an
annotated preview. Everything runs locally and offline.
"""
from __future__ import annotations

import base64
import csv
import json
import os
import re
import string
import time
import uuid
import webbrowser
from pathlib import Path
from threading import Lock, Timer

import cv2
import numpy as np
from flask import Flask, jsonify, request

from pico_counter import CLASS_COLORS, CLASS_NAMES, PicoCounter, load_image_rgb

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = REPO_ROOT / "runs" / "tuning" / "train" / "best_train_model.pt"

# Detections are sent down to this confidence so the in-browser slider can
# reveal borderline boxes below the chosen display threshold.
PAYLOAD_FLOOR = 0.05

# Where screenshots, annotated images, counts.csv and the training export are
# written. Editable from the web UI and the capture window; persisted to config.
OUTPUT_DIR = Path.home() / "Pico-Algae Captures"
CONFIG_FILE = Path.home() / ".pico_capture.json"


def cfg_load() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text())
    except Exception:
        return {}


def cfg_set(key: str, value) -> None:
    cfg = cfg_load()
    cfg[key] = value
    try:
        CONFIG_FILE.write_text(json.dumps(cfg))
    except Exception as e:  # noqa: BLE001
        print("[pico] could not save config:", e)


# Restore a previously chosen output folder (from the web UI or capture window).
_saved_dir = cfg_load().get("output_dir")
if _saved_dir:
    OUTPUT_DIR = Path(_saved_dir)

# Ordered in-memory list of screen captures for the current session.
_captures: list = []
_captures_lock = Lock()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB total upload

_counter: PicoCounter | None = None
_counter_lock = Lock()


def get_counter() -> PicoCounter:
    global _counter
    with _counter_lock:
        if _counter is None:
            print(f"[pico] Loading model: {DEFAULT_CKPT} ...")
            _counter = PicoCounter(DEFAULT_CKPT)
            print(
                f"[pico] Model ready on {_counter.device_label}. "
                f"score_thresh={_counter.score_thresh} "
                f"counting classes={[CLASS_NAMES[c] for c in _counter.classes_to_count]}"
            )
    return _counter


PAGE_HEAD = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pico-Algae Counter</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --line:#334155; --fg:#e2e8f0; --muted:#94a3b8;
          --accent:#38bdf8; --accent2:#22d3ee; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background:var(--bg); color:var(--fg); }
  header { padding:28px 20px 8px; text-align:center; }
  header h1 { margin:0; font-size:1.7rem; letter-spacing:-0.02em; }
  header p { margin:6px 0 0; color:var(--muted); }
  main { max-width:1080px; margin:0 auto; padding:20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px;
          padding:22px; margin:18px 0; }
  .drop { border:2px dashed var(--line); border-radius:14px; padding:40px 20px; text-align:center;
          transition:.15s; cursor:pointer; }
  .drop.hover { border-color:var(--accent); background:rgba(56,189,248,.06); }
  .drop strong { color:var(--accent); }
  .row { display:flex; gap:16px; flex-wrap:wrap; align-items:center; justify-content:center; }
  label.opt { color:var(--muted); font-size:.9rem; }
  input[type=number] { width:90px; background:#0b1222; color:var(--fg); border:1px solid var(--line);
          border-radius:8px; padding:8px; }
  button { background:linear-gradient(135deg,var(--accent),var(--accent2)); color:#04222e;
           border:0; border-radius:10px; padding:12px 22px; font-weight:700; font-size:1rem; cursor:pointer; }
  button:disabled { opacity:.6; cursor:default; }
  .files { margin-top:12px; color:var(--muted); font-size:.9rem; }
  table { width:100%; border-collapse:collapse; margin-top:6px; }
  th, td { padding:10px 12px; text-align:left; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; font-size:.85rem; text-transform:uppercase; letter-spacing:.03em; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  .total { font-weight:800; color:var(--accent2); }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:16px; }
  .shot img { width:100%; border-radius:10px; border:1px solid var(--line); display:block; }
  .shot h3 { margin:0 0 8px; font-size:1rem; word-break:break-all; }
  .pills { display:flex; gap:8px; flex-wrap:wrap; margin:8px 0 12px; }
  .pill { background:#0b1222; border:1px solid var(--line); border-radius:999px; padding:4px 10px; font-size:.85rem; }
  .note { color:var(--muted); font-size:.9rem; }
  a.dl { color:var(--accent); text-decoration:none; font-weight:600; }
  a.back { color:var(--muted); text-decoration:none; }
  .spinner { display:none; margin-top:14px; color:var(--muted); }
  .badge { display:inline-flex; align-items:center; gap:7px; margin:10px auto 0; padding:5px 12px;
           background:var(--card); border:1px solid var(--line); border-radius:999px;
           color:var(--muted); font-size:.85rem; }
  .badge .dot { width:8px; height:8px; border-radius:50%; }
  .badge.gpu .dot { background:#22c55e; box-shadow:0 0 8px #22c55e; }
  .badge.cpu .dot { background:#94a3b8; }
</style>
</head>
<body>
<header>
  <h1>🦠 Pico-Algae Counter</h1>
  <p>Upload microscopy images &mdash; count cells, then click any result to inspect &amp; correct.</p>
</header>
<main>
"""

PAGE_FOOT = "</main></body></html>"


def device_badge(counter: PicoCounter) -> str:
    is_gpu = counter.device.type == "cuda"
    cls = "gpu" if is_gpu else "cpu"
    label = counter.device_label if is_gpu else "Running on CPU"
    return (
        f'<div style="text-align:center">'
        f'<span class="badge {cls}"><span class="dot"></span>{label}</span></div>'
    )


UPLOAD_JS = r"""
  const drop=document.getElementById('drop'), file=document.getElementById('file'),
        dir=document.getElementById('dir'), list=document.getElementById('filelist'),
        go=document.getElementById('go'), form=document.getElementById('f'),
        spin=document.getElementById('spin');
  const picked=new Map();   // name -> File, accumulates across drops/selections
  const IMG_RE=/\.(png|jpe?g|tiff?|bmp|webp)$/i;
  const esc=s=>String(s).replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));

  function addFiles(files){
    for(const f of files){
      if(f && ((f.type && f.type.startsWith('image/')) || IMG_RE.test(f.name||''))) picked.set(f.name, f);
    }
    sync();
  }
  function sync(){
    const dt=new DataTransfer(); picked.forEach(f=>dt.items.add(f)); file.files=dt.files;
    const n=picked.size; go.disabled=n===0;
    const names=[...picked.keys()].sort((a,b)=>a.toLowerCase().localeCompare(b.toLowerCase()));
    list.innerHTML = n ? (n+' file'+(n>1?'s':'')+' selected &nbsp;<a href="#" id="clr">clear</a>'+
      '<div class="note" style="margin-top:4px">'+names.map(esc).join(', ')+'</div>') : '';
    const clr=document.getElementById('clr');
    if(clr) clr.onclick=e=>{ e.preventDefault(); picked.clear(); sync(); };
  }
  function readEntry(entry, out){
    return new Promise(res=>{
      if(entry.isFile){ entry.file(f=>{ out.push(f); res(); }, ()=>res()); }
      else if(entry.isDirectory){
        const rd=entry.createReader(), all=[];
        const batch=()=>rd.readEntries(ents=>{
          if(!ents.length){ Promise.all(all.map(e=>readEntry(e,out))).then(res); }
          else { all.push(...ents); batch(); }
        }, ()=>res());
        batch();
      } else res();
    });
  }
  async function handleDrop(ev){
    const items=ev.dataTransfer.items, entries=[];
    if(items && items.length && items[0].webkitGetAsEntry){
      for(const it of items){ const en=it.webkitGetAsEntry(); if(en) entries.push(en); }
      const out=[]; await Promise.all(entries.map(e=>readEntry(e,out))); addFiles(out);
    } else { addFiles(ev.dataTransfer.files); }
  }

  drop.addEventListener('click', e=>{ if(e.target.id!=='clr' && e.target.id!=='pickdir') file.click(); });
  file.addEventListener('change', ()=>addFiles(file.files));
  dir.addEventListener('change', ()=>addFiles(dir.files));
  ['dragenter','dragover'].forEach(e=>drop.addEventListener(e, ev=>{ ev.preventDefault(); drop.classList.add('hover'); }));
  ['dragleave','drop'].forEach(e=>drop.addEventListener(e, ev=>{ ev.preventDefault(); drop.classList.remove('hover'); }));
  drop.addEventListener('drop', handleDrop);
  form.addEventListener('submit', ()=>{ go.disabled=true; spin.style.display='block'; });
"""


def upload_page(counter: PicoCounter) -> str:
    thr = counter.score_thresh
    form_html = f"""
<form id="f" class="card" action="/analyze" method="post" enctype="multipart/form-data">
  <div id="drop" class="drop">
    <p style="font-size:1.05rem;margin:0 0 6px">Drop images (or a folder) here, or <strong>click to browse</strong></p>
    <p class="note" style="margin:0">PNG / JPG / TIFF microscopy images. Drop as many as you like, one or many
      at a time — they add up. <a href="#" id="pickdir" onclick="document.getElementById('dir').click();return false;">choose a folder</a></p>
    <input id="file" type="file" name="images" accept="image/*" multiple hidden>
    <input id="dir" type="file" webkitdirectory hidden>
    <div id="filelist" class="files"></div>
  </div>
  <div class="row" style="margin-top:18px">
    <span>
      <label class="opt" for="thr">Confidence threshold</label><br>
      <input id="thr" type="number" name="score_thresh" min="0" max="1" step="0.01" value="{thr:.2f}">
    </span>
    <button id="go" type="submit" disabled>Analyze</button>
  </div>
  <div id="spin" class="spinner">Analyzing… this can take a couple of seconds per image.</div>
</form>"""
    return (PAGE_HEAD + device_badge(counter) + form_html
            + f"<script>{UPLOAD_JS}</script>" + PAGE_FOOT)


def _img_data_uri(rgb) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@app.get("/")
def index():
    return upload_page(get_counter())


@app.post("/analyze")
def analyze():
    counter = get_counter()
    thr_raw = request.form.get("score_thresh", "").strip()
    try:
        thr = float(thr_raw) if thr_raw else counter.score_thresh
    except ValueError:
        thr = counter.score_thresh
    thr = min(max(thr, PAYLOAD_FLOOR), 0.95)

    files = request.files.getlist("images")
    blobs = {}  # stem -> (original filename, bytes)
    for f in files:
        name = f.filename or "unnamed"
        blobs[Path(name).stem] = (name, f.read())
    stems = sorted(blobs, key=str.lower)  # alphabetical order

    consumed = set()  # stems attached as a red pair -> not shown as their own tile
    pairs = {}        # base stem -> red stem
    for stem in stems:
        if stem in consumed:
            continue
        alt = _red_pair_of(stem, blobs)
        if alt and alt not in consumed and alt != stem:
            pairs[stem] = alt
            consumed.add(alt)

    images = []
    skipped = []
    for stem in stems:
        if stem in consumed:
            continue
        name, raw = blobs[stem]
        if stem.lower().endswith("_red"):  # fluorescence with no paired _og
            skipped.append((name, "fluorescence image — no paired _og image found"))
            continue
        try:
            rgb = load_image_rgb(raw)
            pred = counter.predict(rgb, score_thresh=PAYLOAD_FLOOR)
        except Exception as exc:  # noqa: BLE001 - surface decode/inference errors per file
            skipped.append((name, str(exc)))
            continue
        dets = [
            {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "c": lab, "s": sc}
            for b, lab, sc in zip(pred.boxes, pred.labels, pred.scores)
        ]
        img = {"name": name, "src": _img_data_uri(pred.image_rgb),
               "w": pred.width, "h": pred.height, "dets": dets}
        alt = pairs.get(stem)
        if alt:
            try:
                alt_rgb = cv2.resize(load_image_rgb(blobs[alt][1]), (pred.width, pred.height),
                                     interpolation=cv2.INTER_AREA)
                img["alt"] = {"name": blobs[alt][0], "src": _img_data_uri(alt_rgb)}
            except Exception:  # noqa: BLE001 - a bad red pair shouldn't drop the base image
                pass
        images.append(img)

    return render_results(images, skipped, thr, counter)


def _red_pair_of(stem: str, blobs: dict):
    """The paired 'red' image for a base stem: '<x>_og'->'<x>_red', or '...N'->'...N+1'."""
    low = {s.lower(): s for s in blobs}
    if stem.lower().endswith("_og"):
        cand = (stem[:-3] + "_red").lower()
        if cand in low:
            return low[cand]
    m = re.search(r"^(.*?)(\d+)$", stem)  # trailing number +1 (e.g. image_3012 -> image_3013)
    if m:
        prefix, num = m.group(1), m.group(2)
        nxt = (prefix + str(int(num) + 1).zfill(len(num))).lower()
        if nxt in low:
            return low[nxt]
    return None


# --------------------------------------------------------------------------
# Screen-capture workflow: grab -> predict -> review in the browser -> save.
# add_capture() is called in-process by the capture control (capture_app.py).
# --------------------------------------------------------------------------

def _auto_crop(rgb: np.ndarray) -> np.ndarray:
    """Trim flat dark/gray borders left when a capture region overshoots the image.

    Keys on uniformity (a solid window-chrome border has near-zero variance,
    while even dark microscopy has texture) and only trims a limited margin per
    side, so it removes overshoot without eating into image content.
    """
    if rgb.ndim != 3:
        return rgb
    g = rgb.mean(axis=2)
    h, w = g.shape

    def flat_dark(line) -> bool:  # uniform + fairly dark -> window chrome
        return float(line.std()) < 4.0 and float(line.mean()) < 55.0

    max_v, max_h = int(h * 0.20), int(w * 0.20)  # never trim more than 20% off a side
    top, bot, left, right = 0, h, 0, w
    while top < max_v and flat_dark(g[top]):
        top += 1
    while bot > h - max_v and flat_dark(g[bot - 1]):
        bot -= 1
    while left < max_h and flat_dark(g[:, left]):
        left += 1
    while right > w - max_h and flat_dark(g[:, right - 1]):
        right -= 1
    if bot - top < h * 0.5 or right - left < w * 0.5:
        return rgb  # safety
    return np.ascontiguousarray(rgb[top:bot, left:right])


def add_capture(rgb_native: np.ndarray) -> str:
    """Detect on a screenshot and append it to the review session. Returns its id."""
    rgb_native = _auto_crop(rgb_native)
    counter = get_counter()
    pred = counter.predict(rgb_native)
    h, w = rgb_native.shape[:2]
    sx, sy = w / pred.width, h / pred.height  # boxes are in the model's resized space
    dets = [
        {"x1": round(b[0] * sx, 1), "y1": round(b[1] * sy, 1),
         "x2": round(b[2] * sx, 1), "y2": round(b[3] * sy, 1), "c": lab, "s": sc}
        for b, lab, sc in zip(pred.boxes, pred.labels, pred.scores)
    ]
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    rid = f"{ts}_{uuid.uuid4().hex[:4]}"
    image = {"id": rid, "name": f"capture_{ts}", "src": _img_data_uri(rgb_native),
             "w": int(w), "h": int(h), "dets": dets}
    with _captures_lock:
        _captures.append({"id": rid, "ts": ts, "rgb": rgb_native, "image": image, "saved": False})
    return rid


@app.get("/captures")
def captures_page():
    counter = get_counter()
    with _captures_lock:
        imgs = [c["image"] for c in _captures]
    return render_results(imgs, [], counter.score_thresh, counter, mode="capture")


@app.get("/captures/since/<int:n>")
def captures_since(n):
    with _captures_lock:
        imgs = [c["image"] for c in _captures[n:]]
        total = len(_captures)
    return jsonify(images=imgs, total=total)


@app.post("/set_output_dir")
def set_output_dir():
    """Set (and remember) the folder where server-side outputs are written."""
    global OUTPUT_DIR
    data = request.get_json(force=True, silent=True) or {}
    raw = (data.get("path") or "").strip().strip('"')
    if not raw:
        return jsonify(ok=False, error="empty path"), 400
    try:
        path = Path(raw).expanduser()
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:  # noqa: BLE001
        return jsonify(ok=False, error=str(e)), 400
    OUTPUT_DIR = path
    cfg_set("output_dir", str(path))
    return jsonify(ok=True, folder=str(path))


@app.get("/list_dir")
def list_dir():
    """Browse the local filesystem (folders only) so the UI can pick a save folder."""
    raw = request.args.get("path", "")
    if not raw:  # top level: available drives + home shortcut
        drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
        if not drives:
            drives = ["/"]
        return jsonify(path="", parent=None, isRoot=True, home=str(Path.home()),
                       dirs=[{"name": d, "path": d} for d in drives])
    try:
        p = Path(raw).expanduser()
        dirs = []
        for child in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            try:
                if child.is_dir() and not child.name.startswith("."):
                    dirs.append({"name": child.name, "path": str(child)})
            except OSError:
                pass
        parent = str(p.parent) if p.parent != p else None
        return jsonify(path=str(p), parent=parent, isRoot=False, home=str(Path.home()), dirs=dirs)
    except Exception as e:  # noqa: BLE001
        return jsonify(ok=False, error=str(e)), 400


@app.post("/save_csv")
def save_csv():
    """Write the summary counts CSV to the chosen output folder."""
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("csv", "")
    if not text.strip():
        return jsonify(ok=False, error="empty csv"), 400
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    name = _sanitize_name(data.get("filename") or "pico_counts") + ".csv"
    path = _unique_path(OUTPUT_DIR / name)
    path.write_text(text, encoding="utf-8")
    return jsonify(ok=True, folder=str(OUTPUT_DIR), file=path.name)


@app.post("/save/<rid>")
def save_capture(rid):
    with _captures_lock:
        rec = next((c for c in _captures if c["id"] == rid), None)
    if rec is None:
        return jsonify(ok=False, error="capture not found"), 404

    data = request.get_json(force=True, silent=True) or {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = _sanitize_name(data.get("name") or rec["image"]["name"])
    stem = f"{base}_{_next_serial(OUTPUT_DIR, base)}"

    written = []
    if data.get("saveRaw", True):
        p = _unique_path(OUTPUT_DIR / f"{stem}_raw.png")
        cv2.imwrite(str(p), cv2.cvtColor(rec["rgb"], cv2.COLOR_RGB2BGR))
        written.append(p.name)
    ann = data.get("annotated_png", "")
    if data.get("saveAnnotated", True) and "," in ann:
        p = _unique_path(OUTPUT_DIR / f"{stem}_annotated.png")
        p.write_bytes(base64.b64decode(ann.split(",", 1)[1]))
        written.append(p.name)

    _append_counts_csv(OUTPUT_DIR / "counts.csv", rec["ts"], stem, data)
    with _captures_lock:
        rec["saved"] = True
    return jsonify(ok=True, folder=str(OUTPUT_DIR), files=written, stem=stem)


@app.post("/export_labels")
def export_labels():
    """Write reviewed images + boxes as training data: images_og/, images_red/,
    labels/ (raw-id x1 y1 x2 y2, absolute px) and an index.csv for both the og
    and 6-channel trainers."""
    data = request.get_json(force=True, silent=True) or {}
    items = data.get("items") or []
    if not items:
        return jsonify(ok=False, error="nothing to export"), 400

    root = OUTPUT_DIR / "training_export"
    og_dir, lbl_dir, red_dir = root / "images_og", root / "labels", root / "images_red"
    og_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    rows, n_red = [], 0
    for it in items:
        og = _decode_datauri(it.get("og_png", ""))
        if og is None:
            continue
        stem = _unique_export_stem(root, _sanitize_name(it.get("name") or "image"))
        og_path = og_dir / f"{stem}.png"
        cv2.imwrite(str(og_path), og)
        red_path = ""
        red = _decode_datauri(it.get("red_png", "")) if it.get("red_png") else None
        if red is not None:
            red_dir.mkdir(parents=True, exist_ok=True)
            rp = red_dir / f"{stem}.png"
            cv2.imwrite(str(rp), red)
            red_path = str(rp)
            n_red += 1
        (lbl_dir / f"{stem}.txt").write_text((it.get("label") or "").strip() + "\n", encoding="utf-8")
        rows.append([stem, str(og_path), red_path, str(lbl_dir / f"{stem}.txt")])

    idx = root / "index.csv"
    new = not idx.exists()
    with open(idx, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["stem", "og_webp", "red_webp", "label_path"])
        w.writerows(rows)

    return jsonify(ok=True, folder=str(root), count=len(rows), withRed=n_red)


def _decode_datauri(datauri: str):
    if not datauri or "," not in datauri:
        return None
    raw = base64.b64decode(datauri.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def _unique_export_stem(root: Path, base: str) -> str:
    stem, i = base, 2
    while (root / "labels" / f"{stem}.txt").exists() or (root / "images_og" / f"{stem}.png").exists():
        stem, i = f"{base}_{i}", i + 1
    return stem


def _sanitize_name(name: str) -> str:
    name = re.sub(r"\.[^.]+$", "", str(name)).strip()          # drop any extension
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")    # filesystem-safe
    return name or "capture"


def _next_serial(folder: Path, base: str) -> int:
    n = 1
    for f in Path(folder).glob(f"{base}_*"):
        m = re.match(re.escape(base) + r"_(\d+)", f.stem)
        if m:
            n = max(n, int(m.group(1)) + 1)
    return n


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix, i = path.stem, path.suffix, 2
    while True:
        cand = path.with_name(f"{stem}_{i}{suffix}")
        if not cand.exists():
            return cand
        i += 1


def _append_counts_csv(path: Path, ts: str, name: str, data: dict) -> None:
    names = [CLASS_NAMES[c] for c in get_counter().classes_to_count if c in CLASS_NAMES]
    counts = data.get("counts", {}) or {}
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "name"] + names + ["total", "colonies", "unassigned"])
        w.writerow(
            [ts, name]
            + [counts.get(n, 0) for n in names]
            + [data.get("total", 0), data.get("colonies", 0), data.get("unassigned", 0)]
        )


def render_results(images, skipped, thr, counter, mode="upload") -> str:
    if not images and mode != "capture":
        body = (
            '<div class="card"><p>No images were analyzed.</p></div>'
            + _skipped_html(skipped)
            + '<p style="text-align:center;margin:24px">'
            '<a class="back" href="/">← Analyze more images</a></p>'
        )
        return PAGE_HEAD + device_badge(counter) + body + PAGE_FOOT

    meta = {
        "countedClasses": counter.classes_to_count,
        "classNames": {str(k): v for k, v in CLASS_NAMES.items()},
        "classColors": {str(k): f"rgb({r},{g},{b})" for k, (r, g, b) in CLASS_COLORS.items()},
        "thr": thr,
        "floor": PAYLOAD_FLOOR,
        "mode": mode,
        "outputDir": str(OUTPUT_DIR),
    }
    if mode == "capture":
        meta["pollUrl"] = "/captures/since/"
        meta["saveBase"] = "/save/"

    payload = {
        "images": images,
        "meta": meta,
        "skipped": [{"name": n, "reason": r} for n, r in skipped],
    }
    # Guard against any "</script>" or "<" inside strings breaking the page.
    data_json = json.dumps(payload).replace("<", "\\u003c")

    footer_link = (
        '<a class="back" href="/">← Back to home</a>'
        if mode == "capture"
        else '<a class="back" href="/">← Analyze more images</a>'
    )
    return (
        PAGE_HEAD
        + device_badge(counter)
        + RESULTS_CSS
        + '<div id="app"></div>'
        + f'<p style="text-align:center;margin:24px">{footer_link}</p>'
        + f"<script>window.PICO = {data_json};</script>"
        + f"<script>{RESULTS_JS}</script>"
        + PAGE_FOOT
    )


def _skipped_html(skipped) -> str:
    if not skipped:
        return ""
    items = "".join(
        f"<tr><td>{n}</td><td class='note'>{r}</td></tr>" for n, r in skipped
    )
    return (
        f'<div class="card"><h3 style="margin-top:0">Skipped ({len(skipped)})</h3>'
        f"<table><tbody>{items}</tbody></table></div>"
    )


RESULTS_CSS = """
<style>
  .toolbar { display:flex; gap:12px; flex-wrap:wrap; align-items:center; justify-content:space-between; }
  .legend { display:flex; gap:14px; flex-wrap:wrap; align-items:center; }
  .legend .lg { display:inline-flex; align-items:center; gap:6px; font-size:.85rem; color:var(--muted); }
  .sw { width:12px; height:12px; border-radius:3px; display:inline-block; }
  .cards2 { display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:16px; }
  .imgcard { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px; }
  .imgcard h3 { margin:0 0 8px; font-size:.95rem; word-break:break-all; }
  .frame { position:relative; border-radius:10px; overflow:hidden; border:1px solid var(--line);
           cursor:zoom-in; background:#000; line-height:0; }
  .frame img { display:block; width:100%; height:auto; }
  .frame svg { position:absolute; inset:0; width:100%; height:100%; }
  .frame .hint { position:absolute; right:8px; bottom:8px; background:rgba(0,0,0,.6); color:#fff;
                 font-size:.72rem; padding:3px 8px; border-radius:999px; pointer-events:none; line-height:1.2; }
  .pills2 { display:flex; gap:6px; flex-wrap:wrap; margin:10px 0 2px; }
  .pill .dot { width:8px; height:8px; border-radius:50%; display:inline-block; }

  .modal { position:fixed; inset:0; background:rgba(2,6,23,.94); z-index:50; display:flex; flex-direction:column; }
  .modal.hidden { display:none; }
  .mbar, .mfoot { display:flex; gap:10px; align-items:center; flex-wrap:wrap; padding:9px 14px;
                  background:var(--card); border-bottom:1px solid var(--line); }
  .mfoot { border-top:1px solid var(--line); border-bottom:0; }
  .mbar .title { font-weight:700; max-width:32vw; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .mstage { flex:1; overflow:hidden; background:#000; position:relative; min-height:0; }
  #esvg { width:100%; height:100%; display:block; touch-action:none; cursor:crosshair; }
  .clsbtn { border:1px solid var(--line); background:#0b1222; color:var(--fg); border-radius:8px;
            padding:6px 11px; font-weight:700; font-size:.85rem; cursor:pointer; display:inline-flex; gap:6px; align-items:center; }
  .clsbtn.active { outline:2px solid var(--accent); outline-offset:1px; }
  .mbtn { border:1px solid var(--line); background:#0b1222; color:var(--fg); border-radius:8px;
          padding:6px 11px; cursor:pointer; font-size:.9rem; }
  .mbtn.on { background:linear-gradient(135deg,var(--accent),var(--accent2)); color:#04222e; border:0; font-weight:700; }
  .mbtn.danger { color:#fca5a5; }
  #erects .det { fill:none; pointer-events:none; }   /* hit-testing is geometric (boxAt) */
  #erects .det.sel { fill:rgba(56,189,248,.18); }
  #etemp { fill:rgba(56,189,248,.18); stroke:var(--accent); }
  .count-chip { display:inline-flex; align-items:center; gap:6px; font-variant-numeric:tabular-nums; font-size:.9rem; }
  .help { color:var(--muted); font-size:.8rem; }
  input[type=range] { accent-color:var(--accent); vertical-align:middle; }
  .warn { background:rgba(251,191,36,.12); border:1px solid rgba(251,191,36,.4); color:#fbbf24;
          border-radius:10px; padding:8px 12px; margin:4px 0 12px; font-size:.9rem; }
  .colonypanel { position:absolute; left:50%; bottom:16px; transform:translateX(-50%); z-index:6;
          display:flex; gap:10px; align-items:center; background:var(--card); border:1px solid var(--line);
          border-radius:12px; padding:9px 14px; box-shadow:0 8px 28px rgba(0,0,0,.45); flex-wrap:wrap; }
  .colonypanel.hidden { display:none; }
  .colonypanel input[type=number] { width:72px; text-align:center; font-size:1.1rem; background:#0b1222;
          color:var(--fg); border:1px solid var(--line); border-radius:8px; padding:6px; }
  .stepbtn { border:1px solid var(--line); background:#0b1222; color:var(--fg); border-radius:8px;
          width:34px; height:34px; font-size:1.2rem; line-height:1; cursor:pointer; }
  .nameinput { background:#0b1222; color:var(--fg); border:1px solid var(--line); border-radius:8px;
          padding:6px 9px; font-size:.95rem; font-weight:700; min-width:140px; max-width:24vw; }
  .clsbtn .num { display:inline-flex; width:16px; height:16px; align-items:center; justify-content:center;
          background:#04222e; color:#7dd3fc; border-radius:4px; font-size:.72rem; font-weight:800; }
  .chk { display:inline-flex; gap:5px; align-items:center; color:var(--muted); font-size:.82rem; cursor:pointer; }
  .redbadge { display:inline-block; background:rgba(236,72,153,.15); border:1px solid rgba(236,72,153,.5);
          color:#f472b6; border-radius:999px; padding:1px 9px; font-size:.72rem; font-weight:600; vertical-align:middle; }
  .folderbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:2px 0 12px; }
  .folderbar .nameinput { font-weight:400; font-family:ui-monospace,Consolas,monospace; font-size:.85rem; }
  .handle { fill:#fff; stroke:#0284c7; stroke-width:1.5px; }
  .h-nw,.h-se { cursor:nwse-resize; } .h-ne,.h-sw { cursor:nesw-resize; }
  .h-n,.h-s { cursor:ns-resize; } .h-e,.h-w { cursor:ew-resize; }
  .browse { position:fixed; inset:0; background:rgba(2,6,23,.6); z-index:60; display:flex; align-items:center; justify-content:center; }
  .browse.hidden { display:none; }
  .browse-panel { background:var(--card); border:1px solid var(--line); border-radius:14px;
          width:min(660px,92vw); max-height:82vh; display:flex; flex-direction:column; overflow:hidden;
          box-shadow:0 20px 60px rgba(0,0,0,.5); }
  .browse-head { padding:12px 16px; border-bottom:1px solid var(--line); display:flex; gap:10px; align-items:center; }
  .browse-path { flex:1; font-family:ui-monospace,Consolas,monospace; font-size:.85rem; color:var(--fg); word-break:break-all; }
  .browse-list { overflow:auto; padding:6px; }
  .browse-item { padding:8px 12px; border-radius:8px; cursor:pointer; display:flex; gap:8px; align-items:center; font-size:.92rem; }
  .browse-item:hover { background:#0b1222; }
  .browse-foot { padding:10px 16px; border-top:1px solid var(--line); display:flex; gap:8px; justify-content:flex-end; align-items:center; }
</style>
"""


RESULTS_JS = r"""
(function(){
  const PICO = window.PICO, M = PICO.meta;
  const NAMES = M.classNames, COLORS = M.classColors;
  const COUNTED = M.countedClasses.map(Number);
  const ALL = Object.keys(NAMES).map(Number).sort();
  const CAPTURE = M.mode === 'capture';   // screen-capture session: save to disk + poll for new
  let uid = 1;

  function normImg(img){
    img.dets = (img.dets || []).map(d => ({
      id: uid++, cls: Number(d.c), score: d.s,
      x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2, added: (d.s === null),
      colClass: d.colClass, colCount: d.colCount
    }));
    if(img.thr == null) img.thr = M.thr;
    img.showAlt = false;              // showing the paired red image?
    return img;
  }
  PICO.images.forEach(normImg);

  const esc = s => String(s).replace(/[&<>"']/g,
    m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
  const clamp = (v,a,b) => Math.max(a, Math.min(b, v));

  function shown(img){ return img.dets.filter(d => d.added || d.score >= img.thr); }
  function counts(img){
    const cls={}; ALL.forEach(k=>cls[k]=0);
    let colonies=0, needReview=0;
    shown(img).forEach(d=>{
      if(d.cls===4){
        colonies++;
        if(d.colClass && d.colCount>0) cls[d.colClass]+=d.colCount;   // colony cells fold into their class
        else needReview++;                                            // flagged but not yet assigned
      } else { cls[d.cls]++; }
    });
    return { cls, colonies, needReview };
  }
  function total(img){ const c=counts(img).cls; return COUNTED.reduce((a,k)=>a+(c[k]||0),0); }

  const app = document.getElementById('app');

  const UNASSIGNED_COL = 'rgb(236,72,153)';  // magenta = colony that still needs a count
  function colonyMarkup(d){
    const assigned = d.colClass && d.colCount>0;
    if(!assigned) return '';   // unassigned: box only, so the cells stay visible to count
    const h=d.y2-d.y1;
    const fs=Math.max(12, Math.min(h*0.28, 40));
    const x=d.x1+fs*0.3, y=d.y1+fs*0.3;   // count in the top-left corner, out of the way
    return `<text x="${x}" y="${y}" font-size="${fs}" text-anchor="start" dominant-baseline="hanging" `+
      `style="paint-order:stroke;stroke:#000;stroke-width:${fs*0.16}px;fill:${COLORS[d.colClass]};font-weight:800;pointer-events:none">${d.colCount}</text>`;
  }
  function detRect(d, opts){
    opts = opts || {};
    const idAttr = opts.id ? ` data-id="${d.id}"` : '';
    const clsAttr = opts.cls ? ` class="${opts.cls}"` : '';
    if(d.cls===4){
      const assigned = d.colClass && d.colCount>0;
      const stroke = assigned ? COLORS[4] : UNASSIGNED_COL;
      const dash = assigned ? '' : ' stroke-dasharray="10 6"';
      const sw = opts.sw || 3;
      return `<rect${clsAttr}${idAttr} x="${d.x1}" y="${d.y1}" width="${d.x2-d.x1}" height="${d.y2-d.y1}" `+
        `fill="none" stroke="${stroke}" stroke-width="${sw}"${dash} vector-effect="non-scaling-stroke"/>` + colonyMarkup(d);
    }
    const sw = opts.sw || 2;
    return `<rect${clsAttr}${idAttr} x="${d.x1}" y="${d.y1}" width="${d.x2-d.x1}" height="${d.y2-d.y1}" `+
      `fill="none" stroke="${COLORS[d.cls]}" stroke-width="${sw}" vector-effect="non-scaling-stroke"/>`;
  }
  function rectsSvg(img){ return shown(img).map(d => detRect(d)).join(''); }

  function renderApp(){
    const head = COUNTED.map(c=>`<th class="num">${esc(NAMES[c])}</th>`).join('');
    let rows=''; const tot={}; COUNTED.forEach(c=>tot[c]=0); let grand=0, colTot=0, reviewTot=0;
    PICO.images.forEach(img=>{
      const r=counts(img), t=total(img);
      const colCell = `${r.colonies}` + (r.needReview ? ` <span style="color:#fbbf24">(${r.needReview}?)</span>` : '');
      rows += `<tr><td>${esc(img.name)}</td>`+COUNTED.map(k=>`<td class="num">${r.cls[k]||0}</td>`).join('')
            + `<td class="num">${colCell}</td><td class="num total">${t}</td></tr>`;
      COUNTED.forEach(k=>tot[k]+=(r.cls[k]||0)); grand+=t; colTot+=r.colonies; reviewTot+=r.needReview;
    });
    const foot = COUNTED.map(k=>`<td class="num">${tot[k]}</td>`).join('');
    const legend = ALL.map(c=>`<span class="lg"><span class="sw" style="background:${COLORS[c]}"></span>${esc(NAMES[c])}`
      + (c===4?' (assign count + class)':'') + `</span>`).join('');
    const warn = reviewTot>0
      ? `<div class="warn">⚠ ${reviewTot} colon${reviewTot===1?'y':'ies'} still ${reviewTot===1?'needs':'need'} a count &mdash; open the image and assign each flagged (?) cluster.</div>`
      : '';

    const n = PICO.images.length;
    const intro = CAPTURE
      ? 'New screenshots appear here automatically. Click one to inspect and correct, then <b>Save to disk</b>.'
      : 'Click any image to enlarge, inspect and correct. Counts and the CSV update automatically.';
    const tableHtml = n ? `
      <table>
        <thead><tr><th>Image</th>${head}<th class="num">Colonies</th><th class="num">Total</th></tr></thead>
        <tbody>${rows}</tbody>
        <tfoot><tr><td><strong>All images</strong></td>${foot}<td class="num">${colTot}</td><td class="num total">${grand}</td></tr></tfoot>
      </table>` : '<p class="note">Waiting for the first capture… take a screenshot from the Pico Capture window.</p>';

    let html = `
    <div class="card">
      <div class="toolbar">
        <h2 style="margin:0">${CAPTURE ? 'Captures' : 'Results'} &mdash; ${n} image${n!==1?'s':''}</h2>
        <span style="display:flex;gap:8px;flex-wrap:wrap">
          ${n ? '<button class="mbtn" id="expBtn" title="save the corrected images + boxes as training data">⬇ Export for training</button>' : ''}
          ${n ? '<button class="mbtn on" id="csvBtn">⬇ Save counts CSV</button>' : ''}
        </span>
      </div>
      <p class="note">${intro}</p>
      <div class="legend" style="margin:6px 0 10px">${legend}</div>
      <div class="folderbar">
        <span class="help">📁 Save folder (screenshots, counts.csv, training export):</span>
        <input class="nameinput" id="outdir" style="flex:1;min-width:240px;max-width:none" value="${esc(M.outputDir||'')}">
        <button class="mbtn" id="browsedir">📂 Browse…</button>
        <button class="mbtn" id="setdir">Set</button>
      </div>
      ${warn}
      ${tableHtml}
    </div>`;

    let cards='';
    PICO.images.forEach((img,i)=>{
      const r=counts(img);
      const pills = COUNTED.map(k=>`<span class="pill"><span class="dot" style="background:${COLORS[k]}"></span> ${esc(NAMES[k])}: <strong>${r.cls[k]||0}</strong></span>`).join('')
        + `<span class="pill total">Total: ${total(img)}</span>`
        + (r.colonies?`<span class="pill"><span class="dot" style="background:${COLORS[4]}"></span> colonies: <strong>${r.colonies}</strong></span>`:'')
        + (r.needReview?`<span class="pill" style="color:#fbbf24">⚠ ${r.needReview} to count</span>`:'');
      const redBadge = img.alt ? ` <span class="redbadge">⇄ red (q)</span>` : '';
      cards += `
      <div class="imgcard">
        <h3>${esc(img.name)}${redBadge}</h3>
        <div class="frame" data-i="${i}">
          <img src="${img.src}" alt="">
          <svg viewBox="0 0 ${img.w} ${img.h}" preserveAspectRatio="none">${rectsSvg(img)}</svg>
          <span class="hint">🔍 click to inspect / edit</span>
        </div>
        <div class="pills2">${pills}</div>
      </div>`;
    });
    html += `<div class="cards2">${cards}</div>`;
    html += renderSkipped();
    app.innerHTML = html;

    const csvBtn = document.getElementById('csvBtn');
    if(csvBtn) csvBtn.onclick = saveCsv;
    const expBtn = document.getElementById('expBtn');
    if(expBtn) expBtn.onclick = exportTraining;
    const outdir=document.getElementById('outdir'), setdir=document.getElementById('setdir'),
          browsedir=document.getElementById('browsedir');
    if(setdir){
      setdir.onclick=()=>applyOutputDir(outdir.value);
      outdir.addEventListener('keydown', e=>{ if(e.key==='Enter'){ e.preventDefault(); applyOutputDir(outdir.value); } });
    }
    if(browsedir) browsedir.onclick=openBrowse;
    app.querySelectorAll('.frame').forEach(f =>
      f.addEventListener('click', () => openEditor(+f.dataset.i)));
  }

  function renderSkipped(){
    const s = PICO.skipped || [];
    if(!s.length) return '';
    return `<div class="card"><h3 style="margin-top:0">Skipped (${s.length})</h3><table><tbody>`
      + s.map(x=>`<tr><td>${esc(x.name)}</td><td class="note">${esc(x.reason)}</td></tr>`).join('')
      + `</tbody></table></div>`;
  }

  // ---------- CSV (reflects live corrections) ----------
  function csvCell(v){ v=String(v); return /[",\n]/.test(v) ? '"'+v.replace(/"/g,'""')+'"' : v; }
  function buildCsvText(){
    const rows=[['image'].concat(COUNTED.map(c=>NAMES[c]),['colonies','total'])];
    const tot={}; COUNTED.forEach(k=>tot[k]=0); let grand=0, colTot=0;
    PICO.images.forEach(img=>{
      const r=counts(img), t=total(img);
      rows.push([img.name].concat(COUNTED.map(k=>r.cls[k]||0),[r.colonies, t]));
      COUNTED.forEach(k=>tot[k]+=(r.cls[k]||0)); grand+=t; colTot+=r.colonies;
    });
    rows.push(['All images'].concat(COUNTED.map(k=>tot[k]),[colTot, grand]));
    return rows.map(r=>r.map(csvCell).join(',')).join('\r\n');
  }
  function saveCsv(){
    if(!PICO.images.length) return;
    const btn=document.getElementById('csvBtn'); if(btn){ btn.disabled=true; btn.textContent='Saving…'; }
    fetch('/save_csv',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({csv:buildCsvText(), filename:'pico_counts'})})
      .then(r=>r.json()).then(res=>{
        if(btn){ btn.disabled=false; btn.textContent='⬇ Save counts CSV'; }
        if(res.ok) toast('✔ Saved '+res.file+' → '+res.folder, true);
        else toast('Could not save CSV: '+(res.error||'?'), false);
      }).catch(e=>{ if(btn){ btn.disabled=false; btn.textContent='⬇ Save counts CSV'; } toast('Could not save CSV: '+e, false); });
  }
  function dl(blob, filename){
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a'); a.href=url; a.download=filename; a.click();
    setTimeout(()=>URL.revokeObjectURL(url), 500);
  }

  // ---------- output folder: set + browse the local tree ----------
  function applyOutputDir(p){
    p=(p||'').trim(); if(!p) return;
    fetch('/set_output_dir',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:p})})
      .then(r=>r.json()).then(res=>{
        if(res.ok){ M.outputDir=res.folder; const o=document.getElementById('outdir'); if(o) o.value=res.folder;
          toast('✔ Saving to '+res.folder, true); }
        else toast('Could not set folder: '+(res.error||'?'), false);
      }).catch(e=>toast('Could not set folder: '+e, false));
  }
  let BR=null, brCur='';
  function ensureBrowse(){
    if(BR) return;
    const m=document.createElement('div'); m.className='browse hidden';
    m.innerHTML = `
      <div class="browse-panel">
        <div class="browse-head">
          <button class="mbtn" id="brUp">↑ Up</button>
          <span class="browse-path" id="brPath"></span>
        </div>
        <div class="browse-list" id="brList"></div>
        <div class="browse-foot">
          <span class="help" style="margin-right:auto">Pick the folder where outputs are saved.</span>
          <button class="mbtn" id="brCancel">Cancel</button>
          <button class="mbtn on" id="brSelect">Select this folder</button>
        </div>
      </div>`;
    document.body.appendChild(m);
    BR={ m, up:m.querySelector('#brUp'), path:m.querySelector('#brPath'), list:m.querySelector('#brList'),
         select:m.querySelector('#brSelect'), parent:null };
    m.querySelector('#brCancel').onclick=()=>m.classList.add('hidden');
    m.addEventListener('mousedown', e=>{ if(e.target===m) m.classList.add('hidden'); });
    BR.up.onclick=()=>browseTo(BR.parent||'');
    BR.select.onclick=()=>{ if(brCur){ applyOutputDir(brCur); } m.classList.add('hidden'); };
  }
  function openBrowse(){
    ensureBrowse();
    const o=document.getElementById('outdir');
    browseTo((o && o.value.trim()) || (M.outputDir||''));
    BR.m.classList.remove('hidden');
  }
  function browseTo(path){
    fetch('/list_dir?path='+encodeURIComponent(path||''))
      .then(r=>r.json()).then(res=>{
        if(res && res.error){ toast('Cannot open: '+res.error, false); return; }
        brCur = res.path||'';
        BR.parent = res.parent;
        BR.path.textContent = res.path || 'This PC';
        BR.up.disabled = !res.path;
        BR.select.disabled = !res.path;   // the drive list itself isn't selectable
        BR.list.innerHTML = (res.dirs && res.dirs.length)
          ? res.dirs.map(d=>`<div class="browse-item" data-path="${esc(d.path)}">📁 ${esc(d.name)}</div>`).join('')
          : '<div class="help" style="padding:10px 12px">(no subfolders)</div>';
        BR.list.querySelectorAll('.browse-item').forEach(it=> it.onclick=()=>browseTo(it.dataset.path));
      }).catch(e=>toast('Cannot browse: '+e, false));
  }

  // ---------- export corrected boxes as training data (images + labels + index) ----------
  function labelText(img){
    return shown(img).map(d=>{
      const raw=(d.cls|0)-1;   // EUK1->0, FE2->1, FC3->2, colony4->3
      const x1=Math.max(0,Math.round(Math.min(d.x1,d.x2))), y1=Math.max(0,Math.round(Math.min(d.y1,d.y2)));
      const x2=Math.min(img.w,Math.round(Math.max(d.x1,d.x2))), y2=Math.min(img.h,Math.round(Math.max(d.y1,d.y2)));
      return raw+' '+x1+' '+y1+' '+x2+' '+y2;
    }).join('\n');
  }
  function exportTraining(){
    if(!PICO.images.length) return;
    const btn=document.getElementById('expBtn'); if(btn){ btn.disabled=true; btn.textContent='Exporting…'; }
    const items=PICO.images.map(img=>({ name:img.name, label:labelText(img),
      og_png:img.src, red_png: img.alt?img.alt.src:null }));
    fetch('/export_labels',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({items})})
      .then(r=>r.json()).then(res=>{
        if(btn){ btn.disabled=false; btn.textContent='⬇ Export for training'; }
        if(res.ok) toast('✔ Exported '+res.count+' image'+(res.count!==1?'s':'')+
          (res.withRed?(' ('+res.withRed+' with red pair)'):'')+' → '+res.folder, true);
        else toast('Export failed: '+(res.error||'?'), false);
      }).catch(e=>{ if(btn){ btn.disabled=false; btn.textContent='⬇ Export for training'; } toast('Export failed: '+e, false); });
  }

  // ---------- Editor (zoom / pan / add / delete / reclassify) ----------
  let ED=null;      // dom refs
  let idx=0, scale=1, tx=0, ty=0, activeCls=COUNTED[0]||1;   // left=draw, right=inspect
  let sel=new Set();                 // selected detection ids (multi-select)
  let panning=false, drawing=false, panLast=null, drawStart=null;
  let movePress=null;                // {id, additive, moved, last, sx, sy} while dragging box(es)
  let resizePress=null;              // {d, h} while dragging a resize handle

  function curImg(){ return PICO.images[idx]; }

  function buildEditor(){
    const m=document.createElement('div'); m.className='modal hidden'; m.id='editor';
    m.innerHTML = `
      <div class="mbar">
        <input class="nameinput" id="ename" spellcheck="false" title="image name (editable)">
        <span class="title" id="etitle"></span>
        <span class="help">left-drag = draw · right = inspect</span>
        <button class="mbtn danger" id="emDel">🗑 Delete</button>
        <span class="help">class</span><span id="eclasses"></span>
        <span style="margin-left:auto;display:flex;gap:8px;align-items:center">
          <button class="mbtn" id="eAlt" style="display:none">q ⇄ og/red</button>
          <button class="mbtn" id="ePrev">◀</button>
          <button class="mbtn" id="eNext">▶</button>
          <button class="mbtn" id="eFit">Fit</button>
          ${CAPTURE ? '<button class="mbtn on" id="eSave">💾 Save to disk</button>' : ''}
          <button class="mbtn" id="eClose">✕ ${CAPTURE ? 'Close' : 'Done'}</button>
        </span>
      </div>
      <div class="mstage">
        <svg id="esvg"><g id="evp">
          <image id="eimg" x="0" y="0"></image>
          <g id="erects"></g>
          <g id="ehandles"></g>
          <rect id="etemp" style="display:none" vector-effect="non-scaling-stroke" stroke-width="2"></rect>
        </g></svg>
        <div id="ecolony" class="colonypanel hidden">
          <span class="help">🧩 Colony &mdash; cells inside:</span>
          <button class="stepbtn" id="ecolMinus" title="one fewer">−</button>
          <input type="number" id="ecolcount" min="0" step="1" value="0">
          <button class="stepbtn" id="ecolPlus" title="one more">+</button>
          <span class="help">class:</span>
          <span id="ecolclasses"></span>
        </div>
      </div>
      <div class="mfoot">
        <span id="ecounts"></span>
        <span style="margin-left:auto;display:flex;gap:10px;align-items:center">
          ${CAPTURE ? '<label class="chk"><input type="checkbox" id="esaveRaw" checked> screenshot</label>'
                    + '<label class="chk"><input type="checkbox" id="esaveAnn" checked> annotated</label>' : ''}
          <span class="help">show model boxes ≥</span>
          <input type="range" id="ethr" min="${M.floor}" max="0.95" step="0.01">
          <span id="ethrval" class="help"></span>
          <button class="mbtn" id="eDlImg">⬇ Save image</button>
        </span>
      </div>
      <div class="mbar" style="border-top:1px solid var(--line);border-bottom:0">
        <span class="help"><b>left-drag = draw box</b> · <b>right-drag = pan</b> · <b>right-click box = select</b> ·
          right-drag box = move · drag <b>handles = resize</b> · <b>Shift/Ctrl + right-click</b> = multi-select ·
          wheel = zoom · keys <b>1-${ALL.length}</b> reclassify · <b>q</b> = og/red · <b>Del</b> removes · <b>←/→</b> images · <b>Esc</b> closes</span>
      </div>`;
    document.body.appendChild(m);
    const $=id=>m.querySelector(id);
    ED={ m, svg:$('#esvg'), vp:$('#evp'), img:$('#eimg'), rects:$('#erects'), handles:$('#ehandles'), temp:$('#etemp'),
         title:$('#etitle'), name:$('#ename'), classes:$('#eclasses'), ecounts:$('#ecounts'),
         thr:$('#ethr'), thrval:$('#ethrval'), alt:$('#eAlt'),
         colony:$('#ecolony'), colcount:$('#ecolcount'), colclasses:$('#ecolclasses'),
         saveRaw:$('#esaveRaw'), saveAnn:$('#esaveAnn') };

    ED.classes.innerHTML = ALL.map((c,i)=>
      `<button class="clsbtn" data-c="${c}" title="key ${i+1}"><span class="num">${i+1}</span>`+
      `<span class="sw" style="background:${COLORS[c]}"></span>${esc(NAMES[c])}</button>`
    ).join('');
    ED.classes.querySelectorAll('.clsbtn').forEach(b=> b.onclick=()=>setClass(+b.dataset.c));

    // colony assignment panel: cell count + which counted class those cells are
    ED.colclasses.innerHTML = COUNTED.map(c=>
      `<button class="clsbtn" data-cc="${c}"><span class="sw" style="background:${COLORS[c]}"></span>${esc(NAMES[c])}</button>`
    ).join('');
    ED.colclasses.querySelectorAll('.clsbtn').forEach(b=> b.onclick=()=>setColClass(+b.dataset.cc));
    ED.colcount.addEventListener('input', ()=>setColCount(parseInt(ED.colcount.value,10)||0));
    $('#ecolMinus').onclick=()=>{ const d=selOne(); if(d) setColCount((d.colCount||0)-1); };
    $('#ecolPlus').onclick =()=>{ const d=selOne(); if(d) setColCount((d.colCount||0)+1); };

    ED.name.addEventListener('input', ()=>{ curImg().name=ED.name.value; });
    ED.name.addEventListener('change', ()=>{ const v=ED.name.value.trim(); if(v) curImg().name=v; renderApp(); });
    ED.alt.onclick=toggleAlt;

    $('#emDel').onclick=deleteSel;
    $('#ePrev').onclick=()=>nav(-1);
    $('#eNext').onclick=()=>nav(1);
    $('#eFit').onclick=fit;
    $('#eClose').onclick=closeEditor;
    $('#eDlImg').onclick=saveImage;
    const saveBtn=$('#eSave'); if(saveBtn) saveBtn.onclick=saveReview;
    ED.thr.addEventListener('input', ()=>{ curImg().thr=parseFloat(ED.thr.value); sel.clear(); renderStage(); renderApp(); });

    ED.svg.addEventListener('mousedown', onDown);
    ED.svg.addEventListener('contextmenu', e=>e.preventDefault());   // right-click = inspect, no menu
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    ED.svg.addEventListener('wheel', onWheel, {passive:false});
  }

  function openEditor(i){
    if(!ED) buildEditor();
    idx=i; sel.clear(); setClass(activeCls);
    ED.m.classList.remove('hidden');
    document.addEventListener('keydown', onKey);
    loadImage();
  }
  function closeEditor(){
    ED.m.classList.add('hidden');
    document.removeEventListener('keydown', onKey);
    renderApp();
  }

  function loadImage(){
    const img=curImg();
    ED.svg.setAttribute('viewBox', `0 0 ${img.w} ${img.h}`);
    ED.img.setAttribute('width', img.w);
    ED.img.setAttribute('height', img.h);
    ED.thr.value=img.thr;
    ED.name.value=img.name;
    updateBg(); updateAltBtn();
    fit();               // resets scale/translate + renders
  }

  function updateBg(){ const img=curImg(); ED.img.setAttribute('href', (img.showAlt && img.alt) ? img.alt.src : img.src); }
  function updateAltBtn(){
    const img=curImg();
    ED.alt.style.display = img.alt ? '' : 'none';
    ED.alt.classList.toggle('on', !!img.showAlt);
    ED.alt.textContent = img.showAlt ? 'q ⇄ RED' : 'q ⇄ og/red';
  }
  function toggleAlt(){ const img=curImg(); if(!img.alt) return; img.showAlt=!img.showAlt; updateBg(); updateAltBtn(); }

  function fit(){ scale=1; tx=0; ty=0; applyVP(); renderStage(); }
  function applyVP(){ ED.vp.setAttribute('transform', `translate(${tx} ${ty}) scale(${scale})`); }

  function renderStage(){
    ED.title.textContent = `(${idx+1}/${PICO.images.length})`;
    ED.thrval.textContent = (+curImg().thr).toFixed(2);
    renderRects();
    renderHandles();
    renderCounts();
    updateColonyPanel();
  }
  function renderRects(){
    const img=curImg();
    ED.rects.innerHTML = shown(img).map(d=>{
      const on = sel.has(d.id);
      const sw = d.cls===4 ? (on?5:3) : (on?4:2);
      return detRect(d, {cls:'det'+(on?' sel':''), id:true, sw});
    }).join('');
    // box mousedown is handled in onDown via e.target so drag-to-move works
  }
  function renderHandles(){
    const d = selOne();   // resize handles on the single selected box
    if(!d){ ED.handles.innerHTML=''; return; }
    let hs=10; try{ const a=ED.vp.getScreenCTM().a; if(a) hs=10/a; }catch(e){}
    const x1=Math.min(d.x1,d.x2), y1=Math.min(d.y1,d.y2), x2=Math.max(d.x1,d.x2), y2=Math.max(d.y1,d.y2);
    const cx=(x1+x2)/2, cy=(y1+y2)/2;
    const P=[['nw',x1,y1],['n',cx,y1],['ne',x2,y1],['e',x2,cy],['se',x2,y2],['s',cx,y2],['sw',x1,y2],['w',x1,cy]];
    ED.handles.innerHTML = P.map(([h,px,py])=>
      `<rect class="handle h-${h}" data-h="${h}" x="${px-hs/2}" y="${py-hs/2}" width="${hs}" height="${hs}" vector-effect="non-scaling-stroke"/>`
    ).join('');
  }
  function renderCounts(){
    const r=counts(curImg());
    let h = COUNTED.map(k=>
      `<span class="count-chip"><span class="sw" style="background:${COLORS[k]}"></span>${esc(NAMES[k])}: <b>${r.cls[k]||0}</b></span>`
    ).join(' &nbsp; ');
    h += ` &nbsp; <span class="count-chip"><span class="sw" style="background:${COLORS[4]}"></span>colonies: <b>${r.colonies}</b></span>`;
    if(r.needReview>0) h += ` &nbsp; <span class="count-chip" style="color:#fbbf24">⚠ ${r.needReview} unassigned</span>`;
    h += ` &nbsp; <span class="count-chip">Total: <b style="color:var(--accent2)">${total(curImg())}</b></span>`;
    if(sel.size>1) h += ` &nbsp; <span class="count-chip" style="color:var(--accent)">${sel.size} selected</span>`;
    ED.ecounts.innerHTML = h;
  }
  function updateColonyPanel(){
    const d=selOne();
    if(d && d.cls===4){
      ED.colony.classList.remove('hidden');
      if(document.activeElement!==ED.colcount) ED.colcount.value = d.colCount||0;
      ED.colclasses.querySelectorAll('.clsbtn').forEach(b=>b.classList.toggle('active', +b.dataset.cc===d.colClass));
    } else {
      ED.colony.classList.add('hidden');
    }
  }

  function setClass(c){
    activeCls=c;
    ED.classes.querySelectorAll('.clsbtn').forEach(b=>b.classList.toggle('active', +b.dataset.c===c));
    const ds=selDets();
    if(ds.length){
      ds.forEach(d=>{
        d.cls=c;
        if(c===4){ if(d.colCount==null) d.colCount=0; if(d.colClass==null) d.colClass=null; }
        else { d.colCount=undefined; d.colClass=undefined; }
      });
      renderStage(); renderApp();
    }
  }
  function setColCount(v){
    const d=selOne(); if(!d||d.cls!==4) return;
    d.colCount=Math.max(0, v|0); renderStage(); renderApp();
  }
  function setColClass(c){
    const d=selOne(); if(!d||d.cls!==4) return;
    d.colClass=c; if(!(d.colCount>0)) d.colCount=1;   // picking a class implies at least one cell
    renderStage(); renderApp();
  }
  function selById(id){ return curImg().dets.find(d=>d.id===id); }
  function selDets(){ return curImg().dets.filter(d=>sel.has(d.id)); }
  function selOne(){ const a=selDets(); return a.length===1 ? a[0] : null; }
  function boxAt(p){   // topmost/smallest box under the point, incl. a small edge tolerance
    let tol=5; try{ const a=ED.vp.getScreenCTM().a; if(a) tol=5/a; }catch(e){}
    let best=null, bestArea=Infinity;
    shown(curImg()).forEach(d=>{
      const x1=Math.min(d.x1,d.x2)-tol, y1=Math.min(d.y1,d.y2)-tol,
            x2=Math.max(d.x1,d.x2)+tol, y2=Math.max(d.y1,d.y2)+tol;
      if(p.x>=x1 && p.x<=x2 && p.y>=y1 && p.y<=y2){
        const a=(x2-x1)*(y2-y1); if(a<bestArea){ bestArea=a; best=d; }
      }
    });
    return best;
  }
  function deleteSel(){
    if(!sel.size) return;
    curImg().dets = curImg().dets.filter(d=>!sel.has(d.id)); sel.clear(); renderStage(); renderApp();
  }

  // coordinate helpers
  function toImg(e){ const p=ED.svg.createSVGPoint(); p.x=e.clientX; p.y=e.clientY;
    const q=p.matrixTransform(ED.vp.getScreenCTM().inverse()); return {x:q.x,y:q.y}; }
  function toVB(e){ const p=ED.svg.createSVGPoint(); p.x=e.clientX; p.y=e.clientY;
    return p.matrixTransform(ED.svg.getScreenCTM().inverse()); }

  function onDown(e){
    if(e.button===2){                              // RIGHT button = inspect (select / move / pan)
      e.preventDefault();
      const additive = e.shiftKey||e.ctrlKey||e.metaKey;
      const box = boxAt(toImg(e));                 // geometry hit-test: edges + near-edge count too
      if(box){
        if(!additive && !sel.has(box.id)){ sel=new Set([box.id]); }
        movePress={id:box.id, additive, moved:false, last:toImg(e), sx:e.clientX, sy:e.clientY};
        renderStage();
      } else {
        if(!additive && sel.size){ sel.clear(); renderStage(); }   // right-click empty -> deselect
        panning=true; panLast=toVB(e);
      }
      return;
    }
    if(e.button!==0) return;                       // left button below
    const t=e.target, hAttr=(t && t.getAttribute) ? t.getAttribute('data-h') : null;
    if(hAttr){                                     // LEFT on a resize handle = resize
      const d=selOne(); if(d){ e.preventDefault(); resizePress={d, h:hAttr}; }
      return;
    }
    const p=toImg(e);                              // LEFT elsewhere = draw a new box
    if(p.x<0 || p.y<0 || p.x>curImg().w || p.y>curImg().h) return;   // not outside the image
    e.preventDefault();
    drawing=true; drawStart={x:p.x, y:p.y};
    ED.temp.style.display=''; ED.temp.setAttribute('stroke', COLORS[activeCls]);
    setTemp(p.x, p.y, 0, 0);
  }
  function onMove(e){
    if(drawing){
      const p=toImg(e), cx=clamp(p.x,0,curImg().w), cy=clamp(p.y,0,curImg().h);   // stay inside image
      setTemp(Math.min(cx,drawStart.x), Math.min(cy,drawStart.y),
              Math.abs(cx-drawStart.x), Math.abs(cy-drawStart.y));
      return;
    }
    if(resizePress){
      const p=toImg(e), d=resizePress.d, h=resizePress.h;
      if(h.indexOf('e')>=0) d.x2=Math.max(p.x, d.x1+3);
      if(h.indexOf('w')>=0) d.x1=Math.min(p.x, d.x2-3);
      if(h.indexOf('n')>=0) d.y1=Math.min(p.y, d.y2-3);
      if(h.indexOf('s')>=0) d.y2=Math.max(p.y, d.y1+3);
      renderRects(); renderHandles();
      return;
    }
    if(movePress){
      if(!movePress.moved){
        if(Math.hypot(e.clientX-movePress.sx, e.clientY-movePress.sy) < 4) return;  // click vs drag
        movePress.moved=true;
        if(movePress.additive && !sel.has(movePress.id)) sel.add(movePress.id);
      }
      const p=toImg(e), dx=p.x-movePress.last.x, dy=p.y-movePress.last.y; movePress.last=p;
      curImg().dets.forEach(d=>{ if(sel.has(d.id)){ d.x1+=dx; d.y1+=dy; d.x2+=dx; d.y2+=dy; }});
      renderRects(); renderHandles();
      return;
    }
    if(panning){ const c=toVB(e); tx+=(c.x-panLast.x); ty+=(c.y-panLast.y); panLast=c; applyVP(); }
  }
  function onUp(e){
    if(resizePress){
      const d=resizePress.d;
      d.x1=round(d.x1); d.y1=round(d.y1); d.x2=round(d.x2); d.y2=round(d.y2);
      resizePress=null; renderStage(); renderApp();
      return;
    }
    if(drawing){
      drawing=false; ED.temp.style.display='none';
      const r=tempRect();
      if(r.w>4 && r.h>4){
        const d={id:uid++, cls:activeCls, score:null, added:true,
                 x1:round(r.x), y1:round(r.y), x2:round(r.x+r.w), y2:round(r.y+r.h)};
        if(activeCls===4){ d.colCount=0; d.colClass=null; }   // new colony starts unassigned
        curImg().dets.push(d); sel=new Set([d.id]); renderStage(); renderApp();
      }
    }
    if(movePress){
      if(!movePress.moved){                       // it was a click, not a drag
        const id=movePress.id;
        if(movePress.additive){ if(sel.has(id)) sel.delete(id); else sel.add(id); }
        else { sel=new Set([id]); }
      } else {                                     // finished moving: snap coords
        curImg().dets.forEach(d=>{ if(sel.has(d.id)){ d.x1=round(d.x1); d.y1=round(d.y1); d.x2=round(d.x2); d.y2=round(d.y2); }});
        renderApp();
      }
      movePress=null; renderStage();
    }
    panning=false;
  }
  function onWheel(e){
    e.preventDefault();
    const before=toImg(e), vb=toVB(e);
    scale=clamp(scale*(e.deltaY<0?1.15:1/1.15), 0.4, 16);
    tx=vb.x-before.x*scale; ty=vb.y-before.y*scale; applyVP();
    renderHandles();   // keep handle size constant on zoom
  }
  function round(v){ return Math.round(v*10)/10; }
  function setTemp(x,y,w,h){ ED.temp.setAttribute('x',x); ED.temp.setAttribute('y',y);
    ED.temp.setAttribute('width',w); ED.temp.setAttribute('height',h); ED.temp._r={x,y,w,h}; }
  function tempRect(){ return ED.temp._r || {x:0,y:0,w:0,h:0}; }

  function nav(delta){
    if(PICO.images.length<2 && delta) return;
    idx=(idx+delta+PICO.images.length)%PICO.images.length;
    sel.clear(); loadImage();
  }
  function onKey(e){
    const typing = document.activeElement && document.activeElement.tagName==='INPUT';
    if(e.key==='Escape'){ if(typing){ document.activeElement.blur(); return; } closeEditor(); return; }
    if(typing) return;   // don't fire shortcuts while editing the name / colony count
    if(e.key==='Delete'||e.key==='Backspace'){ e.preventDefault(); deleteSel(); return; }
    if(e.key==='ArrowLeft'){ nav(-1); return; }
    if(e.key==='ArrowRight'){ nav(1); return; }
    if(e.key==='q'||e.key==='Q'){ toggleAlt(); return; }
    if(e.key==='f'||e.key==='F'){ fit(); return; }
    const n=parseInt(e.key,10);
    if(n>=1 && n<=ALL.length){ setClass(ALL[n-1]); }
  }

  // ---------- annotated image (canvas), shared by download and disk-save ----------
  function annotatedCanvas(img, cb){
    const cv=document.createElement('canvas'); cv.width=img.w; cv.height=img.h;
    const ctx=cv.getContext('2d'); const im=new Image();
    im.onload=()=>{
      ctx.drawImage(im,0,0);
      const baseLw=Math.max(2, img.w/900);
      shown(img).forEach(d=>{
        if(d.cls===4){
          const assigned=d.colClass && d.colCount>0;
          ctx.lineWidth=baseLw; ctx.strokeStyle=assigned?COLORS[4]:UNASSIGNED_COL;
          ctx.strokeRect(d.x1,d.y1,d.x2-d.x1,d.y2-d.y1);
          if(assigned){   // count in the top-left corner, cells left visible
            const h=d.y2-d.y1, fs=Math.max(12, Math.min(h*0.28, 40)), txt=String(d.colCount);
            const tx=d.x1+fs*0.3, ty=d.y1+fs*0.3;
            ctx.textAlign='left'; ctx.textBaseline='top'; ctx.font=`800 ${fs}px sans-serif`;
            ctx.lineWidth=Math.max(2, fs*0.16); ctx.strokeStyle='#000'; ctx.strokeText(txt,tx,ty);
            ctx.fillStyle=COLORS[d.colClass]; ctx.fillText(txt,tx,ty);
          }
        } else {
          ctx.lineWidth=baseLw; ctx.strokeStyle=COLORS[d.cls];
          ctx.strokeRect(d.x1,d.y1,d.x2-d.x1,d.y2-d.y1);
        }
      });
      cb(cv);
    };
    im.src=img.src;
  }
  function saveImage(){
    const img=curImg();
    annotatedCanvas(img, cv=>cv.toBlob(b=>dl(b, img.name.replace(/\.[^.]+$/,'')+'_checked.png')));
  }

  function toast(msg, good){
    let t=document.getElementById('ptoast');
    if(!t){ t=document.createElement('div'); t.id='ptoast';
      t.style.cssText='position:fixed;left:50%;top:18px;transform:translateX(-50%);z-index:100;'+
        'padding:12px 20px;border-radius:10px;font-weight:700;box-shadow:0 8px 28px rgba(0,0,0,.5);max-width:80vw;text-align:center';
      document.body.appendChild(t); }
    t.style.background=good?'#16a34a':'#b91c1c'; t.style.color='#fff'; t.textContent=msg;
    t.style.display='block'; clearTimeout(t._h); t._h=setTimeout(()=>{t.style.display='none';}, 5000);
  }

  // ---------- capture session: save to disk (with options) + poll for new captures ----------
  function saveReview(){
    const img=curImg(), r=counts(img);
    if(r.needReview>0 && !confirm(r.needReview+' colony/colonies still have no count. Save anyway?')) return;
    const wantRaw = !ED.saveRaw || ED.saveRaw.checked;
    const wantAnn = !ED.saveAnn || ED.saveAnn.checked;
    const btn=ED.m.querySelector('#eSave'); if(btn){ btn.disabled=true; btn.textContent='Saving…'; }
    const finish = (annUri)=>{
      const body={ name: img.name, annotated_png: annUri, saveRaw: wantRaw, saveAnnotated: wantAnn,
                   total: total(img), colonies: r.colonies, unassigned: r.needReview, counts:{} };
      COUNTED.forEach(k=>body.counts[NAMES[k]]=r.cls[k]||0);
      fetch(M.saveBase+img.id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
        .then(res=>res.json()).then(res=>{
          if(btn){ btn.disabled=false; btn.textContent='💾 Save to disk'; }
          if(res.ok) toast('✔ Saved '+((res.files&&res.files.length)?res.files.join(', '):res.stem)+' → '+res.folder, true);
          else toast('Save failed: '+(res.error||'unknown'), false);
        }).catch(e=>{ if(btn){ btn.disabled=false; btn.textContent='💾 Save to disk'; } toast('Save failed: '+e, false); });
    };
    if(wantAnn) annotatedCanvas(img, cv=>finish(cv.toDataURL('image/png')));
    else finish('');
  }

  renderApp();
  if(CAPTURE){
    if(PICO.images.length) openEditor(PICO.images.length-1);   // open the latest capture
    let known = PICO.images.length;
    setInterval(()=>{
      fetch(M.pollUrl+known).then(r=>r.json()).then(res=>{
        if(!res || !res.images || !res.images.length) return;
        res.images.forEach(im=>{ normImg(im); PICO.images.push(im); });
        known = res.total;
        const editorOpen = ED && !ED.m.classList.contains('hidden');
        renderApp();
        if(!editorOpen) openEditor(PICO.images.length-1);     // jump in only if not mid-edit
        else toast('📸 New capture added — now '+PICO.images.length, true);
      }).catch(()=>{});
    }, 1500);
  }
})();
"""


def _open_browser(url: str) -> None:
    Timer(1.2, lambda: webbrowser.open(url)).start()


def main() -> None:
    host, port = "127.0.0.1", 5000
    get_counter()  # load model up front so first request is fast
    url = f"http://{host}:{port}"
    print(f"\n  Pico-Algae Counter is running at:  {url}")
    print("  (Close this window to stop the program.)\n")
    _open_browser(url)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
