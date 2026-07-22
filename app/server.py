"""
Pico-Algae Counter - simple local web UI.

Run this, then open the printed http://127.0.0.1:5000 address in a browser.
Colleagues upload microscopy images and get per-class cell counts plus an
annotated preview. Everything runs locally and offline.
"""
from __future__ import annotations

import base64
import json
import webbrowser
from pathlib import Path
from threading import Timer

import cv2
from flask import Flask, request

from pico_counter import CLASS_COLORS, CLASS_NAMES, PicoCounter, load_image_rgb

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = REPO_ROOT / "runs" / "tuning" / "train" / "best_train_model.pt"

# Detections are sent down to this confidence so the in-browser slider can
# reveal borderline boxes below the chosen display threshold.
PAYLOAD_FLOOR = 0.05

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB total upload

_counter: PicoCounter | None = None


def get_counter() -> PicoCounter:
    global _counter
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


def upload_page(counter: PicoCounter) -> str:
    thr = counter.score_thresh
    return PAGE_HEAD + device_badge(counter) + f"""
<form id="f" class="card" action="/analyze" method="post" enctype="multipart/form-data">
  <div id="drop" class="drop">
    <p style="font-size:1.05rem;margin:0 0 6px">Drop images here or <strong>click to browse</strong></p>
    <p class="note" style="margin:0">PNG / JPG / TIFF microscopy images. You can select many at once.</p>
    <input id="file" type="file" name="images" accept="image/*" multiple hidden>
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
</form>
<script>
  const drop = document.getElementById('drop');
  const file = document.getElementById('file');
  const list = document.getElementById('filelist');
  const go = document.getElementById('go');
  const form = document.getElementById('f');
  const spin = document.getElementById('spin');
  function refresh() {{
    const n = file.files.length;
    go.disabled = n === 0;
    list.textContent = n ? (n + ' file' + (n>1?'s':'') + ' selected') : '';
  }}
  drop.addEventListener('click', () => file.click());
  file.addEventListener('change', refresh);
  ['dragenter','dragover'].forEach(e => drop.addEventListener(e, ev => {{
    ev.preventDefault(); drop.classList.add('hover'); }}));
  ['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => {{
    ev.preventDefault(); drop.classList.remove('hover'); }}));
  drop.addEventListener('drop', ev => {{ file.files = ev.dataTransfer.files; refresh(); }});
  form.addEventListener('submit', () => {{ go.disabled = true; spin.style.display = 'block'; }});
</script>
""" + PAGE_FOOT


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

    images = []       # per-image payload dicts for the browser editor
    skipped = []      # (name, reason)
    for f in files:
        name = f.filename or "unnamed"
        stem = Path(name).stem
        if stem.lower().endswith("_red"):
            skipped.append((name, "paired fluorescence image — not the model input"))
            continue
        try:
            rgb = load_image_rgb(f.read())
            pred = counter.predict(rgb, score_thresh=PAYLOAD_FLOOR)
        except Exception as exc:  # noqa: BLE001 - surface any decode/inference error per file
            skipped.append((name, str(exc)))
            continue
        dets = [
            {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "c": lab, "s": sc}
            for b, lab, sc in zip(pred.boxes, pred.labels, pred.scores)
        ]
        images.append(
            {"name": name, "src": _img_data_uri(pred.image_rgb),
             "w": pred.width, "h": pred.height, "dets": dets}
        )

    return render_results(images, skipped, thr, counter)


def render_results(images, skipped, thr, counter) -> str:
    if not images:
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
    }
    payload = {
        "images": images,
        "meta": meta,
        "skipped": [{"name": n, "reason": r} for n, r in skipped],
    }
    # Guard against any "</script>" or "<" inside strings breaking the page.
    data_json = json.dumps(payload).replace("<", "\\u003c")

    return (
        PAGE_HEAD
        + device_badge(counter)
        + RESULTS_CSS
        + '<div id="app"></div>'
        + '<p style="text-align:center;margin:24px">'
          '<a class="back" href="/">← Analyze more images</a></p>'
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
  #esvg { width:100%; height:100%; display:block; touch-action:none; cursor:grab; }
  #esvg.adding { cursor:crosshair; }
  .clsbtn { border:1px solid var(--line); background:#0b1222; color:var(--fg); border-radius:8px;
            padding:6px 11px; font-weight:700; font-size:.85rem; cursor:pointer; display:inline-flex; gap:6px; align-items:center; }
  .clsbtn.active { outline:2px solid var(--accent); outline-offset:1px; }
  .mbtn { border:1px solid var(--line); background:#0b1222; color:var(--fg); border-radius:8px;
          padding:6px 11px; cursor:pointer; font-size:.9rem; }
  .mbtn.on { background:linear-gradient(135deg,var(--accent),var(--accent2)); color:#04222e; border:0; font-weight:700; }
  .mbtn.danger { color:#fca5a5; }
  #erects .det { fill:transparent; pointer-events:all; }
  #esvg.adding #erects .det { pointer-events:none; }
  .det.sel { fill:rgba(56,189,248,.18); }
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
</style>
"""


RESULTS_JS = r"""
(function(){
  const PICO = window.PICO, M = PICO.meta;
  const NAMES = M.classNames, COLORS = M.classColors;
  const COUNTED = M.countedClasses.map(Number);
  const ALL = Object.keys(NAMES).map(Number).sort();
  let uid = 1;

  PICO.images.forEach(img => {
    img.dets = img.dets.map(d => ({
      id: uid++, cls: Number(d.c), score: d.s,
      x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2, added: (d.s === null)
    }));
    img.thr = M.thr;
  });

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

    let html = `
    <div class="card">
      <div class="toolbar">
        <h2 style="margin:0">Results &mdash; ${PICO.images.length} image${PICO.images.length!==1?'s':''}</h2>
        <button class="mbtn on" id="csvBtn">⬇ Download CSV</button>
      </div>
      <p class="note">Click any image to enlarge, inspect and correct. Counts and the CSV update automatically.</p>
      <div class="legend" style="margin:6px 0 12px">${legend}</div>
      ${warn}
      <table>
        <thead><tr><th>Image</th>${head}<th class="num">Colonies</th><th class="num">Total</th></tr></thead>
        <tbody>${rows}</tbody>
        <tfoot><tr><td><strong>All images</strong></td>${foot}<td class="num">${colTot}</td><td class="num total">${grand}</td></tr></tfoot>
      </table>
    </div>`;

    let cards='';
    PICO.images.forEach((img,i)=>{
      const r=counts(img);
      const pills = COUNTED.map(k=>`<span class="pill"><span class="dot" style="background:${COLORS[k]}"></span> ${esc(NAMES[k])}: <strong>${r.cls[k]||0}</strong></span>`).join('')
        + `<span class="pill total">Total: ${total(img)}</span>`
        + (r.colonies?`<span class="pill"><span class="dot" style="background:${COLORS[4]}"></span> colonies: <strong>${r.colonies}</strong></span>`:'')
        + (r.needReview?`<span class="pill" style="color:#fbbf24">⚠ ${r.needReview} to count</span>`:'');
      cards += `
      <div class="imgcard">
        <h3>${esc(img.name)}</h3>
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

    document.getElementById('csvBtn').onclick = downloadCSV;
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
  function downloadCSV(){
    const rows=[['image'].concat(COUNTED.map(c=>NAMES[c]),['colonies','total'])];
    const tot={}; COUNTED.forEach(k=>tot[k]=0); let grand=0, colTot=0;
    PICO.images.forEach(img=>{
      const r=counts(img), t=total(img);
      rows.push([img.name].concat(COUNTED.map(k=>r.cls[k]||0),[r.colonies, t]));
      COUNTED.forEach(k=>tot[k]+=(r.cls[k]||0)); grand+=t; colTot+=r.colonies;
    });
    rows.push(['All images'].concat(COUNTED.map(k=>tot[k]),[colTot, grand]));
    const csv = rows.map(r=>r.map(csvCell).join(',')).join('\r\n');
    dl(new Blob([csv],{type:'text/csv'}), 'pico_counts.csv');
  }
  function dl(blob, filename){
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a'); a.href=url; a.download=filename; a.click();
    setTimeout(()=>URL.revokeObjectURL(url), 500);
  }

  // ---------- Editor (zoom / pan / add / delete / reclassify) ----------
  let ED=null;      // dom refs
  let idx=0, scale=1, tx=0, ty=0, mode='inspect', activeCls=COUNTED[0]||1, selId=null;
  let panning=false, drawing=false, panLast=null, drawStart=null;

  function curImg(){ return PICO.images[idx]; }

  function buildEditor(){
    const m=document.createElement('div'); m.className='modal hidden'; m.id='editor';
    m.innerHTML = `
      <div class="mbar">
        <span class="title" id="etitle"></span>
        <button class="mbtn" id="emInspect">✋ Inspect / Pan</button>
        <button class="mbtn" id="emAdd">✏️ Add box</button>
        <button class="mbtn danger" id="emDel">🗑 Delete</button>
        <span class="help">class</span><span id="eclasses"></span>
        <span style="margin-left:auto;display:flex;gap:8px;align-items:center">
          <button class="mbtn" id="ePrev">◀ Prev</button>
          <button class="mbtn" id="eNext">Next ▶</button>
          <button class="mbtn" id="eFit">Fit</button>
          <button class="mbtn on" id="eClose">✕ Done</button>
        </span>
      </div>
      <div class="mstage">
        <svg id="esvg"><g id="evp">
          <image id="eimg" x="0" y="0"></image>
          <g id="erects"></g>
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
        <span style="margin-left:auto;display:flex;gap:8px;align-items:center">
          <span class="help">show model boxes ≥</span>
          <input type="range" id="ethr" min="${M.floor}" max="0.95" step="0.01">
          <span id="ethrval" class="help"></span>
          <button class="mbtn" id="eDlImg">⬇ Save image</button>
        </span>
      </div>
      <div class="mbar" style="border-top:1px solid var(--line);border-bottom:0">
        <span class="help">Wheel = zoom · drag = pan · <b>Add box</b>: drag on a cell · click a box to select ·
          keys <b>1-${ALL.length}</b> set class · <b>Del</b> removes · <b>←/→</b> images · <b>Esc</b> closes ·
          select a <b>colony</b> to enter its cell count &amp; class</span>
      </div>`;
    document.body.appendChild(m);
    const $=id=>m.querySelector(id);
    ED={ m, svg:$('#esvg'), vp:$('#evp'), img:$('#eimg'), rects:$('#erects'), temp:$('#etemp'),
         title:$('#etitle'), classes:$('#eclasses'), ecounts:$('#ecounts'),
         thr:$('#ethr'), thrval:$('#ethrval'),
         colony:$('#ecolony'), colcount:$('#ecolcount'), colclasses:$('#ecolclasses') };

    ED.classes.innerHTML = ALL.map(c=>
      `<button class="clsbtn" data-c="${c}"><span class="sw" style="background:${COLORS[c]}"></span>${esc(NAMES[c])}</button>`
    ).join('');
    ED.classes.querySelectorAll('.clsbtn').forEach(b=>
      b.onclick=()=>setClass(+b.dataset.c));

    // colony assignment panel: cell count + which counted class those cells are
    ED.colclasses.innerHTML = COUNTED.map(c=>
      `<button class="clsbtn" data-cc="${c}"><span class="sw" style="background:${COLORS[c]}"></span>${esc(NAMES[c])}</button>`
    ).join('');
    ED.colclasses.querySelectorAll('.clsbtn').forEach(b=> b.onclick=()=>setColClass(+b.dataset.cc));
    ED.colcount.addEventListener('input', ()=>setColCount(parseInt(ED.colcount.value,10)||0));
    $('#ecolMinus').onclick=()=>{ const d=selById(selId); if(d) setColCount((d.colCount||0)-1); };
    $('#ecolPlus').onclick =()=>{ const d=selById(selId); if(d) setColCount((d.colCount||0)+1); };

    $('#emInspect').onclick=()=>setMode('inspect');
    $('#emAdd').onclick=()=>setMode('add');
    $('#emDel').onclick=deleteSel;
    $('#ePrev').onclick=()=>nav(-1);
    $('#eNext').onclick=()=>nav(1);
    $('#eFit').onclick=fit;
    $('#eClose').onclick=closeEditor;
    $('#eDlImg').onclick=saveImage;
    ED.thr.addEventListener('input', ()=>{ curImg().thr=parseFloat(ED.thr.value); selId=null; renderStage(); renderApp(); });

    ED.svg.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    ED.svg.addEventListener('wheel', onWheel, {passive:false});
  }

  function openEditor(i){
    if(!ED) buildEditor();
    idx=i; selId=null; setMode('inspect'); setClass(activeCls);
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
    ED.img.setAttribute('href', img.src);
    ED.img.setAttribute('width', img.w);
    ED.img.setAttribute('height', img.h);
    ED.thr.value=img.thr;
    fit();               // resets scale/translate + renders
  }

  function fit(){ scale=1; tx=0; ty=0; applyVP(); renderStage(); }
  function applyVP(){ ED.vp.setAttribute('transform', `translate(${tx} ${ty}) scale(${scale})`); }

  function renderStage(){
    const img=curImg();
    ED.title.textContent = `${img.name}   (${idx+1}/${PICO.images.length})`;
    ED.thrval.textContent = (+img.thr).toFixed(2);
    renderRects();
    renderCounts();
    updateColonyPanel();
  }
  function renderRects(){
    const img=curImg();
    ED.rects.innerHTML = shown(img).map(d=>{
      const sel = d.id===selId;
      const sw = d.cls===4 ? (sel?4:3) : (sel?3:2);
      return detRect(d, {cls:'det'+(sel?' sel':''), id:true, sw});
    }).join('');
    ED.rects.querySelectorAll('.det').forEach(r=>
      r.addEventListener('mousedown', e=>{ if(mode==='inspect'){ e.stopPropagation(); selId=+r.dataset.id; renderStage(); } }));
  }
  function renderCounts(){
    const r=counts(curImg());
    let h = COUNTED.map(k=>
      `<span class="count-chip"><span class="sw" style="background:${COLORS[k]}"></span>${esc(NAMES[k])}: <b>${r.cls[k]||0}</b></span>`
    ).join(' &nbsp; ');
    h += ` &nbsp; <span class="count-chip"><span class="sw" style="background:${COLORS[4]}"></span>colonies: <b>${r.colonies}</b></span>`;
    if(r.needReview>0) h += ` &nbsp; <span class="count-chip" style="color:#fbbf24">⚠ ${r.needReview} unassigned</span>`;
    h += ` &nbsp; <span class="count-chip">Total: <b style="color:var(--accent2)">${total(curImg())}</b></span>`;
    ED.ecounts.innerHTML = h;
  }
  function updateColonyPanel(){
    const d=selById(selId);
    if(d && d.cls===4){
      ED.colony.classList.remove('hidden');
      if(document.activeElement!==ED.colcount) ED.colcount.value = d.colCount||0;
      ED.colclasses.querySelectorAll('.clsbtn').forEach(b=>b.classList.toggle('active', +b.dataset.cc===d.colClass));
    } else {
      ED.colony.classList.add('hidden');
    }
  }

  function setMode(m){
    mode=m;
    ED.svg.classList.toggle('adding', m==='add');
    ED.m.querySelector('#emInspect').classList.toggle('on', m==='inspect');
    ED.m.querySelector('#emAdd').classList.toggle('on', m==='add');
  }
  function setClass(c){
    activeCls=c;
    ED.classes.querySelectorAll('.clsbtn').forEach(b=>b.classList.toggle('active', +b.dataset.c===c));
    const d=selById(selId);
    if(d && d.cls!==c){
      d.cls=c;
      if(c===4){ if(d.colCount==null) d.colCount=0; if(d.colClass==null) d.colClass=null; }
      else { d.colCount=undefined; d.colClass=undefined; }
      renderStage(); renderApp();
    }
  }
  function setColCount(v){
    const d=selById(selId); if(!d||d.cls!==4) return;
    d.colCount=Math.max(0, v|0); renderStage(); renderApp();
  }
  function setColClass(c){
    const d=selById(selId); if(!d||d.cls!==4) return;
    d.colClass=c; if(!(d.colCount>0)) d.colCount=1;   // picking a class implies at least one cell
    renderStage(); renderApp();
  }
  function selById(id){ return curImg().dets.find(d=>d.id===id); }
  function deleteSel(){
    const img=curImg(); const d=selById(selId); if(!d) return;
    img.dets = img.dets.filter(x=>x.id!==selId); selId=null; renderStage();
  }

  // coordinate helpers
  function toImg(e){ const p=ED.svg.createSVGPoint(); p.x=e.clientX; p.y=e.clientY;
    const q=p.matrixTransform(ED.vp.getScreenCTM().inverse()); return {x:q.x,y:q.y}; }
  function toVB(e){ const p=ED.svg.createSVGPoint(); p.x=e.clientX; p.y=e.clientY;
    return p.matrixTransform(ED.svg.getScreenCTM().inverse()); }

  function onDown(e){
    if(mode==='add'){
      e.preventDefault();
      drawing=true; drawStart=toImg(e);
      ED.temp.style.display=''; ED.temp.setAttribute('stroke', COLORS[activeCls]);
      setTemp(drawStart.x, drawStart.y, 0, 0);
    } else {
      // background press -> pan (rect presses handled on the rect, they stopPropagation)
      panning=true; panLast=toVB(e); selId=null; renderStage();
    }
  }
  function onMove(e){
    if(drawing){
      const p=toImg(e);
      setTemp(Math.min(p.x,drawStart.x), Math.min(p.y,drawStart.y),
              Math.abs(p.x-drawStart.x), Math.abs(p.y-drawStart.y));
    } else if(panning){
      const c=toVB(e); tx+=(c.x-panLast.x); ty+=(c.y-panLast.y); panLast=c; applyVP();
    }
  }
  function onUp(e){
    if(drawing){
      drawing=false; ED.temp.style.display='none';
      const r=tempRect();
      if(r.w>4 && r.h>4){
        const d={id:uid++, cls:activeCls, score:null, added:true,
                 x1:round(r.x), y1:round(r.y), x2:round(r.x+r.w), y2:round(r.y+r.h)};
        if(activeCls===4){ d.colCount=0; d.colClass=null; }   // new colony starts unassigned
        curImg().dets.push(d); selId=d.id; renderStage(); renderApp();
      }
    }
    panning=false;
  }
  function onWheel(e){
    e.preventDefault();
    const before=toImg(e), vb=toVB(e);
    scale=clamp(scale*(e.deltaY<0?1.15:1/1.15), 0.4, 16);
    tx=vb.x-before.x*scale; ty=vb.y-before.y*scale; applyVP();
  }
  function round(v){ return Math.round(v*10)/10; }
  function setTemp(x,y,w,h){ ED.temp.setAttribute('x',x); ED.temp.setAttribute('y',y);
    ED.temp.setAttribute('width',w); ED.temp.setAttribute('height',h); ED.temp._r={x,y,w,h}; }
  function tempRect(){ return ED.temp._r || {x:0,y:0,w:0,h:0}; }

  function nav(delta){
    idx=(idx+delta+PICO.images.length)%PICO.images.length;
    selId=null; loadImage();
  }
  function onKey(e){
    const typing = document.activeElement && document.activeElement.tagName==='INPUT';
    if(e.key==='Escape'){ if(typing){ document.activeElement.blur(); return; } closeEditor(); return; }
    if(typing) return;   // don't fire shortcuts while entering a colony count
    if(e.key==='Delete'||e.key==='Backspace'){ e.preventDefault(); deleteSel(); return; }
    if(e.key==='ArrowLeft'){ nav(-1); return; }
    if(e.key==='ArrowRight'){ nav(1); return; }
    if(e.key==='a'||e.key==='A'){ setMode('add'); return; }
    if(e.key==='v'||e.key==='V'){ setMode('inspect'); return; }
    if(e.key==='f'||e.key==='F'){ fit(); return; }
    const n=parseInt(e.key,10);
    if(n>=1 && n<=ALL.length){ setClass(ALL[n-1]); }
  }

  // ---------- save corrected image ----------
  function saveImage(){
    const img=curImg();
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
      cv.toBlob(b=>dl(b, img.name.replace(/\.[^.]+$/,'')+'_checked.png'));
    };
    im.src=img.src;
  }

  renderApp();
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
