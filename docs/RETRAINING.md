# Retraining on the extended dataset (3ch + 6ch)

Goal: retrain the pico-algae detector on the **union** of last year's 249 images
and this year's 121 newly-annotated captures (370 total), producing **both** a
3-channel (og-only, Olympus-friendly) and a 6-channel (og+red early-fusion)
model, then pick the winner by count-MAE. Training runs on the **ITU HPC**.

Class totals in the merged set: EUK 1204, FE 11511, FC 7509, colony 203.

---

## 0. One-time: assemble the merged dataset (local)

Already done by:

```bash
python scripts/build_merged_dataset.py
```

This wrote `data/processed/dataset_merged/` with `images_og/`, `images_red/`,
`labels/`, a frozen `split.csv` (train 304 / val 66, hashed by stem so it is
identical on every machine), and an `index.csv`. Native formats are kept
(webp + png mixed — both loaders handle it).

> The `index.csv` paths point at THIS PC. They get regenerated on the cluster in
> step 2 — do not edit them by hand.

---

> **On the ITU HPC, every step below runs as a SLURM job via Apptainer — do
> not run these commands on the login node.** Ready-made job scripts and the
> submit order live in [`docker/hpc/README.md`](../docker/hpc/README.md); the raw
> commands here are what those jobs execute, kept for reference.

## 1. Upload to the HPC

```bash
# on the HPC login node:
git clone https://github.com/Etele8/pico-algae.git ~/pico-algae   # or: git pull

# from your laptop (dataset is NOT in git):
scp -rp data/processed/dataset_merged \
    USER@hpc.itu.dk:~/pico-algae/data/processed/
```

The model (`runs/tuning/train/best_train_model.pt`, ~158 MB) is the warm-start
seed — copy it too if you want to warm-start:

```bash
scp -rp runs/tuning/train/best_train_model.pt \
    USER@hpc.itu.dk:~/pico-algae/runs/tuning/train/
```

---

## 2. On the HPC: regenerate the index with cluster paths

```bash
cd /home/USER/pico-algae
export DS=/home/USER/pico-algae/data/processed/dataset_merged
python scripts/build_merged_dataset.py --index-only --out-dir "$DS" --root "$DS"
```

`--index-only` does NOT re-copy; it just rewrites `index.csv`/`split.csv` with
the cluster paths. Sanity-check (optional, needs a display; else just eyeball
the printed histogram):

```bash
python scripts/sanity_check_dataset.py --index_csv "$DS/index.csv" --n 20
```

Environment (once): reuse ITU's pre-made PyTorch container and add
opencv/pandas/pyyaml — done by `docker/hpc/setup_env.job`. No venv or Docker Hub
image is needed.

---

## 3. Train both models (same split, same seed → comparable)

The `split` column in `index.csv` freezes the same 66-image validation set for
both runs, so their count-MAE is directly comparable. On ITU submit these as
`docker/hpc/train_6ch.job` and `docker/hpc/train_3ch.job` (they wrap exactly the
commands below inside `apptainer exec --nv`).

```bash
# 6-channel (og + red fusion)
python scripts/train_frcnn.py \
    --index_csv "$DS/index.csv" \
    --out_dir runs/merged_6ch \
    --train_yaml src/configs/train_frcnn.yaml \
    --channels 6 \
    --init_checkpoint runs/tuning/train/best_train_model.pt

# 3-channel (og only)
python scripts/train_frcnn.py \
    --index_csv "$DS/index.csv" \
    --out_dir runs/merged_3ch \
    --train_yaml src/configs/train_frcnn.yaml \
    --channels 3 \
    --init_checkpoint runs/tuning/train/best_train_model.pt
```

Notes:
- `--init_checkpoint` warm-starts from matching tensors only. 3ch loads almost
  everything; 6ch keeps everything except the first conv (6≠3 input channels),
  which stays at COCO init. Omit the flag to train from COCO only.
- Each run writes `checkpoints/last.pt` and `checkpoints/best_mae.pt` plus
  `logs.jsonl` (per-epoch count-MAE). `best_mae.pt` = lowest val MAE.
- Optional hyperparameter search: `scripts/tune_frcnn_train.py` (k-fold) — 6ch
  only today; skip unless the single config underperforms.

---

## 4. Compare

Look at the best `count_mae` (and per-class MAE) in each `logs.jsonl`:

```bash
python - <<'PY'
import json
for tag in ("runs/merged_3ch","runs/merged_6ch"):
    rows=[json.loads(l) for l in open(f"{tag}/logs.jsonl")]
    best=min(rows,key=lambda r:r["count_mae"])
    print(tag, "best epoch",best["epoch"],"MAE",round(best["count_mae"],3))
PY
```

Decision rule (agreed): **ship the 3ch model** unless 6ch is *significantly*
better — only then rework the Olympus flow to also capture a red frame.

---

## 5. Package the winner for the app (3ch)

The app (`app/pico_counter.py`) needs a 3-channel checkpoint that also carries
its tuned settings under `params` (anchors/NMS) plus `classes_to_count` and
`val_score_thresh`. `best_mae.pt` from the trainer only has `{"model", ...}`, so
wrap it with `scripts/package_for_app.py` (submit `docker/hpc/package_3ch.job`
on ITU):

```bash
python scripts/package_for_app.py \
    --ckpt runs/merged_3ch/checkpoints/best_mae.pt \
    --train_yaml src/configs/train_frcnn.yaml \
    --out best_train_model.pt
```

Download `best_train_model.pt`, drop it into `runs/tuning/train/` on each PC
(it replaces the served model). The model is NOT in git and NOT in `Update.bat`,
so distribute the file directly (share drive / USB). Everything else — code —
still updates via the in-app update bar.

If you ever ship 6ch instead, `app/pico_counter.py` (the `in_ch != 3` guard) and
the single-shot capture flow both need changes first — flag it and we'll plan
that separately.
