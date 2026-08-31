# Retraining on the extended dataset (3ch + 6ch)

Retrain the pico-algae detector on the **union** of last year's 249 images and
this year's 121 newly-annotated captures (370 total), producing **both** a
3-channel (og-only, Olympus-friendly) and a 6-channel (og+red early-fusion)
model, then ship whichever counts better.

This doc explains *what* each stage does and *why*. The concrete way to run it
on the **ITU HPC** — SLURM jobs, Apptainer, the exact `sbatch` order — lives in
[`docker/hpc/README.md`](../docker/hpc/README.md); the training itself runs
there, not on this PC (it's CPU-only). Use the reference commands below if you
ever train somewhere else.

## The merged dataset

| class | last year (249) | this year (121) | merged |
|---|---|---|---|
| EUK (0) | 979 | 225 | 1 204 |
| FE (1) | 9 634 | 1 877 | 11 511 |
| FC (2) | 5 458 | 2 051 | 7 509 |
| colony (3) | 110 | 93 | 203 |

The new captures are **FC-rich** (FC > FE, the opposite balance of the old set)
and nearly **double the colony class** — the two things worth watching in the
per-class results, since colony was by far the weakest before. Union training
(not new-only) avoids catastrophic forgetting of last year's distribution.

## Pipeline at a glance

| stage | script | ITU job |
|---|---|---|
| assemble merged dataset (local) | `scripts/build_merged_dataset.py` | — (run once on your PC) |
| add deps to the container | — | `docker/hpc/setup_env.job` |
| regenerate index for the cluster | `scripts/build_merged_dataset.py --index-only` | `docker/hpc/prepare_index.job` |
| train 3ch / 6ch | `scripts/train_frcnn.py --channels {3,6}` | `docker/hpc/train_3ch.job` / `train_6ch.job` |
| package winner for the app | `scripts/package_for_app.py` | `docker/hpc/package_3ch.job` |

## 0. Assemble the merged dataset (local — already done)

```bash
python scripts/build_merged_dataset.py
```

Writes `data/processed/dataset_merged/` (`images_og/`, `images_red/`, `labels/`)
with an `index.csv` and a frozen `split.csv` — **train 304 / val 66**, assigned
by hashing each stem so the split is identical on every machine and stable across
runs. Native formats are kept (webp + png mixed; both loaders read either, and
parse int or float label coords). The script prints the class histogram and flags
any stem collisions or malformed label lines.

> The paths written into `index.csv` point at the machine that built it. On the
> cluster they're regenerated with `--index-only` (see `prepare_index.job`) —
> never hand-edit them. The loaders open the path column verbatim.

## 1. Move code + data to the HPC

```bash
# on the login node:
git clone https://github.com/Etele8/pico-algae.git ~/pico-algae   # or: git pull

# from your laptop (dataset is NOT in git):
scp -rp data/processed/dataset_merged USER@hpc.itu.dk:~/pico-algae/data/processed/
# optional warm-start seed (the current served model, ~158 MB):
scp -rp runs/tuning/train/best_train_model.pt USER@hpc.itu.dk:~/pico-algae/runs/tuning/train/
```

## 2. Train both models (reference command)

Both channel counts come from **one** script. The `split` column freezes the same
66-image validation set for both, so their count-MAE is directly comparable.

```bash
DS=~/pico-algae/data/processed/dataset_merged

python scripts/train_frcnn.py \
    --index_csv "$DS/index.csv" \
    --out_dir runs/merged_3ch \        # or runs/merged_6ch
    --train_yaml src/configs/train_frcnn.yaml \
    --channels 3 \                     # or 6
    --init_checkpoint runs/tuning/train/best_train_model.pt
```

- `--channels 3` = og-only (`PicoOgDetectionDataset` + 3ch model); `--channels 6`
  = og+red early fusion.
- `--init_checkpoint` warm-starts from matching tensors only: 3ch loads almost
  everything from the current model; 6ch keeps all but the first conv (6 ≠ 3 input
  channels), which stays at its COCO init. Omit it to train from COCO only.
- Each run writes `checkpoints/last.pt`, `checkpoints/best_mae.pt` (lowest val
  count-MAE), and `logs.jsonl` (per-epoch metrics).
- On ITU, submit `train_3ch.job` and `train_6ch.job` instead — they wrap exactly
  this inside `apptainer exec --nv` on the `acltr` GPU queue, and can run in
  parallel. Optional k-fold search: `scripts/tune_frcnn_train.py` (6ch only for
  now) — skip unless the single config underperforms.

## 3. Compare, and decide

```bash
for t in runs/merged_3ch runs/merged_6ch; do
  python - "$t" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1] + "/logs.jsonl")]
best = min(rows, key=lambda r: r["count_mae"])
print(sys.argv[1], "best epoch", best["epoch"], "MAE", round(best["count_mae"], 3))
PY
done
```

**Decision rule (agreed): ship the 3ch model** unless 6ch is *significantly*
better on count-MAE. 6ch needs both the og and red frame at inference, which
suits the upload flow but breaks the single-screenshot Olympus capture — so it's
only worth adopting if the accuracy gain justifies reworking that flow.

## 4. Package the winner for the app (3ch)

The app (`app/pico_counter.py`) loads a 3-channel checkpoint and reads its tuned
settings from the file: anchors/NMS under `params`, plus `val_score_thresh` and
`classes_to_count`. The trainer's `best_mae.pt` only holds `{"model", ...}`, so
if you served it raw the app would fall back to the **wrong default anchors**
(`[[16]…]` instead of the trained `[[8]…]`) and detect poorly. Wrap it first:

```bash
python scripts/package_for_app.py \
    --ckpt runs/merged_3ch/checkpoints/best_mae.pt \
    --train_yaml src/configs/train_frcnn.yaml \
    --out best_train_model.pt
```

(On ITU: `package_3ch.job`.) Then distribute the file:

```bash
# from your laptop:
scp -rp USER@hpc.itu.dk:~/pico-algae/best_train_model.pt .
```

Drop `best_train_model.pt` into `runs/tuning/train/` on each PC — it replaces the
served model. The model is **not** in git and **not** carried by `Update.bat`, so
it's shared directly (share drive / USB). Code keeps updating via the in-app
update bar as usual.

## Shipping 6ch instead

If 6ch wins big and you decide to adopt it, two things need work first: the
`in_ch != 3` guard in `app/pico_counter.py`, and the single-shot Olympus capture
(it would have to grab a red frame too). Flag it and we'll plan that separately.
