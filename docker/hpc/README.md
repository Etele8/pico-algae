# Pico-Algae on the ITU HPC (Apptainer + SLURM)

The cluster runs everything as SLURM **jobs** (no heavy work on the login node),
inside **Apptainer** containers. ITU already ships a PyTorch container, so there
is **nothing to build or pull** — we reuse it and add three small packages.

Facts these jobs rely on (from ITU's HPC intro, hpc.itu.dk):

- Login: `ssh USER@hpc.itu.dk`. Your files live in `/home/USER` — **no backup**,
  so download trained models promptly.
- Partitions: **`acltr`** = general GPU nodes (3-day student limit);
  **`scavenge`** = CPU / low-priority (24h). GPU jobs need a **typed** `--gres`
  (e.g. `gpu:v100:1`) or they see *no* GPU.
- `$HOME` is auto-bind-mounted into Apptainer, so absolute paths pass through.
  With `--nv` the container gets the GPU; you do **not** `module load CUDA`.
- Pre-made containers live in `/opt/itu/containers/` (and `/home/CONTAINERS`).

## 1. Get code + data onto the HPC

Clone wherever you like — the jobs default `REPO` to `$SLURM_SUBMIT_DIR`, i.e. the
directory you run `sbatch` from, so **always submit from the repo root** and the
clone path doesn't matter (this README uses `~/pico-algae`; adjust to yours, e.g.
`~/projects/pico-algae`). The merged dataset must sit under that repo at
`data/processed/dataset_merged`.

```bash
# on the HPC login node:
git clone https://github.com/Etele8/pico-algae.git ~/pico-algae      # or: cd <repo> && git pull

# from your laptop (the merged dataset is NOT in git) — target YOUR repo path:
scp -rp data/processed/dataset_merged \
    USER@hpc.itu.dk:~/pico-algae/data/processed/
# optional warm-start seed (the current served model, ~158 MB):
scp -rp runs/tuning/train/best_train_model.pt \
    USER@hpc.itu.dk:~/pico-algae/runs/tuning/train/
```

> If you cloned elsewhere, either `cd` there before `sbatch`, or pass it
> explicitly: `REPO=$HOME/projects/pico-algae sbatch docker/hpc/prepare_index.job`.

`ls /opt/itu/containers/` to confirm the current PyTorch SIF name; if it differs
from the default in the jobs, pass `SIF=/opt/itu/containers/.../<name>.sif` to
each `sbatch`.

## 2. Submit the jobs (in order)

```bash
cd ~/pico-algae
sbatch docker/hpc/setup_env.job          # 1. add opencv/pandas/pyyaml + verify imports (once)
sbatch docker/hpc/prepare_index.job      # 2. rewrite index paths for the cluster
sbatch docker/hpc/train_3ch.job          # 3a. train 3ch (acltr, GPU)
sbatch docker/hpc/train_6ch.job          # 3b. train 6ch (acltr, GPU) — runs in parallel
```

Monitor: `squeue -u $USER` (ST column: PD pending / R running); cancel with
`scancel <JOBID>`; watch a run with `tail -f pico_3ch_<JOBID>.out`. Per-epoch
count-MAE lands in `runs/merged_{3,6}ch/logs.jsonl`.

Check `pico_setup_*.out` first — if it printed `OK - all imports resolve`,
you're good. If the imports failed, build a self-contained SIF instead:
`sbatch docker/hpc/build_container.job`, then pass
`SIF=$HOME/containers/pico-algae.sif` to the later jobs.

## 3. Compare + package the winner

```bash
for t in runs/merged_3ch runs/merged_6ch; do
  python - "$t" <<'PY'
import json,sys
r=[json.loads(l) for l in open(sys.argv[1]+"/logs.jsonl")]
b=min(r,key=lambda x:x["count_mae"]); print(sys.argv[1],"epoch",b["epoch"],"MAE",round(b["count_mae"],3))
PY
done

sbatch docker/hpc/package_3ch.job        # wrap best 3ch -> app-ready best_train_model.pt
```

Then, from your laptop:

```bash
scp -rp USER@hpc.itu.dk:~/pico-algae/best_train_model.pt .
```

and drop it into `runs/tuning/train/` on each PC. Ship 3ch unless 6ch is
*significantly* better (the app rejects non-3ch checkpoints today).

## Files

| file                                   | partition   | what it does                                             |
| -------------------------------------- | ----------- | -------------------------------------------------------- |
| `setup_env.job`                      | scavenge    | add opencv/pandas/pyyaml to`~/.local`, verify imports  |
| `prepare_index.job`                  | scavenge    | regenerate`index.csv`/`split.csv` with cluster paths |
| `train_3ch.job`                      | acltr (GPU) | train og-only model →`runs/merged_3ch`                |
| `train_6ch.job`                      | acltr (GPU) | train og+red fusion →`runs/merged_6ch`                |
| `package_3ch.job`                    | scavenge    | wrap best 3ch into the app's checkpoint format           |
| `pico.def` + `build_container.job` | scavenge    | *fallback* self-contained SIF (no Docker Hub)          |

## Knobs

- **GPU type**: default `--gres=gpu:v100:1` (32 GB, plentiful). Swap for
  `gpu:a30:1`, `gpu:a100_40gb:1`, `gpu:l40s:1`, `gpu:rtx8000:1`, `gpu:h100:1` …
  (`sinfo -o "%P %G %N"` lists what each node has).
- **Overrides**: every job honors `REPO=`, `SIF=`; trainers honor `INIT_CKPT=`
  (`""` = train from COCO only). Flags after the job name pass through, e.g.
  `sbatch docker/hpc/train_3ch.job --epochs 80`.
- **OOM**: `src/configs/train_frcnn.yaml` uses `batch_size: 4`; lower it there
  if a smaller card runs out of memory.
- **Data is not personal** (algae microscopy), so no GDPR restriction applies —
  but `/home` still has no backup, so keep the trained models elsewhere too.
- **Acknowledgment**: publishing results computed here may require crediting the
  resource — check the terms for your allocation.
