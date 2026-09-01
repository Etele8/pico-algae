# Shipping a retrained model to every PC (via the updater)

The model is too big for git (`/runs` is gitignored; GitHub blocks files >100 MB),
so it rides on a **GitHub Release** and the updater fetches it. You publish once;
every colleague's `Update.bat` (or the in-app "new version" bar) downloads it the
next time they update. Only changed models download — a hash guard skips it
otherwise.

## One-time per new model

1. **Package** the app-ready checkpoint (see the retraining flow):
   ```bash
   python scripts/package_for_app.py \
       --ckpt runs/ship_3ch/checkpoints/best_mae.pt \
       --train_yaml src/configs/train_frcnn.yaml \
       --post_summary runs/post_3ch/best_post_summary.json \
       --out best_train_model.pt
   ```

2. **Get its SHA-256:**
   ```bash
   sha256sum best_train_model.pt                       # Linux / Git Bash
   # or on Windows:  (Get-FileHash best_train_model.pt -Algorithm SHA256).Hash
   ```

3. **Publish a GitHub Release** with the file attached (pick a tag, e.g. a date):
   ```bash
   gh release create model-2026-09 best_train_model.pt \
       --title "Counting model 2026-09" --notes "Retrained on the extended dataset."
   ```
   (Or do it in the GitHub web UI: Releases -> Draft a new release -> attach the
   `.pt`.) The asset's download URL is then:
   `https://github.com/Etele8/pico-algae/releases/download/model-2026-09/best_train_model.pt`

4. **Point the manifest at it** — edit [`app/model.json`](../app/model.json):
   ```json
   {
     "version": "model-2026-09",
     "url": "https://github.com/Etele8/pico-algae/releases/download/model-2026-09/best_train_model.pt",
     "sha256": "<the hash from step 2>"
   }
   ```
   `version` is any string that changes each release (the tag is convenient).

5. **Commit + push** `app/model.json`.

That's it. On each PC, the next `Update.bat` run: refreshes the code (which
brings the new `app/model.json`), then downloads the model because its `version`
differs from the local stamp (`runs/tuning/train/MODEL_VERSION.txt`), verifies the
SHA-256, and installs it at `runs/tuning/train/best_train_model.pt`. Unchanged
`version` -> skipped, no big download.

## Notes

- **First fetch is ~160 MB per PC** (once per model). Code-only updates stay tiny.
- **Fresh installs** (Setup.exe / zip) still bundle a model, so a new PC works
  offline out of the box; the updater only pulls a *newer* one. If the bundled
  file already matches the manifest hash, the updater writes the stamp and skips
  the download.
- **Failures are non-fatal:** a download error or checksum mismatch leaves the
  current model untouched and the update still succeeds (see
  [`tools/update_model.ps1`](../tools/update_model.ps1)).
- The manifest is committed; the per-install stamp `MODEL_VERSION.txt` lives
  under `runs/` (gitignored), so it never travels between machines.
