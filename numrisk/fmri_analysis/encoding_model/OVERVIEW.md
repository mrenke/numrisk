# nPRF encoding/decoding pipeline — overview & status (as of 2026-08-04)

This is a status check of the **standard 1D nPRF pipeline** (stimulus 1, `magjudge` task, non-symbolic format only — this in-scanner task was never run in symbolic format), based on script git history + file dates on `/mnt_03/ds-dnumrisk/derivatives`. Variations (2D model, stim2, smoothed, etc.) are noted briefly at the end but were not run down in detail.

## Standard pipeline (stim1)

```
1. GLM (single-trial betas)     glm_denoise/fit_glm_denoise.py      --denoise
                                 → derivatives/glm_stim1.denoise/sub-XX/ses-1/func/
                                     TYPED_FITHRF_GLMDENOISE_RR.npy (raw GLMsingle output)
                                     sub-XX_..._desc-stims1_pe.nii.gz (per-trial betas, T1w space)

2. Encoding model (CV nPRF fit) encoding_model/fit_nprf_cv.py       --denoise
                                 → derivatives/encoding_model.cv.denoise/sub-XX/ses-1/func/
                                     per-run + mean cv-r2, mu, sd, amplitude, baseline maps

3. Decoding                     encoding_model/decode_select_voxels_cv.py  --denoise --mask NPC_R
                                 → derivatives/decoded_pdfs_stim1.volume.cv_vselect.denoise/sub-XX/func/
                                     *_pars.tsv (decoded PDFs per trial → decoding precision = corr(true, decoded n))
```

Step 2 reads step 1's output via `key = f'glm_stim{split_data}.denoise'`; step 3 reads step 1 directly (`Subject.get_single_trial_volume`) and step 2's params (`Subject.get_prf_parameters_volume`, dir `encoding_model.cv.denoise`) — confirmed by reading `numrisk/utils/data.py`.

## ⚠️ Key finding: encoding model & decoding are stale relative to the current GLM output

File dates on sciencecloud tell a clear story:

| Derivative | Content date(s) | Subjects |
|---|---|---|
| `glm_stim1.denoise` (GLM betas) | **all files: 2025-03-27** (single batch rerun) | 64/67 have data (sub-01, sub-03 empty; stray duplicate empty folder `sub-3` alongside `sub-03`) |
| `encoding_model.cv.denoise` (nPRF fit, CV) | 2024-08-05 / 2024-11-21 | 66 subjects |
| `encoding_model.denoise` (nPRF fit, non-CV) | 2024-08-05 / 11-19 / 11-26 | 66 subjects |
| `decoded_pdfs_stim1.volume.cv_vselect.denoise` (decoding) | **all files: 2024-11-21** | 66 subjects |

The GLM step (`glm_stim1.denoise`) was **completely rerun in a single batch on 2025-03-27** — roughly 4 months *after* the encoding model and decoding that currently sit in `encoding_model.cv.denoise` / `decoded_pdfs_stim1...` were computed. In other words: **the nPRF encoding fits and decoding precisions currently on disk were fit on an older version of the GLM betas that no longer exists** (it was overwritten by the March 2025 rerun). This matches your memory of having "rerun something starting with GLMsingle" — it did go through, but only as far as the GLM step; steps 2 and 3 (`fit_nprf_cv.py`, `decode_select_voxels_cv.py`) were never rerun on top of it.

Git history for the scripts themselves stops at **2025-02-12** (last commit: `stim2 & 2D decoding results`), so no code changes happened after that — the March/April reruns used the already-committed script versions, just executed later on the cluster.

### A second, incomplete rerun attempt (Nov 2025)

There are also two newer, mostly-empty derivative folders, both dated **2025-11-06/07**:
- `glm_stim.denoise` — GLM output for **only 4 subjects**
- `glm_stim.denoise.both` — GLM output (combined stim1+stim2 naming) for **only 1 subject**

Neither has any corresponding encoding-model or decoding output — this looks like a pipeline test/restart that stopped right after the GLM step for a handful of subjects and wasn't carried forward.

The non-CV encoding model (`encoding_model.denoise`, from `fit_nprf.py`) is in the same boat — its per-subject files are all from Aug/Nov 2024 too. The only files touched later (2025-04-28) are a `group-all` average summary under `encoding_model.denoise/averages/`, which just re-aggregates the existing (old) per-subject maps — not a refit against the new GLM betas.

### Bottom line / suggested next step
If you want encoding/decoding results that reflect the current (March 2025) GLM betas, you need to **rerun step 2 (`fit_nprf_cv.py --denoise`, and `fit_nprf.py --denoise` if you still use the non-CV version) and step 3 (`decode_select_voxels_cv.py --denoise --mask NPC_R`)** for all subjects — everything currently in `encoding_model.cv.denoise`, `encoding_model.denoise`, and `decoded_pdfs_stim1.volume.cv_vselect.denoise` predates the current GLM betas. Also worth deciding whether the Nov 2025 `glm_stim.denoise(.both)` 4-subject test is meant to become the new standard GLM (e.g. a further-updated GLMsingle config) or can be discarded.

## Related: downstream measure derived from `encoding_model.cv.denoise`

`/home/ubuntu/git/parietal_patterns/prep_results/thesis_ch-DD-neuro_02_scloud.ipynb` derives a summary measure `mean_R2` (mean cv-R² within the NPC ROI, split by hemisphere: `mean_R2_L`, `mean_R2_R`) by reading directly from `derivatives/encoding_model.cv.denoise` (and `encoding_model.denoise` for the non-CV R²). This measure therefore **inherits the same staleness** noted above and should be recomputed if/when the encoding model is refit.

## Variations run (not the "standard" pipeline, brief notes only)

- **stim2** (second stimulus in the pair): `fit_nprf_cv_stim2.py` / `decode_select_voxels_cv_stim2.py` → `encoding_model_stim2.cv.denoise` (2025-02-13), `decoded_pdfs_stim2.volume.cv_vselect.denoise` (2025-02-07). These are the scripts currently wired into `submit_nprf_cv.sh` / `submit_decode_vselect.sh` — i.e. the checked-in submit scripts point at stim2, not stim1; use `fit_nprf_cv.py`/`decode_select_voxels_cv.py` directly (or edit the submit scripts) to rerun stim1.
- **2D pRF model** (`fit_2D_prf_cv.py`, `decode_2D.py`, "same_rfs" mixture model): `encoding_model.2d.*` folders (Nov 2024), `decoding.2d.mixture.same_rfs` (2024-11-21). Same staleness issue applies (predates the March 2025 GLM rerun).
- **smoothed** variant: `encoding_model.2d.mixture.same_rfs.smoothed` (2024-11-19) — smoothed GLM betas as input, not the default.
- **non-CV encoding model**: `fit_nprf.py` → `encoding_model.denoise`, used for non-cross-validated R² only (e.g. in the `mean_R2` measure above).
- `retroicor` / `pca_confounds` flags exist in the scripts but don't appear to have corresponding derivative folders on disk — likely tried and abandoned early on.
- `halfdata/` subfolder and its scripts are from an earlier, unrelated exploration — can be ignored.
