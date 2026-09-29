# Group comparison summary: extent vs. strength, dyscalculia vs. control

Results from `r2_mixture_group_dyscalculia.ipynb`, first pass (2026-09-29). See
that notebook and `tutorial_r2_mixture_extent_vs_strength.md` /
`... _update1.md` for methodology; this file just tracks the numbers.

**Sample**: group labels cross-checked between `group_assignment.csv` and
`subjects_recruit_scan_scanned-final.csv`, 0 disagreements, 33 control / 33
dyscalc. Per-space subject counts below reflect which subjects had usable R² +
ROI mask data in that space.

## Guard rails / fit quality, by space

| | fsaverage5 | T1w |
|---|---|---|
| N fit successfully | 65/66 (sub-03 missing) | 63/66 (sub-03, sub-65, sub-66 missing) |
| `w_s > w_n` flagged | 13/65 (20%) | **63/63 (100%)** |
| "not separated" flagged | 3/65 | 0/63 |
| ROI tail-FDR = inf | 0/65 | **59/63 (94%)** |
| whole-brain `noise_weight` | ~1–2% | ~33–35% |
| seed-stability (std of w_s, seed 0-9) | ~0.0007 | ~0.0008 |

**T1w's guard rails are lighting up hard.** Per tutorial §5, `w_s > w_n` firing
for essentially every subject usually means the mixture is carving up one skewed
blob rather than finding a real noise/signal split, not a biological effect. The
T1w numbers below should be treated as unreliable until this is resolved (start
with the per-subject diagnostic PDF).

The whole-brain `noise_weight` row is also not a fit-quality problem so much as a
definitional one: in T1w, `load_masks`'s "whole brain" is literally the fMRIPrep
brain mask (white matter, subcortex, ventricles included — the loader's own code
comment says *"whole brain, not cortex, in this space"*), whereas fsaverage5's is
cortical surface only. The two `wb_*` columns are not measuring the same thing,
so cross-space comparisons of whole-brain quantities aren't apples-to-apples.

## 1. Noise floor (§7, tested first)

| Test | fsaverage5 group p | T1w group p |
|---|---|---|
| `wb_noise_mu ~ group + mean_fd + n_runs` | 0.594 (ns) | **0.029** |
| `roi_noise_mu ~ group + mean_fd + n_runs` | **0.006** | **0.014** |

In fsaverage5 the noise-floor difference was region-specific (ROI only, not
whole-brain) — not obviously explained by `mean_fd`/`n_runs` (both included as
covariates and neither reaches significance), and not resolved by the deferred
permutation-null/cvR² checks. In T1w it shows up in *both* whole-brain and ROI,
though given the whole-brain definitions differ between spaces (above) and the
ROI fits are mostly degenerate (94% infinite FDR), this shouldn't be read as
confirming a broader/global effect.

## 2. Extent / strength — free ROI fit

| Test | fsaverage5 group p | T1w group p |
|---|---|---|
| logit(signal_weight) ~ group [extent] | 0.655 (ns) | 0.730 (ns) |
| signal_mu ~ group [strength] | 0.603 (ns) | 0.167 (ns) |

No group effect in either space's free fit. In both spaces `roi_noise_mu`
strongly predicts `signal_mu` (fsaverage5: coef 1.56, p<0.001; T1w: coef 0.88,
p<0.001) — the trade-off/collinearity tutorial §6 warns about.

## 3. Extent / strength — noise-pinned ROI fit (§6.2a)

| Test | fsaverage5 group p | T1w group p |
|---|---|---|
| logit(signal_weight) ~ group [extent, pinned] | 0.270 (ns) | 0.187 (ns) |
| signal_mu ~ group [strength, pinned] | **0.001** | 0.102 (ns) |

**Key divergence between spaces**: fsaverage5's noise-pinned fit shows a
significant group effect on strength (dyscalculics lower, matching tutorial
§8's "weaker signal" row); the T1w one does not. Given T1w's near-universal
guard-rail failures, the fsaverage5 result should be weighted more heavily right
now — but "weighted more" is not "trusted," since fsaverage5 still has its own
unresolved ROI-`noise_mu` group difference (§1 above), and none of the deferred
checks (permutation-null anchor, bootstrap CIs, cvR² leave-one-run-out) have run
yet.

## Descriptive means (± SD) by group

**fsaverage5**

| | control | dyscalc |
|---|---|---|
| `roi_signal_weight` | 0.442 ± 0.054 | 0.449 ± 0.063 |
| `roi_signal_mean_r2` | 0.0305 ± 0.0107 | 0.0256 ± 0.0100 |
| `roi_noise_mean_r2` | 0.0144 ± 0.0026 | 0.0127 ± 0.0030 |
| `roi_delta_mu` | 0.731 ± 0.180 | 0.678 ± 0.161 |
| `pinned_signal_weight` | 0.898 ± 0.234 | 0.965 ± 0.122 |
| `pinned_signal_mean_r2` | 0.0225 ± 0.0090 | 0.0175 ± 0.0051 |

**T1w** (fit-quality caveat above applies — descriptive only)

| | control | dyscalc |
|---|---|---|
| `roi_signal_weight` | 0.642 ± 0.040 | 0.655 ± 0.030 |

## Bottom line

- **fsaverage5**: no free-fit group effect; noise-pinned fit shows a "weaker
  signal, not less area" pattern (p=0.001) — but flagged for an unresolved
  ROI-specific noise-floor group difference that isn't explained by the motion
  covariates on hand.
- **T1w**: no significant group effect anywhere, and the fits themselves look
  unreliable by the tutorial's own guard rails (near-universal `w_s>w_n`,
  mostly-infinite FDR thresholds). Read "no effect in T1w" as "these fits need
  fixing first," not as a disconfirmation of the fsaverage5 result.
- Neither space's result is publication-grade yet: §6.2(b) permutation-null
  anchor, §6.1 bootstrap CIs, and the §7 cvR² leave-one-run-out check are all
  still deferred (see the notebook's closing cell).

## Known issue, fixed during this pass

The notebook's noise-pinned-fit regression cell and reporting table initially
showed **stale fsaverage5 output** after switching `SPACE` to `'T1w'` and
re-running — only the whole-brain/free-ROI-fit cells had actually been
re-executed; the pinned-fit cell and reporting cell still displayed Monday's
run (N=65) instead of Tuesday's (N=63). The T1w numbers in this file were
recomputed directly from the saved CSV
(`nPRF_r2mixture_group-dyscalculia_space-T1w_roi-NPC.csv`) to avoid reporting
stale output — re-run the whole notebook top-to-bottom before trusting its
on-screen output for any given `SPACE`.
