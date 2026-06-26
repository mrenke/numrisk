# behavior_magjudge

Analysis of the magnitude-judgment task (in-scanner, symbolic numerals). Historically fit with
`bauer`-based choice/comparison models in `utils_02.py` (`build_model`, `get_subwise_params`).

## New modeling (2026, external)

A parallel, more thorough re-fit of this task was done by a collaborator:

- Repo: https://github.com/ruffgroup/dyscalculic_ddm
- Fits three response layers on top of a shared Bayesian-observer model: a static **choice**
  model (cumulative-normal, no RT), a **DDM**, and a **race-diffusion model (RDM, preferred)**.
- Requires `bauer` **v0.4.0** exactly (`pip install --force-reinstall "git+https://github.com/ruffgroup/bauer.git@v0.4.0"`,
  tag resolves to commit `e406694`) — v0.3.0 / main have subtly different model code.
- Headline result: dyscalculics have noisier number representations (perceptual noise
  ~0.34→0.45, memory noise roughly doubles) with a compensatory increase in decision threshold
  (slower but only slightly less accurate), and stronger compression effects.

Posterior traces/results for these models live on the group share:
`/Volumes/g_econ_department$/projects/2023/renkert_dehollander_ruff_dnumr/data/ddm_results`

### Key difference from our older models

The collaborator's repo describes a cell-means group parameterization,
`random_regressors={p: '0 + C(group)'}` (separate population mean *and* SD per group), as the
default for a between-subjects factor. **Checked against the actual local trace**
(`derivatives/cogmodels_magjudge/model-choice_full_trace.nc`), this is *not* what was used for the
`choice_full` fit: its `regressor` coordinates are `['Intercept', 'group[T.dyscalc]']` — the same
reference-cell design as our old models, just with patsy's explicit `[T.level]` label instead of
the bare `'group'`. That label change (not a coding change) is enough to make even the simple
choice model's group contrast shift slightly relative to our old `'1'`/`'2'` models in
`utils_02.py`. (Other variants from the repo, e.g. `choice_group`, may use the cell-means design —
re-check `idata.posterior[param_name].coords` for any new trace before assuming either coding.)

`get_subwise_params` (`utils_02.py:14-34`) has been updated to find "whichever regressor isn't
`Intercept`" instead of hardcoding the literal string `'group'`, so it now works unchanged on both
old and new (`bauer` v0.4.0) traces. Double check that `group == 1` in your `group_list` actually
means dyscalculic — the new label makes that mapping explicit (`group[T.dyscalc]`), and a mismatch
would silently apply the slope to the wrong group.
