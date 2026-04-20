# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

(D)NumRisk is a computational neuroscience research project studying how humans represent and process numbers and risk. It combines:
- **Behavioral analysis**: Bayesian cognitive model fitting for risky choice and magnitude judgment tasks
- **fMRI analysis**: Population receptive field (nPRF) models, GLM, connectivity, and cortical gradients
- **DTI analysis**: Tractography and tract-based spatial statistics

The study has ~42 participants (controls vs. dyscalculics), two cognitive tasks, and two number formats (symbolic Arabic numerals vs. non-symbolic dot arrays).

## Installation

```bash
cd /path/to/numrisk  # repo root (parent of this directory)
python setup.py install
```

Key external packages (not in setup.py, must be installed manually):
- `bauer` — Bayesian cognitive models for risk decisions: https://github.com/ruffgroup/bauer
- `braincoder` — Population receptive field fitting: https://braincoder-devs.github.io
- `brainspace` — Cortical gradient analysis
- `pymc`, `arviz` — Bayesian/MCMC
- `nilearn`, `nibabel`, `neuromaps` — Neuroimaging

## Data

BIDS-formatted data lives at `/Users/mrenke/data/ds-dnumrisk` (local path hardcoded in many scripts). Derivatives are written to `ds-dnumrisk/derivatives/`.

Subject IDs: zero-padded integers (e.g., `sub-01`). Session is always `1`. fMRI runs 1–6.

## Running Scripts

Most scripts are run directly as Python with argparse:

```bash
# Fit a behavioral cognitive model (from behavior_risk/)
python fit_bauerModels.py <model_label> --bids_folder /path/to/data --format symbolic

# Fit nPRF encoding model (from fmri_analysis/encoding_model/)
python fit_nprf.py <subject_id> --bids_folder /path/to/data

# Fit models in parallel on a cluster (repo root)
python ../fit_models.py
```

Behavioral model traces are saved as NetCDF: `derivatives/cogmodels_risk/model-{label}_format-{format}_trace.netcdf`

## Code Architecture

### Module Structure

```
numrisk/
├── behavior_risk/        # Risky choice task: model fitting & posterior analysis
│   ├── utils.py          # get_data(), get_behavior(), cleanup_behavior()
│   ├── utils_02.py       # build_model() factory for 12+ model variants
│   ├── fit_bauerModels.py # CLI entry point for model fitting
│   └── analyze_bauer_model*.py  # Posterior analysis & group comparisons
├── behavior_magjudge/    # Magnitude judgment task analysis
├── behavior_general/     # Cross-task behavioral analyses
├── fmri_analysis/
│   ├── encoding_model/   # nPRF fitting via braincoder (fit_nprf.py)
│   ├── glm_denoise/      # GLM with denoising
│   ├── connectivity/     # Functional connectivity & PPI
│   ├── gradients/        # Cortical gradient analysis (Margulies alignment)
│   └── surface/          # Volume→surface transformations (neuromaps)
├── dti_analysis/         # DTI preprocessing, TBSS, connectome
├── prepare/              # BIDS conversion of raw MRI and behavioral data
├── neural_general/       # Shared neural utility functions
├── utils/                # Shared utilities (get_target_dir, etc.)
└── visualize&plots&stats/ # Paper figures & statistical reporting
```

### Behavioral Analysis Data Flow

```
Raw TSV event files (BIDS)
  → get_behavior() / cleanup_behavior()  [utils.py]
  → DataFrame: subject, session, format, trial_nr, choice, n1, n2, prob1, prob2
  → get_data()  [joins with participant demographics/group]
  → build_model(model_label, df)  [utils_02.py — factory returning bauer model]
  → model.build_estimation_model() → model.sample(burnin, samples)
  → az.to_netcdf(trace, ...)
```

### Behavioral Model Variants (`build_model` in `utils_02.py`)

Models differ in prior structure and noise specification, all using `bauer` package classes:
- `'1'`: Shared prior (probit-like), `RiskRegressionModel`
- `'2'/'2b'`: Separate priors for safe/risky options
- `'3'`: KLW model (no prior_mu)
- `'4'`: KLW + lapse rate, `RiskLapseRegressionModel`
- `'5'–'12'`: Flexible noise variants, `FlexibleNoiseRiskRegressionModel`

Models regress parameters on `group` (control vs. dyscalculic). Group-level posteriors are compared to assess behavioral differences.

### fMRI Encoding Model Flow

```
Preprocessed BOLD (fmriprep output, 4D NIfTI)
  → paradigm: log(n1) or log(n2) per trial
  → GaussianPRF model [braincoder]
  → ParameterFitter.fit() → parameter maps (R2, mu, sigma, size)
  → Saved to derivatives/encoding_model*/sub-{id}/
  → Gradient analysis: correlate nPRF params with Margulies gradients
```

### Key Shared Patterns

- **Format branching**: Most analyses split on `format` ('symbolic' vs 'non-symbolic'), often using `df.xs(format, level='format')`
- **Multi-index DataFrames**: Subject, session, format, trial_nr as index levels; use `.xs()` for slicing
- **BIDS paths**: Helpers like `get_target_dir(subject, session, bids_folder, derivative_name)` from `numrisk.utils`
- **Subject lists**: Derived from `listdir(bids_folder)` filtering for `sub-XX` folders

### Notebooks

~100 Jupyter notebooks contain exploratory and paper-quality analyses. The primary active notebooks are `behavior_risk/probit_model1.ipynb` and `visualize&plots&stats/paperRisk_01.ipynb`.

### Key Scripts / Notebooks by Purpose

| Purpose | Location |
|---|---|
| IQ scores from IDS-2 screener (reads `add_measure_1/2/3.xlsx`, saves `iq-scores_ids2.csv`) | `prepare/extract_measures_iq.ipynb` |
| Working memory measures (raw Excel → `derivatives/phenotype/`) | `prepare/extract_measures_workingmemory.ipynb` |
| Math skill, anxiety & confidence (raw Excel → `derivatives/phenotype/`) | `prepare/extract_measures_math.ipynb` |
| Cross-measure correlations & regressions | `behavior_general/across-measur_ana.ipynb` |
| Paper figures & stats for risk task | `visualize&plots&stats/paper_riskDD_*.ipynb` |
| Behavioral model fitting (CLI) | `behavior_risk/fit_bauerModels.py` |
| Posterior analysis & group comparisons | `behavior_risk/analyze_bauer_model*.py` |
