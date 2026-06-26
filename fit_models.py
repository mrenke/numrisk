"""
fit_models.py
-------------
Systematically fit cognitive models for the numrisk risky choice paper.

Run from the numrisk/numrisk/behavior_risk/ directory, or call with --script_dir:
    python fit_models.py
    python fit_models.py --models 8 9 12reg --overwrite
    python fit_models.py --dry_run

Models overview (from utils_02.py):
  Model grid — two sources of asymmetry × two noise models:
  ┌──────────────────────────┬──────────────────────────┬──────────────────────────┐
  │                          │  Asymmetric evidence SDs │  Asymmetric prior SDs    │
  │                          │  (n1_sd ≠ n2_sd)         │  (risky_sd ≠ safe_sd)    │
  ├──────────────────────────┼──────────────────────────┼──────────────────────────┤
  │  Normal RiskRegression   │  8  /  8reg              │  13  /  13reg  (NEW)     │
  │  FlexibleNoise           │  9  /  9reg              │  11  /  11reg            │
  └──────────────────────────┴──────────────────────────┴──────────────────────────┘
  Also: model-3 (KLW, baseline), model-12/12reg (natural-space variant of model-8/8reg)

  LOO(Xreg) > LOO(X)  →  formal evidence that group membership improves fit.
  Posterior of group regression coefficients → direction and size of effect.
"""

import argparse
import os
import os.path as op
import sys
import time
import logging

import numpy as np
import arviz as az
import pymc as pm

# ---------------------------------------------------------------------------
# Allow running from any working directory by adding the script's own dir
# to sys.path so that utils.py / utils_02.py are importable.
# ---------------------------------------------------------------------------
SCRIPT_DIR = op.dirname(op.abspath(__file__))
BEHAVIOR_RISK_DIR = op.join(SCRIPT_DIR, 'numrisk', 'behavior_risk')
if BEHAVIOR_RISK_DIR not in sys.path:
    sys.path.insert(0, BEHAVIOR_RISK_DIR)

from utils import get_data
from utils_02 import build_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BIDS_FOLDER   = '/Users/mrenke/data/ds-dnumrisk'
TARGET_FOLDER = op.join(BIDS_FOLDER, 'derivatives', 'cogmodels_risk')

# Full set of models to run in order.
#
# Strategy for the paper:
#   Step 1 — Model architecture comparison (no regressors):
#     Compare 3, 8, 9, 11, 12 to find the best-fitting cognitive account.
#   Step 2 — Group effect test (with vs. without regressors):
#     Compare model-X vs model-Xreg on the winning architecture.
#     LOO improvement = formal evidence for dyscalculia group effect.
#     Posterior of group coefficients = characterise direction/size.
#
# Edit this list to add / remove models.
DEFAULT_MODELS = [
    # --- Step 1: architecture comparison (no group regressors) ---
    '3',       # KLW baseline
    '8',       # asymmetric evidence SDs (n1≠n2), normal noise
    '9',       # asymmetric evidence SDs, FlexNoise
    '13',      # asymmetric prior SDs (risky≠safe), normal noise        ← NEW
    '11',      # asymmetric prior SDs, FlexNoise
    '12',      # natural space variant of model-8
    # --- Step 2: group effect (add group regressors to each architecture) ---
    '8reg',    # group effect on asymmetric evidence SDs, normal noise
    '9reg',    # group effect on asymmetric evidence SDs, FlexNoise
    '13reg',   # group effect on asymmetric prior SDs, normal noise      ← NEW
    '11reg',   # group effect on asymmetric prior SDs, FlexNoise
    '12reg',   # group effect, natural space
]

BURNIN  = 2000
SAMPLES = 2000
TARGET_ACCEPT = 0.9

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(op.join(SCRIPT_DIR, 'fit_models.log'), mode='a'),
    ]
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def trace_path(model_label, fmt, remove_sub_string=None):
    label = model_label
    if remove_sub_string:
        label = f'{label}_rem-{remove_sub_string}'
    return op.join(TARGET_FOLDER, f'model-{label}_format-{fmt}_trace.netcdf')


def already_done(model_label, fmt, remove_sub_string=None):
    return op.exists(trace_path(model_label, fmt, remove_sub_string))


def fit_one(model_label, df, fmt,
            remove_sub_string=None,
            overwrite=False):

    out_path = trace_path(model_label, fmt, remove_sub_string)

    if already_done(model_label, fmt, remove_sub_string) and not overwrite:
        log.info(f'[SKIP]  model-{model_label}  format-{fmt}  — trace already exists at {out_path}')
        return

    log.info(f'[START] model-{model_label}  format-{fmt}')
    t0 = time.time()

    model = build_model(model_label, df)
    model.build_estimation_model()
    trace = model.sample(BURNIN, SAMPLES, target_accept=TARGET_ACCEPT)

    with model.estimation_model:
        pm.compute_log_likelihood(trace)

    os.makedirs(TARGET_FOLDER, exist_ok=True)
    az.to_netcdf(trace, out_path)

    elapsed = (time.time() - t0) / 60
    log.info(f'[DONE]  model-{model_label}  format-{fmt}  — saved to {out_path}  ({elapsed:.1f} min)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Fit cognitive models for numrisk risky choice paper.')
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help='Model label(s) to fit (default: all in DEFAULT_MODELS list)')
    parser.add_argument('--format', default='symbolic',
                        choices=['symbolic', 'non-symbolic'],
                        help='Number format to fit (default: symbolic)')
    parser.add_argument('--bids_folder', default=BIDS_FOLDER,
                        help='Path to BIDS data folder')
    parser.add_argument('--include_all', action='store_true',
                        help='Include ALL subjects (for supplement). Default is to exclude outliers.')
    parser.add_argument('--remove_sub_string', default='32-40-45-46-50',
                        help='Hyphen-separated subject IDs to remove (default: 32-40-45-46-50)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-fit even if a trace file already exists')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be run without fitting anything')
    args = parser.parse_args()

    # Default behaviour: remove outlier subjects UNLESS --include_all is set
    remove_sub_string = None if args.include_all else args.remove_sub_string

    # ------------------------------------------------------------------
    # Print status table
    # ------------------------------------------------------------------
    log.info('=' * 60)
    log.info('numrisk cognitive model fitting')
    log.info(f'  format          : {args.format}')
    log.info(f'  bids_folder     : {args.bids_folder}')
    log.info(f'  remove_subjects : {not args.include_all}  (excluded: {remove_sub_string})')
    log.info(f'  overwrite       : {args.overwrite}')
    log.info(f'  dry_run         : {args.dry_run}')
    log.info('Models to process:')

    to_fit = []
    for m in args.models:
        done = already_done(m, args.format, remove_sub_string)
        status = 'EXISTS' if done else 'PENDING'
        will_run = (not done) or args.overwrite
        action = 'RUN' if will_run else 'SKIP'
        log.info(f'    model-{m:<8}  status={status:<7}  action={action}')
        if will_run:
            to_fit.append(m)

    log.info(f'  → {len(to_fit)} model(s) to fit: {to_fit}')
    log.info('=' * 60)

    if args.dry_run:
        log.info('Dry run — exiting without fitting.')
        return

    if not to_fit:
        log.info('Nothing to do.')
        return

    # ------------------------------------------------------------------
    # Load data once
    # ------------------------------------------------------------------
    log.info('Loading behavioral data...')
    df_full = get_data(args.bids_folder)

    if remove_sub_string:
        remove_sub_list = [int(s) for s in remove_sub_string.split('-')]
        df_full = df_full[~df_full.index.get_level_values('subject').isin(remove_sub_list)]
        log.info(f'Removed subjects: {remove_sub_list}')
    else:
        log.info('Including all subjects (supplement version).')

    df = df_full.xs(args.format, level='format')
    log.info(f'Data loaded: {len(df.index.unique("subject"))} subjects, format={args.format}')

    # ------------------------------------------------------------------
    # Fit models sequentially
    # ------------------------------------------------------------------
    failed = []
    for i, model_label in enumerate(to_fit, 1):
        log.info(f'\n--- [{i}/{len(to_fit)}] Fitting model-{model_label} ---')
        try:
            fit_one(model_label, df, args.format,
                    remove_sub_string=remove_sub_string,
                    overwrite=args.overwrite)
        except Exception as e:
            log.error(f'[FAILED] model-{model_label}: {e}', exc_info=True)
            failed.append(model_label)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info('\n' + '=' * 60)
    log.info(f'Fitting complete. {len(to_fit) - len(failed)}/{len(to_fit)} succeeded.')
    if failed:
        log.warning(f'Failed models: {failed}')
    log.info('Traces saved to: ' + TARGET_FOLDER)
    log.info('=' * 60)


if __name__ == '__main__':
    main()
