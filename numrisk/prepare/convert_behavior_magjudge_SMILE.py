"""
Prepare behavioral data for the magjudge task as run in Caroline's SMILE study.

The SMILE stimulus computer had a scanner-sync bug: 'pulse' and 'response'
events get logged on (almost) every frame, so raw files have ~30k rows for
~30 trials. The 'stim'/'choice' events themselves are unaffected and follow
the same structure as the dnumrisk magjudge sourcedata (see
convert_behavior_magjduge.py): per trial, phase 0-3 = stimulus 1 flicker
frames, phase 4-5 = stimulus 2, one 'choice' row at the end.

We don't need scanner-pulse alignment here: this is behavior-only prep, and
downstream RT computation (behavior_magjudge.utils.cleanup_behavior) only
ever takes onset differences within a run, so a constant per-run offset
(which real pulse alignment would give us anyway) cancels out. We use the
run's first stimulus-1 onset as that offset instead.

Two steps:
  1. copy_raw(): copy raw '*_events.tsv' files from the network share into
     <bids_folder>/sourcedata/behavior_magjudge/, renaming task-risk -> task-magjudge
     to match the dnumrisk sourcedata convention.
  2. convert(): turn each raw file into a 3-rows/trial events file
     (trial_type in {stimulus 1, stimulus 2, choice}) at
     <bids_folder>/sub-X/ses-Y/func/sub-X_ses-Y_task-magjudge_run-N_events.tsv,
     ready for behavior_magjudge.utils.get_behavior().
"""

import os
import os.path as op
import shutil
import argparse
import re
import glob

import numpy as np
import pandas as pd

RAW_SOURCE_DIR = "/Volumes/G_ADABD_Largefiles$/Data/SMILE_Data/measurements/magjudg_logfiles"
DEFAULT_BIDS_FOLDER = "/Users/mrenke/data/ds-smile"

RAW_FN_RE = re.compile(
    r"sub-(?P<subject>\d+)_ses-(?P<session>\d+)_task-risk_run-(?P<run>\d+)_events\.tsv$"
)


def copy_raw(bids_folder=DEFAULT_BIDS_FOLDER, source_dir=RAW_SOURCE_DIR):
    """Copy raw SMILE '*_events.tsv' files into <bids_folder>/sourcedata/behavior_magjudge/,
    renaming task-risk -> task-magjudge. Skips files that already exist locally."""

    target_root = op.join(bids_folder, "sourcedata", "behavior_magjudge")

    raw_files = sorted(glob.glob(op.join(source_dir, "sub-*", "ses-*", "*_events.tsv")))
    print(f"found {len(raw_files)} raw SMILE event files")

    n_copied, n_skipped = 0, 0
    for src in raw_files:
        m = RAW_FN_RE.search(op.basename(src))
        if m is None:
            print(f"  skipping (unexpected filename): {src}")
            continue

        subject, session, run = m["subject"], m["session"], m["run"]
        target_dir = op.join(target_root, f"sub-{subject}", f"ses-{session}")
        target_fn = op.join(
            target_dir,
            f"sub-{subject}_ses-{session}_task-magjudge_run-{run}_events.tsv",
        )

        if op.exists(target_fn):
            n_skipped += 1
            continue

        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(src, target_fn)
        n_copied += 1

    print(f"copied {n_copied} files, skipped {n_skipped} already present")


def _extract_run_events(behavior):
    """Given a raw magjudge behavior dataframe for one run, return a 3-rows/trial
    dataframe (trial_nr, onset, trial_type, n1, n2, choice), or None if no
    complete trials are found."""

    behavior = behavior.copy()
    behavior["trial_nr"] = pd.to_numeric(behavior["trial_nr"], errors="coerce")
    behavior["phase"] = pd.to_numeric(behavior["phase"], errors="coerce")
    behavior = behavior.dropna(subset=["trial_nr", "phase"])
    behavior = behavior[behavior["trial_nr"] > 0]  # drop practice/instruction block

    stim1 = behavior[(behavior["event_type"] == "stim") & (behavior["phase"] == 2)].copy()
    stim1["trial_type"] = "stimulus 1"

    stim2 = behavior[(behavior["event_type"] == "stim") & (behavior["phase"] == 4)].copy()
    stim2["trial_type"] = "stimulus 2"

    choice = behavior[behavior["event_type"] == "choice"].copy()
    choice["trial_type"] = "choice"

    # only keep trials where we have all three pieces (drops truncated last
    # trials and stray spurious rows with an isolated/garbage trial_nr)
    complete_trials = set(stim1["trial_nr"]) & set(stim2["trial_nr"]) & set(choice["trial_nr"])
    if not complete_trials:
        return None

    stim1 = stim1[stim1["trial_nr"].isin(complete_trials)]
    stim2 = stim2[stim2["trial_nr"].isin(complete_trials)]
    choice = choice[choice["trial_nr"].isin(complete_trials)]

    # a trial should have exactly one row per piece; keep the first if not
    stim1 = stim1.drop_duplicates("trial_nr", keep="first")
    stim2 = stim2.drop_duplicates("trial_nr", keep="first")
    choice = choice.drop_duplicates("trial_nr", keep="first")

    events = pd.concat((stim1, stim2, choice)).sort_values(["trial_nr", "onset"]).reset_index(drop=True)

    t0 = stim1["onset"].min()  # run-relative zero point; no real pulses to align to
    events["onset"] -= t0

    events["trial_nr"] = events["trial_nr"].astype(int)
    events = events[["trial_nr", "onset", "trial_type", "n1", "n2", "choice"]]

    return events, len(complete_trials)


def convert(bids_folder=DEFAULT_BIDS_FOLDER, subject_list=None):
    """Convert raw sourcedata magjudge behavior files into 3-rows/trial events
    files under <bids_folder>/sub-X/ses-Y/func/."""

    source_root = op.join(bids_folder, "sourcedata", "behavior_magjudge")
    raw_files = sorted(glob.glob(op.join(source_root, "sub-*", "ses-*", "*_events.tsv")))

    if subject_list is not None:
        raw_files = [f for f in raw_files if any(f"sub-{s}" in f for s in subject_list)]

    print(f"converting {len(raw_files)} sourcedata files")

    fn_re = re.compile(
        r"sub-(?P<subject>\d+)_ses-(?P<session>\d+)_task-magjudge_run-(?P<run>\d+)_events\.tsv$"
    )

    for src in raw_files:
        m = fn_re.search(op.basename(src))
        subject, session, run = m["subject"], m["session"], m["run"]

        behavior = pd.read_table(src)
        result = _extract_run_events(behavior)

        if result is None:
            print(f"  sub-{subject} ses-{session} run-{run}: no complete trials, skipping")
            continue

        events, n_trials = result

        target_dir = op.join(bids_folder, f"sub-{subject}", f"ses-{session}", "func")
        os.makedirs(target_dir, exist_ok=True)
        target_fn = op.join(
            target_dir,
            f"sub-{subject}_ses-{session}_task-magjudge_run-{run}_events.tsv",
        )
        events.to_csv(target_fn, index=False, sep="\t")
        print(f"  sub-{subject} ses-{session} run-{run}: {n_trials} trials -> {target_fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_folder", default=DEFAULT_BIDS_FOLDER)
    parser.add_argument("--source_dir", default=RAW_SOURCE_DIR)
    parser.add_argument("--subject", nargs="*", default=None, help="restrict to these subject IDs (e.g. 101 203)")
    parser.add_argument("--skip_copy", action="store_true", help="skip copying raw files from the network share")
    args = parser.parse_args()

    if not args.skip_copy:
        copy_raw(args.bids_folder, args.source_dir)
    convert(args.bids_folder, args.subject)
