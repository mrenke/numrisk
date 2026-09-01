"""
Combine per-run SMILE magjudge events files (produced by
convert_behavior_magjudge_SMILE.py) into one behavior table for the whole
study, with the same column names/order as
/Users/mrenke/data/ds-dnumrisk/magjudge_behavior_DNumRisk.csv, plus an added
'session' column (SMILE subjects have 1-3 sessions, unlike dnumrisk's single
session).

Group is derived from the subject ID: subjects >= 300 are control,
subjects < 300 are dyscalculia (per Maike). Age is not available from the
SMILE sourcedata on hand, so it's left as NaN.
"""

import os.path as op
import argparse
import glob
import re

import numpy as np
import pandas as pd

from numrisk.behavior_magjudge.utils import cleanup_behavior

DEFAULT_BIDS_FOLDER = "/Users/mrenke/data/ds-smile"

FN_RE = re.compile(
    r"sub-(?P<subject>\d+)_ses-(?P<session>\d+)_task-magjudge_run-(?P<run>\d+)_events\.tsv$"
)


def get_behavior_smile(bids_folder=DEFAULT_BIDS_FOLDER, sessions="all"):
    """Load and combine all subjects' magjudge events files.

    sessions: 'all' to use every session found per subject, or a list of
    session numbers to restrict to (e.g. [1], [2, 3])."""

    files = sorted(glob.glob(op.join(bids_folder, "sub-*", "ses-*", "func", "*_events.tsv")))

    by_subject = {}
    for fn in files:
        m = FN_RE.search(op.basename(fn))
        if m is None:
            continue
        subject, session, run = int(m["subject"]), int(m["session"]), int(m["run"])

        if sessions != "all" and session not in sessions:
            continue

        d = pd.read_csv(fn, sep="\t", index_col=["trial_nr", "trial_type"])
        d["subject"], d["session"], d["run"] = subject, session, run
        by_subject.setdefault(subject, []).append(d)

    print(f"found {len(by_subject)} subjects")

    df_all = []
    for subject, parts in sorted(by_subject.items()):
        df_sub = pd.concat(parts)
        df_sub = df_sub.reset_index().set_index(["subject", "session", "run", "trial_type", "trial_nr"])
        df_sub = df_sub.unstack("trial_type")
        df_sub = cleanup_behavior(df_sub)
        df_all.append(df_sub)

    df_all = pd.concat(df_all)
    return df_all


def main(bids_folder=DEFAULT_BIDS_FOLDER, sessions="all"):
    df = get_behavior_smile(bids_folder, sessions)
    df = df.reset_index()

    df["n1"] = df["n1"].astype(int)
    df["group"] = np.where(df["subject"] >= 300, "control", "dyscalc")
    df["age"] = np.nan

    df["correct_answer_n2"] = df["n2"] > df["n1"]
    df["correct"] = df["correct_answer_n2"] == df["chose_n2"]

    df["choice"] = df["chose_n2"]

    df = df[
        [
            "subject", "session", "run", "trial_nr", "rt", "n1", "n2", "choice", "chose_n2",
            "frac", "log(n2/n1)", "log(n1)", "group", "age", "correct_answer_n2", "correct",
        ]
    ]

    if sessions == "all":
        tag = "allses"
    else:
        tag = "ses" + "".join(str(s) for s in sorted(sessions))

    fn = op.join(bids_folder, f"magjudge_behavior_SMILE-{tag}.csv")
    df.to_csv(fn, index=False)
    print(f"wrote {len(df)} rows, {df['subject'].nunique()} subjects -> {fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_folder", default=DEFAULT_BIDS_FOLDER)
    parser.add_argument(
        "--sessions", nargs="*", default=["1"],
        help="session numbers to include (e.g. --sessions 1, --sessions 2 3), or 'all' for every session found",
    )
    args = parser.parse_args()

    sessions = "all" if [s.lower() for s in args.sessions] == ["all"] else [int(s) for s in args.sessions]
    main(args.bids_folder, sessions)
