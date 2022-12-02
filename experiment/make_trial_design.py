# runs as; python make_trial_design.py $subject 1 1
import argparse
import os.path as op
import os
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from itertools import product


def main(subject, session=None, run=None):

    ## make_trial_design (.tsv fiel already exists for risky-choice task from running calibration, needs to be created newly here)
    task_settings_folder = op.abspath(op.join('settings', 'task'))
    if not op.exists(task_settings_folder):
        os.makedirs(task_settings_folder)

    fn = op.abspath(op.join(task_settings_folder,
                            f'sub-{subject}_ses_task-magJudge.tsv'))

    frac = np.linspace(-6,6, 13)
    frac = np.delete(frac, 6)                     
    fractions = np.power(2,(frac/4))

    base = np.array([5, 7, 10, 14, 20])

    df = create_design_magJudge(fractions, base=base, repetitions=3, n_runs= 6)
    print(df)
    df.to_csv(fn, sep='\t')


def create_design_magJudge(fractions, base=[5, 7, 10, 14, 20], repetitions=3, n_runs=6):

    base = np.array(base)
    repetition = range(1, repetitions+1)

    df = []
    
    tmp = pd.DataFrame(product(base, fractions, repetition), columns=['base number', 'fraction', 'repetition'])

    df.append(tmp)

    df = pd.concat(df).reset_index(drop=True)

    df['n1'] = df['base number']
    df['n2'] = df['base number'].astype(int) * df['fraction']

    df['n1'] = df['n1'].astype(int)
    df['n2'] = df['n2'].astype(int)

    # note df['n1']==df['n2'] for small basenumbers (even though 1 (as frac=0) is removed from fractions)
            
    df = df.sample(frac=1)

    # Get run numbers
    runs = []
    for x in range(1,n_runs+1):
        runs.append(np.repeat(x,len(df)/n_runs))

    df['run'] = np.concatenate(runs)

    df = df.set_index(['run'])
    ixs = np.random.permutation(df.index.unique())
    df = df.loc[ixs].sort_index(
        level='run', sort_remaining=False)

    df['trial'] = np.arange(1, len(df)+1)

    return df    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, nargs='?')
    parser.add_argument('session', default=None, nargs='?')
    parser.add_argument('run', default=None, nargs='?')
    args = parser.parse_args()

    main(subject=args.subject, session=args.session, run=args.run)
