import argparse
import os.path as op
import os
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from itertools import product


def create_design(prob1, prob2, fractions,
                  base=[7, 10, 14, 20, 28], repetitions=1):

    base = np.array(base)
    repetition = range(1, repetitions+1)

    df = []
    for ix, p in enumerate(prob1):
        tmp = pd.DataFrame(product(base, fractions, repetition), columns=[
                           'base number', 'fraction', 'repetition'])
        tmp['p1'] = p
        tmp['p2'] = prob2[ix]

        df.append(tmp)

    df = pd.concat(df).reset_index(drop=True) # all p1 = 55%, then all p1=100%, not so good?! 

    df.loc[df['p1'] == 1.0, 'n1'] = df['base number']
    df.loc[df['p1'] != 1.0, 'n2'] = df['base number']
    df.loc[df['p1'] == 1.0, 'n2'] = (
        df['fraction'] * df['base number']).astype(int)
    df.loc[df['p1'] != 1.0, 'n1'] = (
        df['fraction'] * df['base number']).astype(int)

    df['n1'] = df['n1'].astype(int)
    df['n2'] = df['n2'].astype(int)

    # Shuffle _within_ p1's
    df = df.groupby('p1', as_index=False).apply(
        lambda d: d.sample(frac=1)).reset_index(level=0, drop=True)

    df['trial'] = np.arange(1, len(df)+1)

    return df


def makeDesign(subject, session=None):

    x_lower, x_upper = 0.1 , 0.9# fit_psychometric_curve(log_file, plot=True)

    print(x_lower, x_upper)

    fractions = np.exp(np.linspace(np.log(x_lower), np.log(x_upper), 6, True))
    base = np.array([7, 10, 14, 20, 28])
    prob1 = [1., .55]
    prob2 = [.55, 1.]

    task_settings_folder = op.abspath(op.join('settings', 'task'))
    if not op.exists(task_settings_folder):
        os.makedirs(task_settings_folder)

    fn = op.abspath(op.join(task_settings_folder,
                            #f'sub-{subject}_ses-{session}_task-risk.tsv'))
                            f'sub-{subject}_ses_task-risk.tsv'))

    df = create_design(prob1, prob2, fractions, base=base, repetitions=2)

    print(df)

    df.to_csv(fn, sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, nargs='?')
    parser.add_argument('session', default=None, nargs='?')
    args = parser.parse_args()

    makeDesign(subject=args.subject, session=args.session)
