import os.path as op
import os
import numpy as np
import pandas as pd
#from bauer.models import RiskRegressionModel
#from stress_risk.utils.data import get_all_behavior
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss



def cleanup_behavior(df_, drop_no_responses=True):  
        #df = df_
        df = df_[[]].copy()    
        df['choice'] = df_[('choice', 'choice')]
        df['n1'], df['n2'] = df_[('n1','stimulus')], df_[('n2','stimulus')]
        df['prob1'], df['prob2'] = df_[('prob1','stimulus')], df_[('prob1','stimulus')]

        df['risky_left'] = df_[('prob1', 'stimulus')] == 0.55
        df['chose_risky'] = (df['risky_left'] & (df['choice'] == 1.0)) | (~df['risky_left'] & (df['choice'] == 2.0))
        df.loc[df.choice.isnull(), 'chose_risky'] = np.nan

        df['n_risky'] = df['n1'].where(df['risky_left'], df['n2'])
        df['n_safe'] = df['n2'].where(df['risky_left'], df['n1'])
        df['frac'] = df['n_risky'] / df['n_safe']
        df['log(risky/safe)'] = np.log(df['frac'])

        df['log(n1)'] = np.log(df['n1'])
        
        if drop_no_responses:
            df = df[~df.chose_risky.isnull()]
            df['chose_risky'] = df['chose_risky'].astype(bool)
        return df


def get_behavior(subject_list, bids_folder = '/Users/mrenke/data/ds-dnumrisk'):
    df_all = []
    session = 1

    for subject in subject_list:
        #print(subject)
        df = []
        for format in ['non-symbolic', 'symbolic']:

            fn = op.join(bids_folder, f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-risk_{format}_events.tsv')

            if op.exists(fn):
                d = pd.read_csv(fn, sep='\t',
                            index_col=['trial_nr', 'trial_type'])
                d['subject'], d['session'], d['format'] = int(subject), session, format
                df.append(d)

        df = pd.concat(df)
        df = df.drop([0, 999])
        df = df.reset_index().set_index(['subject', 'session', 'format', 'trial_nr', 'trial_type']) 
        df = df.unstack('trial_type')

        df = cleanup_behavior(df)
        df_all.append(df)

    df_all = pd.concat(df_all) 
    df_all.columns = df_all.columns.droplevel(1) #weird multiindex with "trial_type"

    return df_all

def invprobit(x):
    return ss.norm.ppf(x)

def extract_rnp_precision(trace, model, data, format=False):

    data = data.reset_index()

    if format:
        fake_data = pd.MultiIndex.from_product([data.reset_index()['subject'].unique(),
                                                [0, 1],
                                                data['n_safe'].unique(),
                                                data['format'].unique()],
                                                names=['subject', 'x', 'n_safe', 'format']
                                                ).to_frame().reset_index(drop=True)
    else:
        fake_data = pd.MultiIndex.from_product([data.reset_index()['subject'].unique(),
                                                [0, 1],
                                                data['n_safe'].unique(),
                                                [False, True]],
                                                names=['subject', 'x', 'n_safe', 'risky_left']
                                                ).to_frame().reset_index(drop=True)

    pred = model.predict(trace, 'mean', fake_data, inplace=False)['posterior']['chose_risky_mean']

    pred = pred.to_dataframe().unstack([0, 1])
    pred = pred.set_index(pd.MultiIndex.from_frame(fake_data))

    # return pred

    pred0 = pred.xs(0, 0, 'x')
    intercept = pd.DataFrame(invprobit(pred0), index=pred0.index, columns=pred0.columns)
    gamma = invprobit(pred.xs(1, 0, 'x')) - intercept

    return intercept, gamma