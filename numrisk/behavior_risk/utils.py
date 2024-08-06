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
from os import listdir


def cleanup_behavior(df_, drop_no_responses=True):  
        #df = df_
        df = df_[[]].copy()    
        df['choice'] = df_[('choice', 'choice')]
        df['n1'], df['n2'] = df_[('n1','stimulus')], df_[('n2','stimulus')]
        df['prob1'], df['prob2'] = df_[('prob1','stimulus')], df_[('prob2','stimulus')]

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

        def get_risk_bin(d):
            labels = [f'{int(e)}%' for e in np.linspace(20, 80, 6)]
            try: 
                # return pd.qcut(d, 6, range(1, 7))
                return pd.qcut(d, 6, labels=labels)
            except Exception as e:
                n = len(d)
                ix = np.linspace(0, 6, n, False)

                d[d.sort_values().index] = [labels[e] for e in np.floor(ix).astype(int)]
                
                return d
            
        df['bin(risky/safe)'] = df.groupby(['subject'], group_keys=False)['frac'].apply(get_risk_bin)

        return df


def get_behavior(subject_list=None, bids_folder = '/Users/mrenke/data/ds-dnumrisk', formats=['non-symbolic', 'symbolic']):
    df_all = []
    session = 1

    if subject_list is None:
        subject_list = [f[4:] for f in listdir(bids_folder) if f[0:3] == 'sub' and len(f) == 6]
        #print(f'number of subjects found: {len(np.sort(subject_list))}') # only number of folders, not this specific data

    for subject in subject_list:
        format = 'non-symbolic'
        fn = op.join(bids_folder, f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-risk_{format}_events.tsv')
        if op.exists(fn):
            df = []
            for format in formats:
                fn = op.join(bids_folder, f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-risk_{format}_events.tsv')
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
        else:
            print(f'sub-{subject} failed: File not found: {fn}')

    df_all = pd.concat(df_all) 
    df_all.columns = df_all.columns.droplevel(1) #weird multiindex with "trial_type"

    return df_all

def get_data(bids_folder='/Users/mrenke/data/ds-dnumrisk', subject_list=None):
    df = get_behavior(subject_list, bids_folder=bids_folder)

    # make colum settings compatible for bauer model fitting
    df['choice'] = df['chose_risky'] #df['choice'] == 2.0
    df['p1'] = 1 #df['prob1']
    df['p2'] = 0.55 #df['prob2']
    df['n1'] = df['n_safe']
    df['n2'] = df['n_risky']

    df_participants = pd.read_csv(op.join('/Users/mrenke/data/ds-dnumrisk/add_tables','subjects_recruit_scan_scanned-final.csv'), header=0) #, index_col=0
    df_participants = df_participants.loc[:,['subject ID', 'age','group','gender']].rename(mapper={'subject ID': 'subject'},axis=1).dropna().astype({'subject': int, 'group': int}).set_index('subject')
    #df_participants=df_participants.loc[1:42,:]

    df = df.join(df_participants['group'], on='subject',how='left') # takes only the subs fro df_paricipants that are in the df
    df = df.dropna() # automatially removes subs without group assignment
    n_subs = len(df.index.unique('subject').sort_values())
    print(f'number of subjects in dataframe: {n_subs}')
    print(df.index.unique('subject').sort_values())
    
    return df

def invprobit(x):
    return ss.norm.ppf(x)

def extract_rnp_precision_old(trace, model, data, format=False):

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

def extract_rnp_precision(trace, model, data, format=True, group_level = False):

    data = data.reset_index()

    reg_list = [data.reset_index()['subject'].unique(),[0, 1], data['n1'].unique(), data['group'].unique()]
    names=['subject', 'x', 'n_safe','group']

    if format:
        reg_list.append(data['format'].unique())
        names.append('format')

    if group_level: # needed 
        include_group_specific = None
    else:     # when no subjects! include_group_specific=False
        include_group_specific = True 

    fake_data = pd.MultiIndex.from_product(reg_list,names=names).to_frame().reset_index(drop=True)

    pred = model.predict(trace, 'mean', fake_data, inplace=False, include_group_specific=include_group_specific)['posterior']['chose_risky_mean']

    pred = pred.to_dataframe().unstack([0, 1])
    pred = pred.set_index(pd.MultiIndex.from_frame(fake_data))

    # return pred

    pred0 = pred.xs(0, 0, 'x')
    intercept = pd.DataFrame(invprobit(pred0), index=pred0.index, columns=pred0.columns)
    gamma = invprobit(pred.xs(1, 0, 'x')) - intercept

    return intercept, gamma

