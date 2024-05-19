import os.path as op
import os
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from os import listdir

def cleanup_behavior(df_):
    df = df_[[]].copy()
    df['rt'] = df_.loc[:, ('onset', 'choice')] - df_.loc[:, ('onset', 'stimulus 2')]
    df['n1'], df['n2'] = df_['n1']['stimulus 1'], df_['n2']['stimulus 1']

    df['choice'] = df_[('choice', 'choice')]
    df['chose_n2'] =  (df['choice'] == 2.0)

    #df.loc[df.choice.isnull(), 'chose_risky'] = np.nan

    df['frac'] = df['n2'] / df['n1']
    df['log(n2/n1)'] = np.log(df['frac'])

    df['log(n1)'] = np.log(df['n1'])
    df = df.droplevel(-1,1)
    
    return df

def get_behavior(subject_list=None, bids_folder = '/Users/mrenke/data/ds-dnumrisk'):
    df_all = []
    session = 1
    runs = range(1, 7)

    #if subject_list is None:
    subject_list = [f[4:] for f in listdir(bids_folder) if f[0:3] == 'sub' and len(f) == 6]
    print(f'number of subjects found: {len(np.sort(subject_list))}')

    for subject in subject_list:    
        df_sub = []
        for run in runs:
            fn = op.join(bids_folder, f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-magjudge_run-{run}_events.tsv')
            if op.exists(fn):
                d = pd.read_csv(fn, sep='\t',
                            index_col=['trial_nr', 'trial_type'])
                d['subject'], d['run'] = int(subject), run 
                #d = d.drop([0])
                df_sub.append(d)
        
        #df_sub = pd.concat([df_sub, d])
        df_sub = pd.concat(df_sub)
        df_sub = df_sub.reset_index().set_index(['subject','run','trial_type', 'trial_nr']) 
        df_sub = df_sub.unstack('trial_type')
        df_sub = cleanup_behavior(df_sub)
        df_all.append(df_sub)

    df_all = pd.concat(df_all) 
    return df_all


def get_data(bids_folder='/Users/mrenke/data/ds-dnumrisk', subject_list=None):
    df = get_behavior(subject_list, bids_folder=bids_folder)

    df_participants = pd.read_csv(op.join('/Users/mrenke/data/ds-dnumrisk/add_tables','subjects_recruit&scan_scanned-final.csv'), header=0) #, index_col=0
    df_participants = df_participants.loc[:,['subject ID', 'age','group','gender']].rename(mapper={'subject ID': 'subject'},axis=1).dropna().astype({'subject': int, 'group': int}).set_index('subject')

    df = df.join(df_participants['group'], on='subject',how='left') # takes only the subs fro df_paricipants that are in the df
    df = df.dropna() # automatially removes subs without group assignment
    n_subs = len(df.index.unique('subject').sort_values())
    print(f'number of subjects in dataframe: {n_subs}')
    print(df.index.unique('subject').sort_values())

    df['choice'] = df['chose_n2']
    return df


def invprobit(x):
    return ss.norm.ppf(x)

def extract_rnp_precision(trace, model, data, group=False):

    data = data.reset_index()

    reg_list = [data.reset_index()['subject'].unique(),[0, 1], data['n1'].unique()]
    names=['subject', 'x', 'n1']
    
    if group:
        reg_list.append(data['group'].unique())
        names.append('group')   

    fake_data = pd.MultiIndex.from_product(reg_list,names=names).to_frame().reset_index(drop=True)
    include_group_specific = not group
    pred = model.predict(trace, 'mean', fake_data, inplace=False, include_group_specific=include_group_specific)['posterior']['chose_n2_mean']

    pred = pred.to_dataframe().unstack([0, 1])
    pred = pred.set_index(pd.MultiIndex.from_frame(fake_data))

    # return pred

    pred0 = pred.xs(0, 0, 'x')
    intercept = pd.DataFrame(invprobit(pred0), index=pred0.index, columns=pred0.columns)
    gamma = invprobit(pred.xs(1, 0, 'x')) - intercept

    return intercept, gamma

