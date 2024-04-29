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

    df['choice'] = df['chose_n2']
    return df


def get_subjects(bids_folder='/Users/mrenke/data/ds-dnumrisk'):
    #subjects = list(range(1, 200))
    #sub_folders = [f[3:] for f in listdir(bids_folder) if f[:3] == 'sub']

    sub_folders = [int(f[4:]) for f in listdir(bids_folder) if f[:3] == 'sub' and len(f) == 6]
    subjects = [Subject(subject, bids_folder) for subject in sub_folders]

    return subjects

def get_all_behavior(bids_folder='/Users/mrenke/data/ds-dnumrisk'):
    subjects = get_subjects(bids_folder)
    behavior = [s.get_behavior(drop_no_responses=True) for s in subjects]

    return pd.concat(behavior)


class Subject(object):

    def __init__(self, subject, bids_folder='/Users/mrenke/data/ds-dnumrisk'):

        self.subject = '%02d' % int(subject)
        self.bids_folder = bids_folder

    def get_behavior(self, session=1, drop_no_responses=True):


        df = pd.DataFrame()
   
        runs = range(1, 7)
        for run in runs:

            fn = op.join(self.bids_folder, f'sub-{self.subject}/ses-{session}/func/sub-{self.subject}_ses-{session}_task-magjudge_run-{run}_events.tsv')

            if op.exists(fn):
                d = pd.read_csv(fn, sep='\t',
                            index_col=['trial_nr', 'trial_type'])
                d['subject'], d['run'] = int(self.subject), run 
                d = d.drop([0])
                df = pd.concat([df, d])


        df = df.reset_index().set_index(['subject','run','trial_type', 'trial_nr']) 
        df = df.unstack('trial_type')

        return self._cleanup_behavior(df)
        
    @staticmethod #It cannot access either class attributes or instance attributes.
    def _cleanup_behavior(df_):
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


from itertools import product

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

from scipy.stats import bernoulli, norm

def get_posterior(mu1, sd1, mu2, sd2):

    var1, var2 = sd1**2, sd2**2

    return mu1 + (var1/(var1+var2))*(mu2 - mu1), np.sqrt((var1*var2)/(var1+var2))

def get_diff_dist(mu1, sd1, mu2, sd2):
    return mu2 - mu1, np.sqrt(sd1**2+sd2**2)

def simulate_choices_subind(trials_df, sub_param):
    
    n1_prior_mu = np.mean(np.log(trials_df['n1']))
    n1_prior_sd = np.std(np.log(trials_df['n1']))

    n2_prior_sd = np.std(np.log(trials_df['n2']))

    choices = []
    #for r in trials_df.iterrows():
    for i in range(0,len(trials_df)):
        r = trials_df.iloc[i]

        n1_evidence_mu = np.log(r['n1'])
        n2_evidence_mu = np.log(r['n2'])
       
        n1_evidence_sd = sub_param.loc[r['subject'],'n1_evidence_sd']
        n2_evidence_sd = sub_param.loc[r['subject'],'n2_evidence_sd']

        n2_prior_mu = sub_param.loc[r['subject'],'n2_prior_mu']


        post_n1_mu, post_n1_sd = get_posterior(n1_prior_mu, 
                                                n1_prior_sd, 
                                                n1_evidence_mu, 
                                                n1_evidence_sd
                                                )

        post_n2_mu, post_n2_sd = get_posterior(n2_prior_mu,
                                                n2_prior_sd,
                                                n2_evidence_mu, 
                                                n2_evidence_sd)

        diff_mu, diff_sd = get_diff_dist( post_n1_mu, post_n1_sd, post_n2_mu, post_n2_sd)

        p = norm.cdf(diff_mu, scale = diff_sd)
        choice = int(np.random.binomial(1, p, 1)) # 
        
        choices.append(choice)

    trials_df['sim_choice'] = choices

    return trials_df