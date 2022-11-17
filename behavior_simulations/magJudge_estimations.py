#%% 
import os
import os.path as op
import pandas as pd
import numpy as np
import re

runs = range(1,7)

bids_folder = '/Users/mrenke/data/ds-adultnumrisk'
sub_list = [f for f in os.listdir(bids_folder) if re.match(r'sub*', f)]

#%% create all-runs reordered events files

for sub in sub_list:
    df = []
    for run in runs:    
        fn = op.join(bids_folder, sub, 'func', f'{sub}_task-numrisk_run-{run}_events.tsv')
        d = pd.read_csv(fn, sep='\t', index_col=['trial_type'])
        d['run'] = run 
        df.append(d)
    df = pd.concat(df).reset_index(drop=False)

    df_cl = pd.DataFrame(None, columns= ['n1','n2', 'choice','run'])
    i = 0
    for ix in range(0,len(df)):
        if df['trial_type'].iloc[ix][:5] == 'stim1':
            d = pd.DataFrame([[int(df['trial_type'].iloc[ix][6:]), np.NaN, np.NaN,df['run'].iloc[ix] ]], columns= ['n1','n2', 'choice','run'])
            df_cl = pd.concat([df_cl,d])
        elif df['trial_type'].iloc[ix][:5] == 'stim2':
            df_cl['n2'].iloc[i] = int(df['trial_type'].iloc[ix][6:])
        elif df['trial_type'].iloc[ix] == 'left': #n1
            df_cl['choice'].iloc[i] = 1
            i += 1
        elif df['trial_type'].iloc[ix] == 'right': #n2
            df_cl['choice'].iloc[i] = 2
            i += 1

    df_cl['ratio'] =   df_cl['n1']/df_cl['n2']      
    df_cl['log_ratio'] = df_cl['ratio'].apply(lambda d: np.log(d) if np.isnan(d)==False else np.NaN)
    df_cl['factor'] = df_cl['ratio'].apply(lambda d: round(np.log2(d)*4) if np.isnan(d)==False else np.NaN) #formular from Miguel's manuscript

    df_cl= df_cl.reset_index(drop=True)

    df_cl.to_csv(op.join(bids_folder, sub, 'func', f'{sub}_task-numrisk_all-runs_reordered_events.tsv'))

# %% analyse behavior 
# activate behav_fit
# reference" stressrisk/stressrisk/subject_selection/fit_rnp.ipynb
import pymc as pm

df = []
for sub in sub_list:
        fn = op.join(bids_folder, sub, 'func', f'{sub}_task-numrisk_all-runs_reordered_events.tsv')
        d = pd.read_csv(fn)
        d['subject'] = int(sub[4:])
        df.append(d)
df = pd.concat(df)

df = df.set_index('subject',append=True)
df = df.dropna()
#%%
from statistics import NormalDist
from aesara import tensor as at

def get_posterior(mu1, sd1, mu2, sd2):

    var1, var2 = sd1**2, sd2**2

    return mu1 + (var1/(var1+var2))*(mu2 - mu1), np.sqrt((var1*var2)/(var1+var2))

def get_diff_dist(mu1, sd1, mu2, sd2):
    return mu2 - mu1, np.sqrt(sd1**2+sd2**2)

def cumulative_normal(x, mu, sd, s=at.sqrt(2.)):
#     Cumulative distribution function for the standard normal distribution
    return at.clip(0.5 + 0.5 *
                   at.erf((x - mu) / (sd*s)), 1e-9, 1-1e-9)

def fit_model(d):

    n1_mu = np.log(d['n1'])
    n2_mu = np.log(d['n2'])

    choices = d['choice'] -1 # choose n2 = 1
    

    unique_subjects = d.index.unique(level='subject')
    n_subjects = len(unique_subjects)
    subject_ix = d.index.codes[1]
    coords = {"subject": unique_subjects}
    base_numbers = [5., 7., 10., 14., 20., 28.]

    mean_n1 = np.mean(np.log(base_numbers)) # same as np.mean(np.log(d['n1']))
    std_n1 = np.std(np.log(base_numbers))

    mean_n2 = np.mean(np.log(d['n2']))
    std_n2 = np.std(np.log(d['n2']))
    
    with pm.Model(coords=coords) as model:

        n2_prior_mu = mean_n2
        n2_prior_sd = std_n2

        n1_prior_mu = mean_n1
        n1_prior_sd = std_n1

        n1_evidence_sd_mu = pm.HalfNormal("n1_evidence_sd_mu", sigma=1.)
        n1_evidence_sd_sd = pm.HalfCauchy("n1_evidence_sd_sd", 1.)
        n1_evidence_sd = pm.TruncatedNormal('n1_evidence_sd',
                                          mu=n1_evidence_sd_mu,
                                          sigma=n1_evidence_sd_sd,
                                          lower=0,
                                          dims=('subject'))

        n2_evidence_sd_mu = pm.HalfNormal("n2_evidence_sd_mu", sigma=1.)
        n2_evidence_sd_sd = pm.HalfCauchy("n2_evidence_sd_sd", 1.)
        n2_evidence_sd = pm.TruncatedNormal('n2_evidence_sd',
                                          mu=n2_evidence_sd_mu,
                                          sigma=n2_evidence_sd_sd,
                                          lower=0,
                                          dims=('subject'))                                  


        post_n1_mu, post_n1_sd = get_posterior(n1_prior_mu, 
                                                n1_prior_sd, 
                                                n1_mu,
                                                n1_evidence_sd[subject_ix])


        post_n2_mu, post_n2_sd = get_posterior(n2_prior_mu, 
                                                n2_prior_sd,
                                                n2_mu,
                                                n2_evidence_sd[subject_ix])

        diff_mu, diff_sd = get_diff_dist(post_n1_mu, post_n1_sd, post_n2_mu, post_n2_sd) # mu2 - mu1

        p = cumulative_normal(at.log(1), diff_mu, diff_sd)

        ll = pm.Bernoulli('ll_bernoulli', p=p, observed=choices)



    with model:
        #trace = pm.sample(2000, tune=2000, target_accept=0.95, return_inferencedata=True)
        #trace = pm.sample(1000, tune=2000, target_accept=0.95, cores=1)
        trace = pm.sample(1000, tune=2000, init='adapt_diag', target_accept=0.8, cores=1)
    return trace

#%%
trace = fit_model(df)
#%%
import arviz as az

az.plot_trace(trace)
# %%
import seaborn as sns

tmp1 = trace.posterior['n1_evidence_sd'].to_dataframe()
tmp2 = trace.posterior['n2_evidence_sd'].to_dataframe()

tmp = tmp1.join(tmp2)
dd = tmp.groupby('subject').mean()

dd.
#%%
plt.scatter(dd['n1_evidence_sd'],dd['n2_evidence_sd'])
plt.ylabel('n1_evidence_sd')
plt.xlabel('n2_evidence_sd')
plt.show()

#%%
import matplotlib.pyplot as plt

fractions = 2**(np.linspace(-6,6,13)/4) # assumes weber's law

sns.catplot('subject', 'n1_evidence_sd', data=tmp.reset_index(), kind='violin')

plt.hist(df_cl['factor'], bins=13)

# %%
