
from itertools import product
import numpy as np
import pandas as pd
from scipy.stats import norm


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

def get_posterior(mu1, sd1, mu2, sd2):

    var1, var2 = sd1**2, sd2**2

    return mu1 + (var1/(var1+var2))*(mu2 - mu1), np.sqrt((var1*var2)/(var1+var2))

def get_diff_dist(mu1, sd1, mu2, sd2):
    return mu2 - mu1, np.sqrt(sd1**2+sd2**2)


def simulate_choices(trials_df, sub_param):
    
    n1_prior_mu = np.mean(np.log(trials_df['n1']))
    n1_prior_sd = np.std(np.log(trials_df['n1']))

    n2_prior_sd = np.std(np.log(trials_df['n2']))

    choices = []
    #for r in trials_df.iterrows():
    for i in range(0,len(trials_df)):
        r = trials_df.iloc[i]

        n1_evidence_mu = np.log(r['n1'])
        n2_evidence_mu = np.log(r['n2'])

        n1_evidence_sd = sub_param[sub_param['subject'] == r['subject']]['n1_evidence_sd']
        n2_evidence_sd = sub_param[sub_param['subject'] == r['subject']]['n2_evidence_sd']

        n2_prior_mu = sub_param[sub_param['subject'] == r['subject']]['n2_prior_mu']

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