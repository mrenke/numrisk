#%%
import os.path as op
import pandas as pd

bids_folder = '/Users/mrenke/data/ds-adultnumrisk'

df = pd.read_csv(op.join(bids_folder,'derivatives', 'summary_data','magData.csv'))
df['subject'] = df['subcode']
df['n1'] = df['sure_bet']
df['n2'] = df['prob_bet']
df['choice'] = df['leftright'].map({-1:1, 1:2})
df['subject'] = pd.Categorical(df['subject'])
df = df.set_index('subject')

# %%
import pymc as pm

def get_posterior(mu1, sd1, mu2, sd2):
    var1, var2 = sd1**2, sd2**2
    return mu1 + (var1/(var1+var2))*(mu2 - mu1), np.sqrt((var1*var2)/(var1+var2))
def get_diff_dist(mu1, sd1, mu2, sd2):
    return mu2 - mu1, np.sqrt(sd1**2+sd2**2)
def cumulative_normal(x, mu, sd, s=at.sqrt(2.)):
#     Cumulative distribution function for the standard normal distribution
    return at.clip(0.5 + 0.5 *
                   at.erf((x - mu) / (sd*s)), 1e-9, 1-1e-9)

class MagnitudeComparisonModel(object):
    
    def __init__(self, data):
        self.data = data
        self.subject_ix, self.unique_subjects = pd.factorize(self.data.index.get_level_values('subject'))
        self.n_subjects = len(self.unique_subjects) 
            
    def build_model(self):

        base_numbers = self.data.n1.unique()
        choices = self.data.choice == 2.
        
        mean_n1 = np.mean(np.log(base_numbers)) # same as np.mean(np.log(d['n1']))
        std_n1 = np.std(np.log(base_numbers))
        mean_n2 = np.mean(np.log(self.data['n2']))
        std_n2 = np.std(np.log(self.data['n2']))
        
        n2_prior_mu = mean_n2
        n2_prior_sd = std_n2
        n1_prior_mu = mean_n1
        n1_prior_sd = std_n1
        
        n1_mu = np.log(self.data['n1'])
        n2_mu = np.log(self.data['n2'])

        self.coords = {
            "subject": self.unique_subjects}

                                              
        with pm.Model(coords=self.coords) as self.model:
                
            def build_hierarchical_nodes(name, mu=0.0, sigma=.5):
                nodes = {}

                nodes[f'{name}_mu'] = pm.Normal(f"{name}_mu", 
                                              mu=mu, 
                                              sigma=sigma)
                
                nodes[f'{name}_sd'] = pm.HalfCauchy(f'{name}_sd', .25)
                nodes[f'{name}_offset'] = pm.Normal(f'{name}_offset', mu=0, sigma=1, dims=('subject',))
                nodes[f'{name}_untransformed'] = pm.Deterministic(name, nodes[f'{name}_mu'] + nodes[f'{name}_sd'] * nodes[f'{name}_offset'],
                                              dims=('subject',))
                
                nodes[f'{name}_trialwise'] = at.softplus(nodes[f'{name}_untransformed'][self.subject_ix])
                
                return nodes

            # Hyperpriors for group nodes
            
            nodes = {}
            
            nodes.update(build_hierarchical_nodes('evidence_sd1'), mu=-1.)
            nodes.update(build_hierarchical_nodes('evidence_sd2'), mu=-1.)
            
            evidence_sd = at.stack((nodes['evidence_sd1_trialwise'], nodes['evidence_sd2_trialwise']), axis=0)


            post_n1_mu, post_n1_sd = get_posterior(n1_prior_mu, 
                                                    n1_prior_sd, 
                                                    n1_mu,
                                                    evidence_sd[0, self.subject_ix])
            post_n2_mu, post_n2_sd = get_posterior(n2_prior_mu, 
                                                    n2_prior_sd,
                                                    n2_mu,
                                                    evidence_sd[1, self.subject_ix])

            diff_mu, diff_sd = get_diff_dist(post_n1_mu, post_n1_sd, post_n2_mu, post_n2_sd)
            p = cumulative_normal(at.log(1), diff_mu, diff_sd)
            ll = pm.Bernoulli('ll_bernoulli', p=p, observed=choices)
            
    def sample(self, draws=1000, tune=1000, target_accept=0.8):
        
        with self.model:
            self.trace = pm.sample(draws, tune=tune, target_accept=target_accept, core = 1)
        
        return self.trace            

# %%
magjudge = MagnitudeComparisonModel(df)
magjudge.build_model()
trace = magjudge.sample()

# %%
import seaborn as sns
import numpy as np
def softplus(x): 
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0) 

evidence_sd1 = softplus(trace.posterior['evidence_sd1_mu'].to_dataframe())
evidence_sd2 = softplus(trace.posterior['evidence_sd2_mu'].to_dataframe())

sns.distplot(evidence_sd1)
sns.distplot(evidence_sd2)