#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:44:47 2022

@author: mrenke
"""
import scipy.stats as ss
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


pati = '/Users/mrenke/Desktop/dyscalc_study/modeling_PacmanData'

cd '/Users/mrenke/Desktop/dyscalc_study/modeling_PacmanData'

data_oldformat = pd.read_spss('Data_Pacman.sav')

df = pd.read_csv('NMB_merged.txt', sep=",") #, header=None

ix_small = df['Condition.paramod'] == 1.0
ix_medium = df['Condition.paramod'] == 2.0
ix_large = df['Condition.paramod'] == 3.0
ix_control = df['Condition.paramod'] == 0.0

df = df.loc[~ix_control,] # remove control condition

df.loc[ix_small,'ratio'] = 0.91 #hardest --> 0 on Images.ACC = false response
df.loc[ix_medium,'ratio'] = 0.83
df.loc[ix_large,'ratio'] = 0.7


df['log_ratio'] = -np.log(df['ratio'])


#sns.lmplot('log_ratio', 'Images.ACC', data=df, logistic=True)

# filter out excluded subjects
Subs_exclude = np.array(data_oldformat.loc[data_oldformat['Einschluss'] == 'nein', 'VP'])
sub_ex = []
for sub in Subs_exclude:
    ix_ex = df['Subject'] == int(sub)
    df = df.loc[~ix_ex,] 
    sub_ex.append(int(sub))
    
sub_IDs = df['Subject'].unique()
N_subs = np.shape(data_oldformat['VP'])[0]

group = []
for sub in sub_IDs:
    for s in range(0,N_subs):
        num = int(data_oldformat.loc[s]['VP'])       
        if sub == num:
            group.append(data_oldformat.loc[s]['Group'])
    
            
for s in range(0,len(sub_IDs)):
    ix = df['Subject'] == sub_IDs[s]
    df.loc[ix, 'group'] = group[s]       


#
#%% MCMC model fit
import bambi as bmb
import arviz as az

model = bmb.Model("Images.ACC ~ 0 + log_ratio + (0+log_ratio|Subject) + (0+log_ratio|group)", df.reset_index(), 
                  family="bernoulli") #By default, the link function for this family is link="logit" /logit = invlogit


model_onlyGroup = bmb.Model("Images.ACC ~ 0 + log_ratio + (0+log_ratio|group)", df.reset_index(), 
                  family="bernoulli")


model

results = model.fit(2000, 3000, target_accept=0.975, init='adapt_diag')
results_onlyGroup = model_onlyGroup.fit(2000, 3000, target_accept=0.975, init='adapt_diag')


tmp = results['posterior']['log_ratio|group'].to_dataframe().unstack('group_coord_group_factor')['log_ratio|group']
sns.distplot(tmp['DD'] - tmp['CC'])

plt.axvline(0, c='k', ls='--')
plt.title('Difference in slope Dyslexic minus Controls')
sns.despine()

((tmp['DD'] - tmp['CC']) < 0.0).mean()


#
#%% 

