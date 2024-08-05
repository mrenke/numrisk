import os.path as op
import os
import numpy as np
import pandas as pd

import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss


def plot_ppc(df, ppc, plot_type=1, var_name='ll_bernoulli', level='subject', col_wrap=5):

    assert (var_name in ['p', 'll_bernoulli'])

    ppc = ppc.xs(var_name, 0, 'variable').copy()

    df = df.copy()

    if plot_type == 1:
        groupby = ['log(n2/n1)']
    elif plot_type in [2, 4]:
        groupby = ['n1']
    elif plot_type in [3, 5]:
        groupby = ['n1', 'log(n2/n1)']
    else:
        raise NotImplementedError
    
    if level == 'group':
        ppc = ppc.groupby(['subject', 'group']+groupby).mean()
        groupby = ['group'] + groupby

    if level == 'subject':
        groupby = ['subject'] + groupby

    ppc_summary = summarize_ppc(ppc, groupby=groupby)
    p = df.groupby(groupby).mean()[['chose_n2']]
    ppc_summary = ppc_summary.join(p).reset_index()
    ppc_summary['Prop. chosen n2'] = ppc_summary['chose_n2']
    print(ppc_summary)
    
    #
    if plot_type in [1,3]:
        x = 'log(n2/n1)'
    elif plot_type in [2]:
        x = 'n1'
            
    if level == 'group':
        col_level = 'group'
    elif level == 'subject':
        col_level = 'subject'
    else:
        col_level = None

    # start plotting
    if plot_type in [1, 2]:
        fac = sns.FacetGrid(ppc_summary,
                            col=col_level,
                            #hue='format',
                            col_wrap=col_wrap if level == 'subject' else None)
    if plot_type in [3]:
        fac = sns.FacetGrid(ppc_summary,
                            col=col_level,
                            hue='n1',
                            col_wrap=col_wrap if level == 'subject' else None)
     
    if plot_type in [1,2,3]:
        fac.map_dataframe(plot_prediction, x=x)
        fac.map(plt.scatter, x, 'Prop. chosen n2')
        fac.map(lambda *args, **kwargs: plt.axhline(.5, c='k', ls='--'))

    if plot_type in [1,3]: # x = log(n2/n1)
        fac.map(lambda *args, **kwargs: plt.axvline(0, c='k', ls='--'))
        plt.xticks([])

    fac.add_legend()

    return fac





def summarize_ppc(ppc, groupby=None):

    if groupby is not None:
        ppc = ppc.groupby(groupby).mean()

    e = ppc.mean(1).to_frame('p_predicted')
    hdi = pd.DataFrame(az.hdi(ppc.T.values), index=ppc.index,
                       columns=['hdi025', 'hdi975'])

    #print(hdi)
    return pd.concat((e, hdi), axis=1)

def plot_prediction(data, x, color, y='p_predicted', alpha=.25, **kwargs):
    data = data[~data['hdi025'].isnull()]
    plt.fill_between(data[x], data['hdi025'],
                     data['hdi975'], color=color, alpha=alpha)
    plt.plot(data[x], data[y], color=color)    

def format_bambi_ppc(trace, model, df):

    preds = []
    for key, kind in zip(['ll_bernoulli', 'p'], ['pps', 'mean']):
        pred = model.predict(trace, kind=kind, inplace=False) 
        if kind == 'pps':
            pred = pred['posterior_predictive']['chose_n2'].to_dataframe().unstack(['chain', 'draw'])['chose_n2']
        else:
            pred = pred['posterior']['chose_n2_mean'].to_dataframe().unstack(['chain', 'draw'])['chose_n2_mean']
        pred.index = df.index
        pred = pred.set_index(pd.MultiIndex.from_frame(df), append=True)
        preds.append(pred)

    pred = pd.concat(preds, keys=['ll_bernoulli', 'p'], names=['variable'])
    return pred
