import os.path as op
import os
import numpy as np
import pandas as pd

import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss


# ppc
def format_bambi_ppc(trace, model, df):

    preds = []
    for key, kind in zip(['ll_bernoulli', 'p'], ['pps', 'mean']):
        pred = model.predict(trace, kind=kind, inplace=False) 
        if kind == 'pps':
            pred = pred['posterior_predictive']['chose_risky'].to_dataframe().unstack(['chain', 'draw'])['chose_risky']
        else:
            pred = pred['posterior']['chose_risky_mean'].to_dataframe().unstack(['chain', 'draw'])['chose_risky_mean']
        pred.index = df.index
        pred = pred.set_index(pd.MultiIndex.from_frame(df), append=True)
        preds.append(pred)

    pred = pd.concat(preds, keys=['ll_bernoulli', 'p'], names=['variable'])
    return pred


def plot_ppc(df, ppc, plot_type=1, var_name='ll_bernoulli', level='subject', col_wrap=5):

    assert (var_name in ['p', 'll_bernoulli'])

    ppc = ppc.xs(var_name, 0, 'variable').copy()

    df = df.copy()

    if level == 'group':
        df['log(risky/safe)'] = df['bin(risky/safe)']
        ppc = ppc.reset_index('log(risky/safe)')
        ppc['log(risky/safe)'] = ppc.index.get_level_values('bin(risky/safe)')

    if plot_type == 1:
        groupby = ['format', 'log(risky/safe)']
    elif plot_type in [2, 4]:
        groupby = ['format', 'n_safe']
    elif plot_type in [3, 5]:
        groupby = ['format', 'n_safe', 'log(risky/safe)']
    else:
        raise NotImplementedError

    if level == 'group':
        ppc = ppc.groupby(['subject', 'group']+groupby).mean()

    if level == 'subject':
        groupby = ['subject'] + groupby

    # print(ppc)
    ppc_summary = summarize_ppc(ppc, groupby=groupby)
    print(ppc_summary)
    p = df.groupby(groupby).mean()[['chose_risky']]
    print(p)
    # ppc_summary = pd.concat((p, ppc_summary), axis=1).sort_index()
    ppc_summary = ppc_summary.join(p).reset_index()

    print(ppc_summary)

    if 'n_safe' in groupby:
        ppc_summary['Safe offer'] = ppc_summary['n_safe'].astype(int)

    ppc_summary['Prop. chosen risky'] = ppc_summary['chose_risky']

    if 'log(risky/safe)' in groupby:
        if level == 'group':
            ppc_summary['Predicted acceptance'] = ppc_summary['log(risky/safe)']
        else:
            ppc_summary['Log-ratio offer'] = ppc_summary['log(risky/safe)']

    if plot_type in [2, 7]:
            x = 'Safe offer'
    else:
        if level == 'group':
            x = 'Predicted acceptance'
        else:
            x = 'Log-ratio offer'

    if level == 'group':
        col_level = 'group'
    elif level == 'subject':
        col_level = 'subject'
    else:
        col_level = None

    if plot_type in [1, 2]:
        fac = sns.FacetGrid(ppc_summary,
                            col=col_level,
                            hue='format',
                            col_wrap=col_wrap if level == 'subject' else None)


    #print("X", x)
    if plot_type in [1,2,3, 5]:
        fac.map_dataframe(plot_prediction, x=x)
        fac.map(plt.scatter, x, 'Prop. chosen risky')
        fac.map(lambda *args, **kwargs: plt.axhline(.5, c='k', ls='--'))

    if plot_type in [1, 3, 5]:
        if level == 'subject':
            fac.map(lambda *args, **kwargs: plt.axvline(np.log(1./.55), c='k', ls='--'))
        else:
            fac.map(lambda *args, **kwargs: plt.axvline(3.5, c='k', ls='--'))
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