from itertools import combinations
#import pingouin
import matplotlib.pyplot as plt
import seaborn
import os
import os.path as op
import argparse
import arviz as az
import numpy as np
import seaborn as sns
from bauer.utils.bayes import softplus
import pandas as pd

from utils import get_data
from utils_02 import build_model, get_rnp

def main(model_label, bids_folder='/Users/mrenke/data/ds-dnumrisk',format='non-symbolic',col_wrap=5, only_ppc=False, # AUC=False,E_dif=False, 
plot_traces=False):

# behav_fit3
# does only work when executed via terminal, not in interactive shell of VSC

    df = get_data(bids_folder)

    model = build_model(model_label, df)
    model.build_estimation_model()

    idata = az.from_netcdf(op.join(bids_folder, f'derivatives/cogmodels_risk/model-{model_label}_format-{format}_trace.netcdf'))

    target_folder = op.join(bids_folder, f'derivatives/cogmodels_risk/figures/{model_label}_format-{format}')
    if not op.exists(target_folder):
        os.makedirs(target_folder)

    if plot_traces:
        az.plot_trace(idata, var_names=['~p'])
        plt.savefig(op.join(target_folder, 'traces.pdf'))

    if model.prior_estimate == 'klw':
        idata.posterior['rnp'] = get_rnp(idata.posterior['evidence_sd'], idata.posterior['prior_sd'])
        idata.posterior['rnp_mu'] = get_rnp(idata.posterior['evidence_sd_mu'], idata.posterior['prior_sd_mu'])
        model.free_parameters['rnp'] = '' # appending to a dictionary

    for par in model.free_parameters:
        traces = idata.posterior[par+'_mu'].to_dataframe()

        par_helper = par if par != 'rnp' else 'evidence_sd'

        for regressor, t in traces.groupby(par_helper+'_regressors'):
            t = t.copy()
            print(regressor, t)
            if (par in ['prior_sd', 'evidence_sd']) & (regressor == 'Intercept'): #  'risky_prior_std', 'safe_prior_std', 'n1_evidence_sd', 'n2_evidence_sd',
                t = softplus(t)

            plt.figure()
            sns.kdeplot(t, fill=True)
            if regressor != 'Intercept':
                plt.axvline(0.0, c='k', ls='--')
                txt = f'p({par} < 0.0) = {np.round((t.values < 0.0).mean(), 3)}'
                plt.xlabel(txt)

            else:
                if par == 'risky_prior_mu':
                    plt.axvline(np.log(df['n_risky']).mean(), c='k', ls='--')
                elif par == 'risky_prior_sd':
                    plt.axvline(np.log(df['n_risky']).std(), c='k', ls='--')
                elif par == 'safe_prior_mu':
                    for n_safe in np.log([7., 10., 14., 20., 28.]):
                        plt.axvline(n_safe, c='k', ls='--')

                    plt.axvline(np.log(df['n_safe']).mean(), c='k', ls='--', lw=2)
                elif par == 'safe_prior_sd':
                    plt.axvline(np.log(df['n_safe']).std(), c='k', ls='--')

            plt.savefig(op.join(target_folder, f'group_par-{par}.{regressor}.pdf'))
            plt.close()







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/Users/mrenke/data/ds-dnumrisk')
    #parser.add_argument('--AUC', action='store_true')
    #parser.add_argument('--E_dif', action='store_true')
    parser.add_argument('--format', default='non-symbolic')

    parser.add_argument('--only_ppc', action='store_true')
    parser.add_argument('--no_trace', dest='plot_traces', action='store_false')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder, only_ppc=args.only_ppc, plot_traces=args.plot_traces, format=args.format) # , AUC=args.AUC, E_dif=args.E_dif