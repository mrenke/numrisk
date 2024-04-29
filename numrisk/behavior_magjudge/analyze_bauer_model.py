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
from utils_02 import build_model #, plot_ppc

def main(model_label, bids_folder='/Users/mrenke/data/ds-dnumrisk',col_wrap=5, only_ppc=False, # AUC=False,E_dif=False, 
plot_traces=False):

# behav_fit3
# does only work when executed via terminal, not in interactive shell of VSC

    df = get_data(bids_folder)

    model = build_model(model_label, df)
    model.build_estimation_model()

    idata = az.from_netcdf(op.join(bids_folder, f'derivatives/cogmodels_magjudge/model-{model_label}_trace.netcdf'))

    target_folder = op.join(bids_folder, f'derivatives/cogmodels_magjudge/figures/{model_label}')
    if not op.exists(target_folder):
        os.makedirs(target_folder)

    if plot_traces:
        az.plot_trace(idata, var_names=['~p'])
        plt.savefig(op.join(target_folder, 'traces.pdf'))


    for par in model.free_parameters:
        traces = idata.posterior[par+'_mu'].to_dataframe()


        for regressor, t in traces.groupby(par+'_regressors'):
            t = t.copy()
            print(regressor, t)
            if (any(substr in par for substr in ['prior_std', 'evidence_sd'])) & (regressor == 'Intercept'): #  'risky_prior_std', 'safe_prior_std', 'n1_evidence_sd', 'n2_evidence_sd',
                t = softplus(t)

            plt.figure()
            sns.kdeplot(t, fill=True)
            if regressor != 'Intercept':
                plt.axvline(0.0, c='k', ls='--')
                txt = f'p({par} < 0.0) = {np.round((t.values < 0.0).mean(), 3)}'
                plt.xlabel(txt)

            #else:
                #if par == 'risky_prior_mu':

            plt.savefig(op.join(target_folder, f'group_par-{par}.{regressor}.pdf'))
            plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/Users/mrenke/data/ds-dnumrisk')
    #parser.add_argument('--AUC', action='store_true')
    #parser.add_argument('--E_dif', action='store_true')

    parser.add_argument('--only_ppc', action='store_true')
    parser.add_argument('--no_trace', dest='plot_traces', action='store_false')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder, only_ppc=args.only_ppc, plot_traces=args.plot_traces) # , AUC=args.AUC, E_dif=args.E_dif