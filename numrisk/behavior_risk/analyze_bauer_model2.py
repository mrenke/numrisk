# part 2 of model analysis, posterior predictive checks

from itertools import combinations
#import pingouin
import matplotlib.pyplot as plt
import seaborn
import os
import os.path as op
import argparse
import arviz as az
from utils import get_data, build_model, plot_ppc
import numpy as np
import seaborn as sns
from bauer.utils.bayes import softplus
import pandas as pd

def main(model_label, bids_folder='/Users/mrenke/data/ds-stressrisk', AUC=False,col_wrap=5):

# behav_fit3
# does only work when executed via terminal, not in interactive shell of VSC

    df = get_data(bids_folder)
    if AUC:
        df_comb_shifts = pd.read_csv('/Users/mrenke/data/ds-stressrisk/interim_sum_data/r-prior-shift_Ediff_AUC.csv')
        df = df.join(df_comb_shifts.set_index('subject')['AUC'], on='subject', how='left')
        df.dropna(subset=['AUC'], inplace=True) # remove subjects without AUC

    model = build_model(model_label, df)
    model.build_estimation_model()

    idata = az.from_netcdf(op.join(bids_folder, f'derivatives/cogmodels_risk/model-{model_label}_trace.netcdf'))

    target_folder = op.join(bids_folder, f'derivatives/cogmodels_risk/figures/{model_label}')
    if not op.exists(target_folder):
        os.makedirs(target_folder)

    ppc = model.ppc(trace=idata.sel(draw=slice(None, None, 10)), data=df)

    # "Chose risky" vs "chose 2nd option coding"
    ppc.loc[ppc.index.get_level_values('risky_first')] = 1 - ppc.loc[ppc.index.get_level_values('risky_first')]

    for plot_type in [1,2,3, 5, 6, 7, 8, 9]:
    #for plot_type in [1,2,3, 5]:
       
        for var_name in ['p', 'll_bernoulli']:
            for level in ['group']:
            #for level in ['subject']:
                target_folder = op.join(bids_folder, 'derivatives', 'cogmodels', 'figures', model_label, var_name)

                if not op.exists(target_folder):
                    os.makedirs(target_folder)

                fn = f'{level}_plot-{plot_type}_model-{model_label}_pred.pdf'
                plot_ppc(df, ppc, level=level, plot_type=plot_type, var_name=var_name, col_wrap=col_wrap).savefig(
                    op.join(target_folder, fn))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/Users/mrenke/data/ds-stressrisk')
    parser.add_argument('--AUC', action='store_true')

    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder, AUC=args.AUC)