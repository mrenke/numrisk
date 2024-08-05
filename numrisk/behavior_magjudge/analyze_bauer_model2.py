# part 2 of model analysis, posterior predictive checks

from itertools import combinations
#import pingouin
import matplotlib.pyplot as plt
import seaborn
import os
import os.path as op
import argparse
import arviz as az
from utils import get_data
from utils_02 import build_model 
from utils_03 import plot_ppc

import numpy as np
import seaborn as sns
from bauer.utils.bayes import softplus
import pandas as pd

def main(model_label, bids_folder='/Users/mrenke/data/ds-dnumrisk', col_wrap=5):

    df = get_data(bids_folder)
    #df = df.loc[~df.index.get_level_values('subject').isin([65,66])] # old dataset

    model = build_model(model_label, df)
    model.build_estimation_model()

    idata = az.from_netcdf(op.join(bids_folder, f'derivatives/cogmodels_magjudge/model-{model_label}_trace.netcdf'))

    target_folder = op.join(bids_folder, f'derivatives/cogmodels_magjudge/figures/{model_label}')
    if not op.exists(target_folder):
        os.makedirs(target_folder)

    ppc = model.ppc(idata=idata.sel(draw=slice(None, None, 10)), paradigm=df)

    for plot_type in [1,2,3]:
       
        for var_name in ['ll_bernoulli']: # 'p', 
            for level in ['subject', 'group']:
                target_folder = op.join(bids_folder, 'derivatives', 'cogmodels_magjudge', 'figures', model_label, var_name)

                if not op.exists(target_folder):
                    os.makedirs(target_folder)

                fn = f'{level}_plot-{plot_type}_model-{model_label}_pred.pdf'
                plot_ppc(df, ppc, level=level, plot_type=plot_type, var_name=var_name, col_wrap=col_wrap).savefig(
                    op.join(target_folder, fn))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/Users/mrenke/data/ds-dnumrisk')

    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder)