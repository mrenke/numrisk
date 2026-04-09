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
from os import listdir, remove

def main(model_label, bids_folder='/Users/mrenke/data/ds-dnumrisk',format='non-symbolic', col_wrap=5,
        remove_subjects = False, remove_sub_string = '32-40-45-46-50'):

# behav_fit3
# does only work when executed via terminal, not in interactive shell of VSC

    subject_list = [f[4:] for f in listdir(bids_folder) if f[0:3] == 'sub' and len(f) == 6]
    if remove_subjects:
        remove_sub_list = [int(s) for s in remove_sub_string.split('-')]
        subject_list = [subject for subject in subject_list if subject not in remove_sub_list]
    
    df = get_data(bids_folder,subject_list)
    df = df.xs(format,0, level='format')
    model = build_model(model_label, df)
    model.build_estimation_model()

    if remove_subjects:
        model_label = f'{model_label}_rem-{remove_sub_string}'
    idata = az.from_netcdf(op.join(bids_folder, f'derivatives/cogmodels_risk/model-{model_label}_format-{format}_trace.netcdf'))

    target_folder = op.join(bids_folder, f'derivatives/cogmodels_risk/figures/{model_label}_format-{format}')
    
    if not op.exists(target_folder):
        os.makedirs(target_folder)

    ppc = model.ppc(idata=idata.sel(draw=slice(None, None, 10)), paradigm=df)
    # "Chose risky" vs "chose 2nd option coding"
    #ppc.loc[ppc.index.get_level_values('risky_first')] = 1 - ppc.loc[ppc.index.get_level_values('risky_first')]

    ppc['format'] = format
    ppc = ppc.set_index(['format'], append=True)
    
    df['format'] = format
    df = df.set_index('format', append=True)

    for plot_type in [1,2,3]: #,3, 5, 6, 7, 8, 9, no "order" (risky_first) in this risk data
       
        for var_name in ['ll_bernoulli']: # 'p', 
            for level in ['subject','group']:
                target_folder = op.join(bids_folder, 'derivatives', 'cogmodels_risk', 'figures', f'{model_label}_format-{format}', var_name)

                if not op.exists(target_folder):
                    os.makedirs(target_folder)

                fn = f'{level}_plot-{plot_type}_model-{model_label}_pred.pdf'
                print(f'plot type: {plot_type}')
                plot_ppc(df, ppc, level=level, plot_type=plot_type, var_name=var_name, col_wrap=col_wrap).savefig(
                    op.join(target_folder, fn))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/Users/mrenke/data/ds-dnumrisk')
    parser.add_argument('--format', default='non-symbolic')
    parser.add_argument('--remove_subjects', action='store_true') #default=False)
    parser.add_argument('--remove_sub_string', default='32-40-45-46-50') # default='32-40-45-46-50'

    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder,format=args.format, remove_subjects=args.remove_subjects, remove_sub_string=args.remove_sub_string)