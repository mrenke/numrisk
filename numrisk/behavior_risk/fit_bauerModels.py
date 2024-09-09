import argparse
from bauer.models import RiskRegressionModel

import os.path as op
import os
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from utils import get_data
from utils_02 import build_model

from os import listdir

def main(model_label, burnin=2000, samples=2000, bids_folder = '/Users/mrenke/data/ds-dnumrisk',format='non-symbolic', remove_404650 = False):

    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels_risk')
    
    if not op.exists(target_folder):
        os.makedirs(target_folder)

    subject_list = [f[4:] for f in listdir(bids_folder) if f[0:3] == 'sub' and len(f) == 6]
    if remove_404650:
        subject_list = [subject for subject in subject_list if subject not in ['40', '46', '50']]

    df = get_data(bids_folder,subject_list)
    df = df.xs(format,0, level='format')

    # different evidences for safe (n1) & risky (n2) options: everything already coded so that n1 always safe and n2 always risky & choice = chose_risky
    if any(char in model_label for char in ['6', '7','9'])  : # unnecessary 
        print('different evidences for safe (n1) & risky (n2) options!')

    target_accept = 0.9

    model = build_model(model_label, df)
    model.build_estimation_model()
    trace = model.sample(burnin, samples, target_accept=target_accept)
    
    with model.estimation_model:
        pm.compute_log_likelihood(trace)

    if remove_404650:
        model_label = f'{model_label}_remove404650'

    az.to_netcdf(trace,
                    op.join(target_folder, f'model-{model_label}_format-{format}_trace.netcdf'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/Users/mrenke/data/ds-dnumrisk')
    parser.add_argument('--format', default='non-symbolic')
    parser.add_argument('--remove_404650', action='store_true') #default=False)

    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder, format=args.format, remove_404650=args.remove_404650)

