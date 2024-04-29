import argparse

import os.path as op
import os
import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from utils import get_data
from utils_02 import build_model

def main(model_label, burnin=2000, samples=2000, bids_folder = '/Users/mrenke/data/ds-dnumrisk'):
    
    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels_magjudge')
    
    if not op.exists(target_folder):
        os.makedirs(target_folder)

    df = get_data(bids_folder)
    df.reset_index('run', inplace=True) 

    target_accept = 0.9

    model = build_model(model_label, df)
    model.build_estimation_model()
    trace = model.sample(burnin, samples, target_accept=target_accept)

    # compute log likelihood (not done automatically anymore)
    with model.estimation_model:
        pm.compute_log_likelihood(trace)

    az.to_netcdf(trace,
                    op.join(target_folder, f'model-{model_label}_trace.netcdf'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/Users/mrenke/data/ds-dnumrisk')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder)


