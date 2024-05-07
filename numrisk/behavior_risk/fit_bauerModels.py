import argparse
from bauer.models import RiskRegressionModel

import os.path as op
import os
import arviz as az
import numpy as np
import pandas as pd

from utils import get_data
from utils_02 import build_model

def main(model_label, burnin=2000, samples=2000, bids_folder = '/Users/mrenke/data/ds-dnumrisk',format='non-symbolic'):

    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels_risk')
    
    if not op.exists(target_folder):
        os.makedirs(target_folder)

    df = get_data(bids_folder)
    df = df.xs(format,0, level='format')

    target_accept = 0.9

    model = build_model(model_label, df)
    model.build_estimation_model()
    trace = model.sample(burnin, samples, target_accept=target_accept)
    az.to_netcdf(trace,
                    op.join(target_folder, f'model-{model_label}_format-{format}_trace.netcdf'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/Users/mrenke/data/ds-dnumrisk')
    parser.add_argument('--format', default='non-symbolic')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder, format=args.format)

