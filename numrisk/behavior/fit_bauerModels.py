

# %%
import argparse
from bauer.models import RiskRegressionModel
#from tms_risk.utils.data import get_all_behavior
from stress_risk.utils.data import get_all_behavior
import os.path as op
import os
import arviz as az
import numpy as np
import pandas as pd
from utils import add_cond2df, build_model, get_behavior

def main(model_label, burnin=1000, samples=1000, bids_folder = bids_folder = '/Users/mrenke/data/ds-dnumr'):

    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')
    
    if not op.exists(target_folder):
        os.makedirs(target_folder)

    df = get_data(bids_folder)


    if model_label in ['0', '5']:
        target_accept = 0.9
    else:
        target_accept = 0.8

    model = build_model(model_label, df)
    model.build_estimation_model()
    trace = model.sample(burnin, samples, target_accept=target_accept)
    az.to_netcdf(trace,
                    op.join(target_folder, f'model-{model_label}_trace.netcdf'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/Users/mrenke/data/ds-stressrisk')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder)

