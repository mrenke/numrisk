from nilearn.connectome import ConnectivityMeasure
from brainspace.utils.parcellation import reduce_by_labels
import numpy as np
import os.path as op
from numrisk.fmri_analysis.gradients.utils import cleanTS, get_glasser_parcels
import pandas as pd
import os
import argparse

import time
def print_current_time(step):
    print(f"{step} at {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Get Glasser parcellation and mask
from numrisk.fmri_analysis.gradients.utils import get_glasser_parcels
mask_glasser, labeling_glasser = get_glasser_parcels(space = 'fsaverage5' )

from utils_01 import get_cleanTS_run, get_events_confounds
import statsmodels.api as sm


psy_context1 = 'stimulus1_int'
psy_context2 = 'stimulus2_int'
regressors_names = ['seedTS', 'psy_context1', 'psy_context2', 'seedTS:psy_context1', 'seedTS:psy_context2']
formula = 'targetTS ~ ' + ' + '.join(regressors_names)

ses = 1
def main(sub,bids_folder):
    sub = '%02d' % int(sub)
    target_folder = op.join(bids_folder,'derivatives','correlation_matrices')

    # prep data
    DMs = []
    TSs = []
    for run in range(1, 7):
        try:
            dm = get_events_confounds(sub, ses=ses, run=run, bids_folder=bids_folder)
            dm['run'] = run
            ts = get_cleanTS_run(sub, run=run, bids_folder=bids_folder)

            TSs.append(ts)
            DMs.append(dm)
        except:
            print(f'!!! ------ problems for run {run} (sub {sub} )')
    dm = pd.concat(DMs) # 1128 timesteps = 188 x 6
    dm.index.name = 'onset'
    dm.set_index('run',append=True,inplace=True)

    TSs = np.array(TSs)
    TSs = TSs.transpose(1, 0, 2).reshape(TSs.shape[1], (TSs.shape[0]*TSs.shape[2])) # concatenate runs
    clean_ts = reduce_by_labels(TSs[mask_glasser], labeling_glasser[mask_glasser], axis=1, red_op='mean',dtype=float)
    print_current_time(f'sub-{sub}: data prepared, start computing PPI matrices')

    # Compute PPI matrices
    tvals = {param: [] for param in regressors_names}
    tvals_mat = np.zeros((len(tvals.keys()), 360,360))
    for seed_parcel_n in range(360):
        seedTS = clean_ts[seed_parcel_n]
        for target_parcel_n in range(seed_parcel_n, 360):  # Start from seed_parcel_n to only compute upper triangular part
            df_glm = pd.DataFrame(np.array([seedTS, clean_ts[target_parcel_n], dm[psy_context1], dm[psy_context2]]).T, columns = ['seedTS','targetTS','psy_context1', 'psy_context2'])
            result = sm.formula.ols(formula=formula, data=df_glm).fit()
            for i, param in enumerate(tvals.keys()):
                tvals_mat[i, seed_parcel_n, target_parcel_n] = result.tvalues[param]
                tvals_mat[i, target_parcel_n, seed_parcel_n] = result.tvalues[param] # symetric!

    np.save(op.join(target_folder, f'sub-{sub}_PPI-allParamsMatrices.npy'), tvals_mat)
    print_current_time(f'finished sub-{sub}: PPI matrices generated & saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder', default='/mnt_03/ds-dnumrisk')

    cmd_args = parser.parse_args()
    main(cmd_args.subject, cmd_args.bids_folder)
