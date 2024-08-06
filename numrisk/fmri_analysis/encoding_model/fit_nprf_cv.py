import argparse
import pandas as pd
from braincoder.models import GaussianPRF
from braincoder.optimize import ParameterFitter
from braincoder.utils import get_rsq
from numrisk.utils import get_target_dir
from nilearn.input_data import NiftiMasker

import os.path as op
import numpy as np


def main(subject,  bids_folder='/data/ds-dnumrisk', retroicor=False, smoothed=False, pca_confounds=False, denoise=False, split_data = None):
         
    session = 1

    if split_data == None:
        runs = range(1, 7)
        split_data = ''
    else:
        if split_data == 'run_123':
            runs = range(1, 4)
        elif split_data == 'run_456':
            runs = range(4,7)
        split_data = f'_{split_data}' 

    key = f'glm_stim1{split_data}'
    target_dir = f'encoding_model{split_data}.cv'

    if denoise:
        key += '.denoise'
        target_dir += '.denoise'

    if retroicor:
        key += '.retroicor'
        target_dir += '.retroicor'

    if smoothed:
        key += '.smoothed'
        target_dir += '.smoothed'

    if pca_confounds:
        key += '.pca_confounds'
        target_dir += '.pca_confounds'

    target_dir = get_target_dir(subject, session, bids_folder, target_dir)

    paradigm = [pd.read_csv(op.join(bids_folder, f'sub-{subject}', f'ses-{session}',
                                    'func', f'sub-{subject}_ses-{session}_task-magjudge_run-{run}_events.tsv'), sep='\t')
                for run in runs]
    paradigm = pd.concat(paradigm, keys=runs, names=['run'])
    paradigm = paradigm[paradigm.trial_type ==
                        'stimulus 1'].set_index('trial_nr', append=True)

    paradigm['log(n1)'] = np.log(paradigm['n1'])
    paradigm = paradigm['log(n1)']

    print(paradigm)

    model = GaussianPRF()
    # SET UP GRID
    mus = np.log(np.linspace(5, 80, 60, dtype=np.float32))
    sds = np.log(np.linspace(2, 30, 60, dtype=np.float32))
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    mask = op.join(bids_folder, 'derivatives',
                   f'fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-magjudge_run-1_space-T1w_desc-brain_mask.nii.gz')
    if ~op.exists(mask): # for some subs brain mask creation for some runs did not work apparently, so lets just take another run (they should be pretty similar?)
        mask = op.join(bids_folder, 'derivatives',
                f'fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-magjudge_run-2_space-T1w_desc-brain_mask.nii.gz')

    masker = NiftiMasker(mask_img=mask)

    data = op.join(bids_folder, 'derivatives', key,
                   f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-magjudge_space-T1w_desc-stims1_pe.nii.gz')

    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index)
    print(data)

    data = pd.DataFrame(data, index=paradigm.index)

    cv_r2s = []

    for test_run in runs:

        test_data, test_paradigm = data.loc[test_run].copy(
        ), paradigm.loc[test_run].copy()
        print(test_data, test_paradigm)
        train_data, train_paradigm = data.drop(
            test_run, level='run').copy(), paradigm.drop(test_run, level='run').copy()

        optimizer = ParameterFitter(model, train_data, train_paradigm)

        grid_parameters = optimizer.fit_grid(
            mus, sds, amplitudes, baselines, use_correlation_cost=True)
        grid_parameters = optimizer.refine_baseline_and_amplitude(
            grid_parameters, n_iterations=2)

        optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
                      r2_atol=0.00001)

        target_fn = op.join(
            target_dir, f'sub-{subject}_ses-{session}_run-{test_run}_desc-r2.optim_space-T1w_pars.nii.gz')

        masker.inverse_transform(optimizer.r2).to_filename(target_fn) # save r2 maps

        cv_r2 = get_rsq(test_data, model.predict(parameters=optimizer.estimated_parameters,
                                                 paradigm=test_paradigm.astype(np.float32)))

        target_fn = op.join(
            target_dir, f'sub-{subject}_ses-{session}_run-{test_run}_desc-cvr2.optim_space-T1w_pars.nii.gz')

        masker.inverse_transform(cv_r2).to_filename(target_fn) # save cv-r2 maps

        for par, values in optimizer.estimated_parameters.T.iterrows():
            print(values)
            target_fn = op.join(
                target_dir, f'sub-{subject}_ses-{session}_run-{test_run}_desc-{par}.optim_space-T1w_pars.nii.gz')

            masker.inverse_transform(values).to_filename(target_fn) # save parameter maps

        cv_r2s.append(cv_r2)
    
    cv_r2 = pd.concat(cv_r2s, keys=runs, names=['run']).groupby(level=1, axis=0).mean()

    target_fn = op.join(
        target_dir, f'sub-{subject}_ses-{session}_desc-cvr2.optim_space-T1w_pars.nii.gz')

    masker.inverse_transform(cv_r2).to_filename(target_fn) # save cv-r2 maps (all run mean)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    #parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data')
    parser.add_argument('--retroicor', action='store_true')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--pca_confounds', action='store_true')
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--split_data', default=None)

    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder, smoothed=args.smoothed, retroicor=args.retroicor,
            pca_confounds=args.pca_confounds, denoise=args.denoise,
            split_data = args.split_data)