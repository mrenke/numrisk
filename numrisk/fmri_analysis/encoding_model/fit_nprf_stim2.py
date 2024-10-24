import argparse
import pandas as pd
from braincoder.models import GaussianPRF
from braincoder.optimize import ParameterFitter
from nilearn.input_data import NiftiMasker
from numrisk.utils import get_target_dir
import os
import os.path as op
import numpy as np

def main(subject, bids_folder, denoise=True, retroicor=False, smoothed=True,
        pca_confounds=False):
    
    subject = f'{int(subject):02d}'
    session = 1
    runs = range(1, 7)

    n_stim = 2
    key = f'glm_stim{n_stim}'
    target_dir = f'encoding_model_stim{n_stim}'

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
    paradigm = paradigm[paradigm.trial_type == f'stimulus {n_stim}'].set_index('trial_nr', append=True)

    paradigm[f'log(n{n_stim})'] = np.log(paradigm[f'n{n_stim}'])
    paradigm = paradigm[f'log(n{n_stim})'].dropna()

    print(paradigm)

    model = GaussianPRF()

    # SET UP GRID
    mus = np.log(np.linspace(5, 80, 60, dtype=np.float32))
    sds = np.log(np.linspace(2, 30, 60, dtype=np.float32))
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    mask = op.join(bids_folder, 'derivatives',
                   f'fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-magjudge_run-1_space-T1w_desc-brain_mask.nii.gz')
    if not op.exists(mask): # for some subs brain mask creation for some runs did not work apparently, so lets just take another run (they should be pretty similar?)
        mask = op.join(bids_folder, 'derivatives',
                f'fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-magjudge_run-2_space-T1w_desc-brain_mask.nii.gz')

    masker = NiftiMasker(mask_img=mask)

    data = op.join(bids_folder, 'derivatives', key,
                                          f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-magjudge_space-T1w_desc-stims{n_stim}_pe.nii.gz')

    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index)
    print(data)

    data = pd.DataFrame(data, index=paradigm.index)

    optimizer = ParameterFitter(model, data, paradigm)

    grid_parameters = optimizer.fit_grid(mus, sds, amplitudes, baselines, use_correlation_cost=True)
    grid_parameters = optimizer.refine_baseline_and_amplitude(grid_parameters, n_iterations=2)


    optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
            r2_atol=0.00001)

    target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-r2.optim_space-T1w_pars.nii.gz')

    masker.inverse_transform(optimizer.r2).to_filename(target_fn)

    for par, values in optimizer.estimated_parameters.T.iterrows():
        print(values)
        target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-{par}.optim_space-T1w_pars.nii.gz')
        masker.inverse_transform(values).to_filename(target_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    #parser.add_argument('session', default=1)
    parser.add_argument('--bids_folder', default='/Volumes/mrenkeED/data/ds-dnumr')
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--retroicor', action='store_true')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--pca_confounds', action='store_true')

    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder, denoise=args.denoise, smoothed=args.smoothed,retroicor=args.retroicor, # args.session,
            pca_confounds=args.pca_confounds)
    