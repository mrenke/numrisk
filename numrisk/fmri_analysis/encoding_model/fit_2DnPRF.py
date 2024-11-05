import argparse
from numrisk.utils import Subject

from braincoder.models import GaussianPointPRF2D, GaussianMixturePRF2D
from braincoder.optimize import ParameterFitter
import os.path as op
import pandas as pd
import numpy as np
from braincoder.utils.stats import get_rsq
import os
from nilearn.input_data import NiftiMasker

def main(subject_id, bids_folder, smoothed=False, mixture_model=False, same_rfs=False,
         task='magjudge'): # nordic=True,
    
    session = 1
    key = 'encoding_model.2d'
    n_stim = 2 # we take activity from the second stimulus
    base_dir = f'glm_stim{n_stim}.denoise'

    if mixture_model:
        key += '.mixture'
        #base_dir += '.mixture'
        if same_rfs:
            key += '.same_rfs'
    if smoothed:
        key += '.smoothed'
        base_dir += '.smoothed'

    subject = f'{int(subject_id):02d}'
    target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func')
    os.makedirs(target_dir, exist_ok=True)
    print(f'mixture_model: {mixture_model}, same_rfs: {same_rfs}')

    sub = Subject(subject_id, bids_folder=bids_folder)
    behavior = sub.get_behavior_magjudge()


    paradigm = np.log(behavior[['n1', 'n2']].copy()).dropna()

    mask = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-magjudge_run-1_space-T1w_desc-brain_mask.nii.gz')
    if not op.exists(mask): # for some subs brain mask creation for some runs did not work apparently, so lets just take another run (they should be pretty similar?)
        mask = op.join(bids_folder, 'derivatives',
                f'fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-magjudge_run-2_space-T1w_desc-brain_mask.nii.gz')

    masker = NiftiMasker(mask_img=mask)

    data = op.join(bids_folder, 'derivatives', base_dir,
                                          f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-magjudge_space-T1w_desc-stims{n_stim}_pe.nii.gz')

    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index)
    #print(data)

    data = pd.DataFrame(data, index=paradigm.index)


    if mixture_model:
        model = GaussianMixturePRF2D(same_rfs=same_rfs)
    else:
        model = GaussianPointPRF2D(correlated_response=True)

    if same_rfs:
        mu = np.log(np.linspace(5, 25, 26, dtype=np.float32)[:1:-1])
        sd = np.linspace(1., 4., 16)
    else:
        mu_x = np.log(np.linspace(5, 25, 10, dtype=np.float32)[:1:-1])
        mu_y = mu_x
        sd_x = np.linspace(1., 4, 8)
        sd_y = np.linspace(1, 4, 8)
        rho = [0.0]

    weights = [0.5]
    amplitudes = [1.]
    baselines = [0.]

    fitter = ParameterFitter(model, data, paradigm)

    # N1 parameters
    if mixture_model:
        if same_rfs:
            grid_pars = fitter.fit_grid(mu, sd, weights, amplitudes, baselines, use_correlation_cost=True)
        else:
            grid_pars = fitter.fit_grid(mu_x, mu_y, sd_x, sd_y, weights, amplitudes, baselines, use_correlation_cost=True)
    else:
        grid_pars = fitter.fit_grid(mu_x, mu_y, sd_x, sd_y, rho, amplitudes, baselines, use_correlation_cost=True)

    grid_pars = fitter.refine_baseline_and_amplitude(grid_pars)

    if same_rfs:
        fixed_pars = ['mu', 'sd']
    else:
        fixed_pars = ['mu_x', 'mu_y', 'sd_x', 'sd_y']

    if mixture_model:
        fixed_pars += ['weight']
    else:
        fixed_pars += ['rho']

    grid_pars = fitter.fit(init_pars=grid_pars, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000, fixed_pars=fixed_pars, r2_atol=0.00001)

    pred_grid = model.predict(parameters=grid_pars, paradigm=paradigm)
    r2_grid = get_rsq(data, pred_grid)
    masker.inverse_transform(r2_grid).to_filename(op.join(target_dir, f'sub-{subject}_task-task_desc-grid.r2_space-T1w_pars.nii.gz'))
    for column in grid_pars.columns:
        masker.inverse_transform(grid_pars[column]).to_filename(op.join(target_dir, f'sub-{subject}_task-task_desc-grid.{column}_space-T1w_pars.nii.gz'))
    
    pars_gd = fitter.fit(init_pars=grid_pars, learning_rate=.003, store_intermediate_parameters=False, max_n_iterations=10000,
            r2_atol=0.00001)

    pred_gd = model.predict(parameters=pars_gd, paradigm=paradigm)
    r2_gd = get_rsq(data, pred_gd)
    masker.inverse_transform(r2_gd).to_filename(op.join(target_dir, f'sub-{subject}_task-{task}_desc-gd.r2_space-T1w_pars.nii.gz'))

    for column in pars_gd.columns:
        masker.inverse_transform(pars_gd[column]).to_filename(op.join(target_dir, f'sub-{subject}_task-{task}_desc-gd.{column}_space-T1w_pars.nii.gz'))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('subject_id', type=str)
    parser.add_argument('--bids_folder', type=str, default='/data/ds-retinonumeral')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--mixture_model', action='store_true')
    parser.add_argument('--same_rfs', action='store_true')
    args = parser.parse_args()

    main(args.subject_id, args.bids_folder, args.smoothed, args.mixture_model, same_rfs=args.same_rfs)