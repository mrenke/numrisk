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

def main(subject_id, bids_folder, smoothed=False, mixture_model=False, same_rfs=False,task = 'magjudge'): # nordic=True, 
    print(f'mixture_model: {mixture_model}, same_rfs: {same_rfs}')
    subject = f'{int(subject_id):02d}'
    session = 1
    runs = range(1, 7)

    key = 'encoding_model.2d.cv'
    n_stim = 2 # we are interested in activity from the second stimulus
    base_dir = f'glm_stim{n_stim}.denoise'
    if mixture_model:
        key += '.mixture'
        if same_rfs:
            key += '.same_rfs'
    else:
        if same_rfs:
            raise NotImplementedError('Cannot have a 2D Gaussian with same RF right now.')

    if smoothed:
        key += '.smoothed'
        base_dir += '.smoothed'

    target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func')
    os.makedirs(target_dir, exist_ok=True)

    sub = Subject(subject_id, bids_folder=bids_folder)
    behavior = sub.get_behavior_magjudge()
    paradigm = np.log(behavior[['n1', 'n2']].copy()).dropna().droplevel('subject')

    # get brain data
    masker = sub.get_brain_masker()
    data = op.join(bids_folder, 'derivatives', base_dir, # base-dir defines we get actuvity from stim2
                                          f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-magjudge_space-T1w_desc-stims{n_stim}_pe.nii.gz')
    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index)
    data = pd.DataFrame(data, index=paradigm.index)

    # Select the model based on the mixture_model flag
    if mixture_model:
        model = GaussianMixturePRF2D(same_rfs=same_rfs)
    else:
        model = GaussianPointPRF2D(correlated_response=True)

    # Set up parameters for the model
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

    cv_r2s = []

    # Cross-validation loop: for each session and run, split data into train/test sets
    for test_run in runs:

        test_data, test_paradigm = data.loc[test_run].copy(), paradigm.loc[test_run].copy()
        train_data, train_paradigm = data.drop(test_run, level='run').copy(), paradigm.drop(test_run, level='run').copy()

        fitter = ParameterFitter(model, train_data, train_paradigm)

        # Grid fitting parameters for N1 model
        if mixture_model:
            if same_rfs:
                grid_pars = fitter.fit_grid(mu, sd, weights, amplitudes, baselines, use_correlation_cost=True)
            else:
                grid_pars = fitter.fit_grid(mu_x, mu_y, sd_x, sd_y, weights, amplitudes, baselines, use_correlation_cost=True)
        else:
            grid_pars = fitter.fit_grid(mu_x, mu_y, sd_x, sd_y, rho, amplitudes, baselines, use_correlation_cost=True)

        grid_pars = fitter.refine_baseline_and_amplitude(grid_pars, n_iterations=2)
        fixed_pars = ['mu_x', 'mu_y', 'sd_x', 'sd_y'] if not same_rfs else ['mu', 'sd']
        if mixture_model:
            fixed_pars += ['weight']
        else:
            fixed_pars += ['rho']

        grid_pars = fitter.fit(init_pars=grid_pars, learning_rate=.05, store_intermediate_parameters=False,
                               max_n_iterations=10000, fixed_pars=fixed_pars, r2_atol=0.00001)

        # Predict and calculate R^2 values for training and test sets
        pred_train = model.predict(parameters=grid_pars, paradigm=train_paradigm)
        pred_test = model.predict(parameters=grid_pars, paradigm=test_paradigm)
        r2_grid = get_rsq(train_data, pred_train)
        cv_r2_grid = get_rsq(test_data, pred_test)

        # Save the results
        masker.inverse_transform(r2_grid).to_filename(op.join(target_dir, f'sub-{subject_id}_ses-{session}_task-{task}_run-{test_run}_desc-grid.r2_space-T1w_pars.nii.gz'))
        masker.inverse_transform(cv_r2_grid).to_filename(op.join(target_dir, f'sub-{subject_id}_ses-{session}_task-{task}_run-{test_run}_desc-grid.cvr2_space-T1w_pars.nii.gz'))

        # Gradient descent fitting
        pars_gd = fitter.fit(init_pars=grid_pars, learning_rate=.01, store_intermediate_parameters=False,
                             max_n_iterations=10000, r2_atol=0.00001)

        pred_train = model.predict(parameters=pars_gd, paradigm=train_paradigm)
        pred_test = model.predict(parameters=pars_gd, paradigm=test_paradigm)

        r2_gd = get_rsq(train_data, pred_train)
        cv_r2_gd = get_rsq(test_data, pred_test)

        masker.inverse_transform(r2_gd).to_filename(op.join(target_dir, f'sub-{subject}_ses-{session}_task-{task}_run-{test_run}_desc-gd.r2_space-T1w_pars.nii.gz'))
        masker.inverse_transform(cv_r2_gd).to_filename(op.join(target_dir, f'sub-{subject}_ses-{session}_task-{task}_run-{test_run}_desc-gd.cvr2_space-T1w_pars.nii.gz'))

        for column in pars_gd.columns:
            masker.inverse_transform(pars_gd[column]).to_filename(op.join(target_dir, f'sub-{subject}_ses-{session}_task-{task}_run-{test_run}_desc-gd.{column}_space-T1w_pars.nii.gz'))

        # Append the cross-validated R^2 values
        cv_r2s.append(cv_r2_gd)

    # Combine all cross-validated R^2 results and save them
    #test_keys = [test_ix for test_ix, _ in paradigm.groupby('run')]
    cv_r2s = pd.concat(cv_r2s, keys=runs, names=['run']).groupby(level=1, axis=0).mean()
    print(f'shape : ')
    print(np.shape(cv_r2s))
    try:    
        masker.inverse_transform(cv_r2s).to_filename(op.join(target_dir, f'sub-{subject}_task-{task}_desc-cvr2_space-T1w_pars.nii.gz'))
    except:
        print('had to transpose')
        masker.inverse_transform(cv_r2s.T).to_filename(op.join(target_dir, f'sub-{subject}_task-{task}_desc-cvr2_space-T1w_pars.nii.gz'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('subject_id', type=str)
    parser.add_argument('--bids_folder', type=str, default='/data/ds-ds-dnumrisk')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--mixture_model', action='store_true')
    parser.add_argument('--same_rfs', action='store_true')

    args = parser.parse_args()

    main(subject_id=args.subject_id, bids_folder=args.bids_folder, smoothed=args.smoothed, mixture_model=args.mixture_model, same_rfs=args.same_rfs)