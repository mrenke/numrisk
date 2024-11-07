# Standard argparse stuff
import argparse
import pandas as pd
from numrisk.utils import Subject
import numpy as np
from braincoder.optimize import ParameterFitter, ResidualFitter
from braincoder.models import GaussianPointPRF2D, GaussianMixturePRF2D, LogGaussianPRF
from braincoder.utils.stats import get_rsq
import pingouin as pg
import os
import os.path as op

def main(subject_id, smoothed, bids_folder, two_dimensional=True,
         mixture_model=False, same_rfs=False, n_voxels=100, 
         roi='NPC_R',  runs = range(1, 7)):
    subject = f'{int(subject_id):02d}'
    session = 1
    n_stim = 2
   
    sub = Subject(subject_id, bids_folder)
    behavior = sub.get_behavior_magjudge()
    paradigm = np.log(behavior[['n1', 'n2']].copy()).dropna().droplevel('subject')

    data = sub.get_single_trial_volume(session, n_stim=n_stim, roi=None, # NPC_R filter will be applied later when r2-mask is generated
                                       smoothed=smoothed, denoise=True).astype(np.float32)
    data.index = paradigm.index

    key = 'decoding'

    if two_dimensional:
        key += '.2d'
    else:
        raise Exception('Only 2D models are implemented, for 1D models use `decode.py`.')

    if mixture_model:
        key += '.mixture'
    
    if same_rfs:
        key += '.same_rfs'

    if smoothed:
        key += '.smoothed'

    target_dir = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    
    if mixture_model:
        model = GaussianMixturePRF2D(same_rfs=same_rfs)
    else:
        model = GaussianPointPRF2D(correlated_response=True)

    x = np.log(np.arange(5, 26, 1.))
    y = x.copy()
    paradigm_x, paradigm_y = np.meshgrid(x, y)

    # Stack into a 10,000 x 2 array
    stimulus_range = np.column_stack((paradigm_x.ravel(), paradigm_y.ravel()))


    pdfs = []
    keys = []
    pdfs = []

    for test_run in runs:
        print(f'Decoding run {test_run}')

        test_data, test_paradigm = data.loc[test_run].copy(), paradigm.loc[test_run].copy()
        train_data, train_paradigm = data.drop(test_run, level='run').copy(), paradigm.drop(test_run, level='run').copy()

        pars_gd = sub.get_cv_prf_parameters_volume(session=1,holdout_run=test_run, two_dimensional=True, # n=1,
                                  mixture_model=True,
                                    same_rfs=True,
                                    roi='NPC_R')

        r2_mask = pars_gd['r2'].sort_values(ascending=False).index[:n_voxels]
        print(pars_gd.loc[r2_mask].copy())
        
        pars = pars_gd.loc[r2_mask]

        train_data = train_data[r2_mask].astype(np.float32)

        model.init_pseudoWWT(stimulus_range, pars)   

        residfit = ResidualFitter(model, train_data, train_paradigm.astype(np.float32),
                                parameters=pars)

        omega, dof = residfit.fit(init_sigma2=10.0,
                init_dof=10.0,
                method='t',
                learning_rate=0.05,
                max_n_iterations=20000)

        print('DOF', dof)

        pdf = model.get_stimulus_pdf(test_data[r2_mask].astype(np.float32), omega=omega, dof=dof, stimulus_range=stimulus_range, normalize=False)
        keys.append(test_run)
        pdfs.append(pdf)

        E_n1 = np.exp((pdf * pdf.columns.get_level_values('x')).div(pdf.sum(axis=1), axis=0).sum(axis=1))
        E_n2 = np.exp((pdf * pdf.columns.get_level_values('y')).div(pdf.sum(axis=1), axis=0).sum(axis=1))

        print(pd.concat((test_paradigm, E_n1, E_n2), axis=1).corr())

    pdf = pd.concat(pdfs, keys=keys, names=['run'])

    pdf.to_csv(op.join(target_dir, f'sub-{subject}_mask-{roi}_n_voxels-{n_voxels}_pdf.tsv'), sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str)
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--two_dimensional', action='store_true')
    parser.add_argument('--mixture_model', action='store_true')
    parser.add_argument('--same_rfs', action='store_true')
    parser.add_argument('--n_voxels', type=int)
    parser.add_argument('--roi', default='NPC_R')
    parser.add_argument('--bids_folder', default='~/data/ds-dnumrisk')
    args = parser.parse_args()

    main(args.subject, smoothed=args.smoothed, bids_folder=args.bids_folder,
         two_dimensional=args.two_dimensional,
         mixture_model=args.mixture_model, same_rfs=args.same_rfs,
            n_voxels=args.n_voxels,
         roi=args.roi)