import argparse
import os
#import pingouin
import numpy as np
import os.path as op
import pandas as pd
from nilearn import surface
from braincoder.optimize import ResidualFitter
from braincoder.models import GaussianPRF
from braincoder.utils import get_rsq
import numpy as np
from numrisk.utils import Subject
from braincoder.models import GaussianPRF
from braincoder.optimize import ParameterFitter

stimulus_range = np.linspace(0, 6, 1000)
# stimulus_range = np.log(np.arange(400))
space = 'T1w'

def main(subject, smoothed, bids_folder='/data',
    denoise=True, mask='NPC_R'):
    
    session = 1
    runs = range(1, 7)
    n_stim = 2

    sub = Subject(subject, bids_folder)
    subject = f'{int(subject):02d}'

    target_dir = f'decoded_pdfs_stim{n_stim}.volume.cv_vselect'

    if denoise:
        target_dir += '.denoise'

    if smoothed:
        target_dir += '.smoothed'

    target_dir = op.join(bids_folder, 'derivatives', target_dir, f'sub-{subject}', 'func')

    if not op.exists(target_dir):
        os.makedirs(target_dir)


    paradigm = sub.get_behavior_magjudge(drop_no_responses=False, runs=runs)
    paradigm[f'log(n{n_stim})'] = np.log(paradigm[f'n{n_stim}'])
    paradigm = paradigm.droplevel(['subject'])
    paradigm = paradigm[f'log(n{n_stim})'].dropna()

    data = sub.get_single_trial_volume(session, n_stim=n_stim, roi=mask,smoothed=smoothed,denoise=denoise).astype(np.float32)
    data.index = paradigm.index
    print(data)

    pdfs = []

    # SET UP GRID
    mus = np.log(np.linspace(5, 80, 60, dtype=np.float32))
    sds = np.log(np.linspace(2, 30, 60, dtype=np.float32))
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    # select voxels double-loop (drop first loop test_data totally to use it in the decoding loop later)
    cv_r2s = []
    cv_keys = []
    for test_run in runs:

        test_data, test_paradigm = data.loc[test_run].copy(), paradigm.loc[test_run].copy() # this data will only be used in the decoding loop later
        train_data, train_paradigm = data.drop(test_run, level='run').copy(), paradigm.drop(test_run, level='run').copy()

        for test_run2 in train_data.index.unique(level='run'):
            test_data2 = train_data.loc[test_run2].copy()
            test_paradigm2 = train_paradigm.loc[test_run2].copy()
            train_data2 = train_data.drop(test_run2).copy()
            train_paradigm2 = train_paradigm.drop(test_run2, level='run').copy()
            print(test_data2.shape, train_data2.shape, train_paradigm2.shape, test_paradigm2.shape)

            print(train_data2)
            print(train_paradigm2)

            model = GaussianPRF()
            optimizer = ParameterFitter(model, train_data2, train_paradigm2)

            grid_parameters = optimizer.fit_grid(
                mus, sds, amplitudes, baselines, use_correlation_cost=True)
            grid_parameters = optimizer.refine_baseline_and_amplitude(
                grid_parameters, n_iterations=2)

            print(grid_parameters.describe())

            optimizer.fit(init_pars=grid_parameters, learning_rate=.005, store_intermediate_parameters=False, max_n_iterations=10000,
                      r2_atol=0.00001)

            print(optimizer.estimated_parameters.describe())
        
            cv_r2 = get_rsq(test_data2, model.predict(parameters=optimizer.estimated_parameters,
                                                    paradigm=test_paradigm2.astype(np.float32))).to_frame('r2').T

            cv_r2s.append(cv_r2)
            cv_keys.append({'subject':subject, 'session':session, 
            'test_run1':test_run, 'test_run2':test_run2})
    
    cv_r2s = pd.concat(cv_r2s, axis=0)
    cv_r2s.index = pd.MultiIndex.from_frame(pd.DataFrame(cv_keys))
    target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_mask-{mask}_space-{space}_r2s.tsv')
    cv_r2s.to_csv(target_fn, sep='\t')

    cv_r2s = cv_r2s.groupby(['test_run1']).mean()
    print(cv_r2s)

    # decoding loop 
    pdfs = []
    for test_run in runs:

        test_data, test_paradigm = data.loc[test_run].copy(), paradigm.loc[test_run].copy()
        train_data, train_paradigm = data.drop(test_run, level='run').copy(), paradigm.drop(test_run, level='run').copy()

        pars = sub.get_prf_parameters_volume(session, cross_validated=True,
                denoise=denoise, 
                smoothed=smoothed, 
                run=test_run, roi=mask)

        model = GaussianPRF(parameters=pars)

        r2_mask = cv_r2s.loc[test_run] > 0.0
        print(r2_mask)
        print(train_data)

        train_data = train_data.loc[:, r2_mask]
        test_data = test_data.loc[:, r2_mask]

        model.apply_mask(r2_mask)

        model.init_pseudoWWT(stimulus_range, model.parameters)
        residfit = ResidualFitter(model, train_data,
                                  train_paradigm.astype(np.float32))

        omega, dof = residfit.fit(init_sigma2=10.0,
                method='t',
                max_n_iterations=10000)

        print('DOF', dof)

        bins = stimulus_range.astype(np.float32)

        pdf = model.get_stimulus_pdf(test_data, bins,
                model.parameters,
                omega=omega,
                dof=dof)


        print(pdf)
        E = (pdf * pdf.columns).sum(1) / pdf.sum(1)

        #print(pd.concat((E, test_paradigm), axis=1))
        #print(pingouin.corr(E, test_paradigm))

        pdfs.append(pdf)

    pdfs = pd.concat(pdfs)

    target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_mask-{mask}_space-{space}_pars.tsv')
    pdfs.to_csv(target_fn, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    #parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--mask', default='NPC_R')

    args = parser.parse_args()

    main(args.subject, args.smoothed, # args.session,
            denoise=args.denoise,

            bids_folder=args.bids_folder, mask=args.mask,
            )