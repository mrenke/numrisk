import argparse
import os
#import pingouin
import numpy as np
import os.path as op
import pandas as pd
from braincoder.optimize import ResidualFitter
from braincoder.models import GaussianPRF
from braincoder.utils import get_rsq
from numrisk.utils import Subject
import numpy as np

stimulus_range = np.linspace(0, 6, 1000)
# stimulus_range = np.log(np.arange(400))
mask = 'NPC_R'
space = 'T1w'

def main(subject, smoothed, pca_confounds, denoise = True, n_voxels=100, bids_folder='/data',
        mask='NPC_R', retroicor = False,
        split_data = None):


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

    target_dir = f'decoded_pdfs.volume{split_data}'

    if denoise:
        target_dir += '.denoise'

    if smoothed:
        target_dir += '.smoothed'

    if pca_confounds:
        target_dir += '.pca_confounds'

    target_dir = op.join(bids_folder, 'derivatives', target_dir, f'sub-{subject}', 'func')
    
    print(denoise, target_dir)

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    print(1)
    sub = Subject(subject, bids_folder)
    paradigm = sub.get_behavior_magjudge(drop_no_responses=False, runs=runs)
    paradigm['log(n1)'] = np.log(paradigm['n1'])
    paradigm = paradigm.droplevel(['subject'])
    paradigm = paradigm['log(n1)']

    data = sub.get_single_trial_volume(session, roi=mask,split_data=split_data, smoothed=smoothed, pca_confounds=pca_confounds, denoise=denoise, retroicor=retroicor).astype(np.float32)
    data.index = paradigm.index
    print(data)

    pdfs = []

    for test_run in runs:

        test_data, test_paradigm = data.xs(test_run, 0, 'run').copy(), paradigm.xs(test_run, 0, 'run').copy()
        train_data, train_paradigm = data.drop(test_run, level='run').copy(), paradigm.drop(test_run, level='run').copy()

        pars = sub.get_prf_parameters_volume(session, cross_validated=True,
                smoothed=smoothed, pca_confounds=pca_confounds,
                denoise=denoise,
                roi=mask,
                run=test_run)
        
        print(pars)

        model = GaussianPRF(parameters=pars)
        pred = model.predict(paradigm=train_paradigm.astype(np.float32))

        r2 = get_rsq(train_data, pred)
        print(r2.describe())
        r2_mask = r2.sort_values(ascending=False).index[:n_voxels]

        train_data = train_data[r2_mask]
        test_data = test_data[r2_mask]

        print(r2.loc[r2_mask])
        model.apply_mask(r2_mask)

        model.init_pseudoWWT(stimulus_range, model.parameters)
        residfit = ResidualFitter(model, train_data,
                                  train_paradigm.astype(np.float32))

        omega, dof = residfit.fit(init_sigma2=10.0,
                init_dof=10.0,
                method='t',
                learning_rate=0.005,
                max_n_iterations=20000)

        print('DOF', dof)

        bins = stimulus_range.astype(np.float32)

        pdf = model.get_stimulus_pdf(test_data, bins,
                model.parameters,
                omega=omega,
                dof=dof)


        print(pdf)
        E = (pdf * pdf.columns).sum(1) / pdf.sum(1)

        print(pd.concat((E, test_paradigm), axis=1))
        #print(pingouin.corr(E, test_paradigm['log(n1)']))

        pdfs.append(pdf)

    pdfs = pd.concat(pdfs)

    target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_mask-{mask}_nvoxels-{n_voxels}_space-{space}_pars.tsv')
    pdfs.to_csv(target_fn, sep='\t')


if __name__ == '__main__':
    print('ues')
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    #parser.add_argument('session', default=None, type=int)
    parser.add_argument('--bids_folder', default='/data')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--pca_confounds', action='store_true')
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--mask', default='NPC_R')
    parser.add_argument('--n_voxels', default=100, type=int)
    parser.add_argument('--split_data', default=None)

    args = parser.parse_args()

    main(subject=args.subject,  smoothed=args.smoothed, pca_confounds=args.pca_confounds, denoise=args.denoise,
            n_voxels=args.n_voxels,
            bids_folder=args.bids_folder, mask=args.mask,
            split_data = args.split_data)
