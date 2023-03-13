import re
import os
import os.path as op
import shutil
import argparse
import pandas as pd
import glob
from nilearn import image
import numpy as np
import json

def main(subject, session, bids_folder='/Volumes/mrenkeED/data/ds-dnumr', task = 'magjudge'):
    sourcedata_root = op.join(bids_folder, 'sourcedata', 'mri',
    f'SNS_MRI_DNUMR_S{subject:05d}_{session:02d}')

    if session == 1:
        # # *** ANATOMICAL DATA ***
        # So not vt1w, which are reconstructed at different angle
        t1w = glob.glob(op.join(sourcedata_root, '*_t1w*.nii'))
        print(op.join(sourcedata_root, '*_t1w*.nii'))
        assert(len(t1w) != 0), "No T1w {t1w}"

        #flair = glob.glob(op.join(sourcedata_root, '*flair*.nii'))
        #assert(len(flair) == 1), f"More than 1/no FLAIR {flair}"

        target_dir = op.join(bids_folder, f'sub-{subject:02d}', f'ses-{session}', 'anat')
        if not op.exists(target_dir):
            os.makedirs(target_dir)

        if len(t1w) == 1:
            shutil.copy(t1w[0], op.join(target_dir, f'sub-{subject:02d}_ses-{session}_T1w.nii'))
        else:
            for run0, t in enumerate(t1w):
                print(t)
                shutil.copy(t, op.join(target_dir, f'sub-{subject:02d}_ses-{session}_run-{run0+1}_T1w.nii'))


        #shutil.copy(flair[0], op.join(target_dir, f'sub-{subject:02d}_ses-{session}_FLAIR.nii'))

    # # *** FUNCTIONAL DATA ***
    with open(op.abspath('./bold_template.json'), 'r') as f:
        json_template = json.load(f)
        print(json_template)

    reg = re.compile('.*run(?P<run>[0-9]+).*')
    funcs = glob.glob(op.join(sourcedata_root, '*run*.nii'))

    runs = [int(reg.match(fn).group(1)) for fn in funcs]

    target_dir = op.join(bids_folder, f'sub-{subject:02d}', f'ses-{session}', 'func')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    for run, fn in zip(runs, funcs):
        shutil.copy(fn, op.join(target_dir, f'sub-{subject:02d}_ses-{session}_task-{task}_run-{run}_bold.nii'))

        json_sidecar = json_template
        json_sidecar['PhaseEncodingDirection'] = 'i' if (run % 2 == 1) else 'i-'

        with open(op.join(target_dir, f'sub-{subject:02d}_ses-{session}_task-{task}_run-{run}_bold.json'), 'w') as f:
            json.dump(json_sidecar, f)


    # *** physio logfiles ***
    physiologs = glob.glob(op.join(sourcedata_root, '*run*scanphyslog*.log'))
    runs = [int(reg.match(fn).group(1)) for fn in physiologs]

    for run, fn in zip(runs, physiologs):
        shutil.copy(fn, op.join(target_dir, f'sub-{subject:02d}_ses-{session}_task-{task}_run-{run}_physio.log'))

    # *** Fieldmaps ***
    func_dir = op.join(bids_folder, f'sub-{subject:02d}', f'ses-{session}', 'func')
    target_dir = op.join(bids_folder, f'sub-{subject:02d}', f'ses-{session}', 'fmap')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    with open(op.abspath('./fmap_template.json'), 'r') as f:
        json_template = json.load(f)
        print(json_template)
  
    for target_run in range(1, 7):
        bold =  op.join(func_dir, f'sub-{subject:02d}_ses-{session}_task-{task}_run-{target_run}_bold.nii')

        if not op.exists(bold):
            print(f"Skipping EPI search for run {target_run}")
            continue


        source_run = target_run + 1
        index_slice = slice(5, 10)
        if source_run == 7:
            source_run = 5
            index_slice = slice(-10, -5)
        
        direction = 'RL' if (source_run % 2 == 1) else 'LR'

        epi = op.join(func_dir, f'sub-{subject:02d}_ses-{session}_task-{task}_run-{source_run}_bold.nii')
        
        if not op.exists(epi):
            print(f"PROBLEM with target run {target_run}")
            if target_run % 2 == 0:
                potential_source_runs = np.arange(1, 7, 2)
            else:
                potential_source_runs = np.arange(2, 7, 2)

            distances = np.abs(target_run - potential_source_runs)
            potential_source_runs = potential_source_runs[np.argsort(distances)]

            for source_run in potential_source_runs:
                print(source_run)
                epi = op.join(func_dir, f'sub-{subject:02d}_ses-{session}_task-{task}_run-{source_run}_bold.nii')
                if op.exists(epi):
                    print(f'Using {source_run} as EPI for target {target_run}')
                    print(epi)
                    if (source_run > target_run):
                        index_slice = slice(5, 10)
                    else:
                        index_slice = slice(-10, -5)
                    print(f"Index slice: {index_slice}")
                    break

        epi = image.index_img(epi, index_slice)

        target_fn = op.join(target_dir, f'sub-{subject:02d}_ses-{session}_dir-{direction}_run-{target_run}_epi.nii')
        epi.to_filename(target_fn)

        json_sidecar = json_template
        json_sidecar['PhaseEncodingDirection'] = 'i' if (source_run % 2 == 1) else 'i-'
        json_sidecar['IntendedFor'] = f'ses-{session}/func/sub-{subject:02d}_ses-{session}_task-{task}_run-{target_run}_bold.nii'

        with open(op.join(target_dir, f'sub-{subject:02d}_ses-{session}_dir-{direction}_run-{target_run}_epi.json'), 'w') as f:
            json.dump(json_sidecar, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=int)
    parser.add_argument('--session', type=int, default=1)
    parser.add_argument('--bids_folder', default='/Volumes/mrenkeED/data/ds-dnumr')
    args = parser.parse_args()

    main(args.subject, args.session, bids_folder=args.bids_folder)
