# run AFTER transform_npc.py

import argparse
import os
import os.path as op
import numpy as np
from nilearn import surface
from neuropythy.freesurfer import subject as fs_subject
from neuropythy.io import load, save
from neuropythy.mri import (is_image, is_image_spec, image_clear, to_image)


def main(subject, bids_folder, hemi, roi):
    ses_anat = 1
    target_subject = f'sub-{subject}'
    
    if hemi == 'R':
        fs_hemi = 'rh'
    elif hemi == 'L':
        fs_hemi = 'lh'

    target_dir = op.join(bids_folder, 'derivatives', f'{roi}_masks', target_subject)

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    fsnative_fn = op.join(target_dir, f'{target_subject}_desc-NPC_{hemi}_space-fsnative_hemi-{fs_hemi}.{roi}.gii')
    mask_data = surface.load_surf_data(fsnative_fn).astype(bool)

    subjects_dir = op.join(bids_folder, 'derivatives', 'freesurfer')


    target_fn = op.join(target_dir, f'sub-{subject}_space-T1w_desc-NPC_{hemi}.nii.gz')
    sub = fs_subject(op.join(bids_folder, 'derivatives', 'freesurfer', f'sub-{subject}'))
    im = load(op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}',f'ses-{ses_anat}',
    'anat', f'sub-{subject}_ses-{ses_anat}_desc-preproc_T1w.nii.gz'))
    im = to_image(image_clear(im, fill=0.0), dtype=np.int)

    print('Generating volume...')
    if hemi == 'L':
        mask_data = (mask_data, np.zeros(sub.rh.vertex_count))
    elif hemi == 'R':
        mask_data = (np.zeros(sub.lh.vertex_count), mask_data)

    new_im = sub.cortex_to_image(mask_data, # must give datarray for left and right (here: one is mask and other array with only 0s/False)
            im,
            hemi=None,
            method='nearest',
            fill=0.0)

    print('Exporting volume file: %s' % target_fn)
    save(target_fn, new_im)
    print('surface_to_image complete!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder', default='/Volumes/mrenkeED/data/ds-dnumrisk')
    parser.add_argument('--hemi', default='R')
    parser.add_argument('--roi', default ='ips')
    args = parser.parse_args()

    main(args.subject,  bids_folder=args.bids_folder, hemi = args.hemi, roi = args.roi)