# take average surface (*....fsaverag*.gii file) IPS mask and transform it to native (subejct specific T1) space
import argparse
import os
import os.path as op
from nipype.interfaces.freesurfer import SurfaceTransform


def main(subject, bids_folder, hemi, roi):

    subjects_dir = op.join(bids_folder, 'derivatives', 'freesurfer')

    target_subject = f'sub-{subject}'

    if hemi == 'R':
        fs_hemi = 'rh'
    elif hemi == 'L':
        fs_hemi = 'lh'
    
    in_file = op.join(bids_folder, f'derivatives/surface_masks/desc-NPC_{hemi}_space-fsaverage_hemi-{fs_hemi}.label.gii')

    target_dir = op.join(bids_folder, 'derivatives', f'{roi}_masks', target_subject)

    if not op.exists(target_dir):
        os.makedirs(target_dir)
    
    # NPC = numerosity parietal cortex ~ IPS
    out_file = op.join(target_dir, f'{target_subject}_desc-NPC_{hemi}_space-fsnative_hemi-{fs_hemi}.{roi}.gii')

    #'sub-01_desc-NPC_R_space-fsnative_hemi-rh.label.gii')  
    #out_file = op.join(subjects_dir, target_subject, 'surf', f'{hemi}.{roi}.mgz') #f'sub-{subject}',


    def transform_surface(in_file,
            out_file, 
            target_subject,
            hemi,
            source_subject='fsaverage'):

        sxfm = SurfaceTransform(subjects_dir=subjects_dir)
        sxfm.inputs.source_file = in_file
        sxfm.inputs.out_file = out_file
        sxfm.inputs.source_subject = source_subject
        sxfm.inputs.target_subject = target_subject
        sxfm.inputs.hemi = fs_hemi
        # sxfm.cmdline #helps with debugging 
        r = sxfm.run()
        return r

    transform_surface(in_file, out_file, target_subject, hemi)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder', default='/Volumes/mrenkeED/data/ds-dnumrisk')
    parser.add_argument('--hemi', default='R')
    parser.add_argument('--roi', default ='ips')
    args = parser.parse_args()

    main(args.subject,  bids_folder=args.bids_folder, hemi = args.hemi, roi = args.roi)