import argparse
import os.path as op
from numrisk.utils.data import Subject
from nilearn import surface
import nibabel as nb
#from stress_risk.fmri_analysis.encoding_model.fit_nprf import get_key_target_dir
from tqdm import tqdm
from nipype.interfaces.freesurfer import SurfaceTransform


def transform_fsaverage(in_file, fs_hemi, source_subject, bids_folder, target_space = 'fsaverage'):

        subjects_dir = op.join(bids_folder, 'derivatives', 'freesurfer')

        sxfm = SurfaceTransform(subjects_dir=subjects_dir)
        sxfm.inputs.source_file = in_file
        sxfm.inputs.out_file = in_file.replace('fsnative', target_space)
        sxfm.inputs.source_subject = source_subject
        sxfm.inputs.target_subject = target_space
        sxfm.inputs.hemi = fs_hemi

        r = sxfm.run()
        return r

def main(subject, session, bids_folder, param,cv=False, smoothed=False): # parameter: default='r2')
    
    sub = f'{int(subject):02d}'
    subject = Subject(subject, bids_folder=bids_folder)

    surfinfo = subject.get_surf_info_fs()

    par_keys = [param] # 'cvr2'  ?? /'mu', 'sd', 'amplitude', 'baseline', 'r2'

    key = 'encoding_model.cv.denoise' if cv else 'encoding_model.denoise'
    if smoothed:
        key += '.smoothed'
    target_dir = op.join(bids_folder,'derivatives',key,f'sub-{sub}', f'ses-{session}', 'func')

    for hemi in ['L', 'R']:
        fs_hemi = 'lh' if hemi == 'L' else 'rh'

        for ix, par in enumerate(par_keys):
            prf_pars_volume_im = op.join(target_dir, f'sub-{sub}_ses-{session}_desc-{par}.optim_space-T1w_pars.nii.gz')
            print(prf_pars_volume_im)
            samples = surface.vol_to_surf(prf_pars_volume_im, surfinfo[hemi]['outer'], inner_mesh=surfinfo[hemi]['inner'])
            im = nb.gifti.GiftiImage(darrays=[nb.gifti.GiftiDataArray(samples)])
            target_fn =  op.join(target_dir, f'sub-{sub}_ses-{session}_desc-{par}.optim.nilearn_space-fsnative_hemi-{hemi}.func.gii')
            print(f'saving as {target_fn}')

            nb.save(im, target_fn) # save in native surface space
            transform_fsaverage(target_fn, fs_hemi, f'sub-{sub}', bids_folder, target_space = 'fsaverage') # transform to common surface space
            transform_fsaverage(target_fn, fs_hemi, f'sub-{sub}', bids_folder, target_space = 'fsaverage5') # transform to common surface space


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--session', default=1)
    parser.add_argument('--bids_folder', default='/mnt_03/ds-dnumrisk' )
    parser.add_argument('--cv', action='store_true')
    parser.add_argument('--parameter', default='r2')
    parser.add_argument('--smoothed', action='store_true')

    args = parser.parse_args()

    main(args.subject, args.session, bids_folder=args.bids_folder, cv=args.cv ,param = args.parameter,  smoothed=args.smoothed)