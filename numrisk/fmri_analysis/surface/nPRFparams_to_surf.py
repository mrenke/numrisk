import argparse
import os.path as op
from numrisk.utils.data import Subject
from nilearn import surface
import nibabel as nb
#from numrisk.fmri_analysis.encoding_model.fit_nprf import get_key_target_dir
#from tqdm import tqdm
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

def main(subject_id, session, bids_folder,smoothed, denoise=True):
    
    sub = Subject(subject_id, bids_folder=bids_folder)
    subject = f'{int(subject_id):02d}'

    surfinfo = sub.get_surf_info_fs()

    par_keys = ['mu', 'sd', 'amplitude', 'baseline', 'r2'] # 'cvr2'  ??

    prf_pars_volume = sub.get_prf_parameters_volume(session, smoothed=smoothed, denoise=denoise, keys=par_keys,
                                                    return_image=True, cross_validated=False) 

    #_, target_dir = get_key_target_dir(f'{int(subject):02d}', session, bids_folder, smoothed, denoise=denoise, pca_confounds=False, retroicor=retroicor, natural_space=natural_space)    
    dir = 'encoding_model'
    if denoise:
        dir += '.denoise'  
    if smoothed:
        dir += '.smoothed' 
    target_dir = op.join(bids_folder, 'derivatives', dir,f'sub-{subject}', f'ses-{session}','func')

    print(f'Writing to {target_dir}')

    for hemi in ['L', 'R']:
        samples = surface.vol_to_surf(prf_pars_volume, surfinfo[hemi]['outer'], inner_mesh=surfinfo[hemi]['inner'])
        fs_hemi = 'lh' if hemi == 'L' else 'rh'

        for ix, par in enumerate(par_keys):
            im = nb.gifti.GiftiImage(darrays=[nb.gifti.GiftiDataArray(samples[:, ix])])
            target_fn =  op.join(target_dir, f'sub-{subject}_ses-{session}_desc-{par}.optim.nilearn_space-fsnative_hemi-{hemi}.func.gii')
            nb.save(im, target_fn)

            transform_fsaverage(target_fn, fs_hemi, f'sub-{subject}', bids_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--session', default=1)
    parser.add_argument('--bids_folder', default='/mnt_03/ds-dnumrisk')
    parser.add_argument('--smoothed', action='store_true')
    #parser.add_argument('--denoise', action='store_true')

    args = parser.parse_args()

    main(args.subject, args.session, bids_folder=args.bids_folder, smoothed=args.smoothed) # ,denoise=args.denoise