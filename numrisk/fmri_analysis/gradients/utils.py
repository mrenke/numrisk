#import cortex
from nilearn import image
import numpy as np
import os.path as op
import nibabel as nib
import os
from nilearn import signal
import pandas as pd
from nipype.interfaces.freesurfer import SurfaceTransform # needs the fsaverage & fsaverage5 in ..derivatives/freesurfer folder!
from nilearn import datasets

# for plotting surface map
from brainspace.utils.parcellation import map_to_labels
from  nilearn.datasets import fetch_surf_fsaverage
import nilearn.plotting as nplt
import matplotlib.pyplot as plt
#from utils_02 import get_events_confounds

def cleanTS(sub, ses, remove_task_effects = False, runs = range(1, 7),space = 'fsaverage5', bids_folder='/Users/mrenke/data/ds-dnumrisk', task = 'magjudge'):
    # load in data as timeseries and regress out confounds (for each run sepeprately)

    fmriprep_confounds_include = ['global_signal', 'dvars', 'framewise_displacement', 'trans_x',
                                    'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                    'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'cosine00', 'cosine01', 'cosine02'
                                    ] # 

    # check if fsaverage5.gii exists, if not, perform fsavTofsav5 [fsnative should automatically have been produced during fmriprep]
    ex_file = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func', 
            f'sub-{sub}_ses-{ses}_task-{task}_run-1_space-{space}_hemi-L_bold.func.gii')   
    if (os.path.exists(ex_file) == False) & (space == 'fsaverage5'):
        print(f'sub-{sub} fsaverage5.gii missing, fsavTofsav5 will be performed')
        fsavTofsav5(sub,ses, bids_folder)

    # get number of vertices
    if space == 'fsaverage5':
        number_of_vertex = 20484  # 'fsaverage5', 10242 * 2
    elif space == 'fsnative': # takes way to long to estimate CC
        timeseries = [None] * 2
        for i, hemi in enumerate(['L', 'R']): # have to load in both hemispheres to get the number of vertices (can be different for L&R)
            ex_file = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func', 
            f'sub-{sub}_ses-{ses}_task-{task}run-1_space-{space}_hemi-L_bold.func.gii')
            timeseries[i] = nib.load(ex_file).agg_data()
        timeseries = np.vstack(timeseries)
        number_of_vertex = timeseries.shape[0]

    # loop over runs and concatenate timeseries
    clean_ts_runs = np.empty([number_of_vertex,0])
    for run in runs:
        timeseries = [None] * 2
        for i, hemi in enumerate(['L', 'R']):
            filename = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func', 
            f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-{space}_hemi-{hemi}_bold.func.gii')        
            timeseries[i] = nib.load(filename).agg_data()
        timeseries = np.vstack(timeseries) # (20484, 135)

        fmriprep_confounds_file = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func', f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-confounds_timeseries.tsv')
        fmriprep_confounds = pd.read_table(fmriprep_confounds_file)[fmriprep_confounds_include] 
        fmriprep_confounds= fmriprep_confounds.fillna(method='bfill')

        if remove_task_effects:
            dm = get_events_confounds(sub, ses, run, bids_folder)
            regressors_to_remove = pd.concat([dm.reset_index(drop=True), fmriprep_confounds], axis=1)
        else:
            regressors_to_remove = fmriprep_confounds
        clean_ts = signal.clean(timeseries.T, confounds=regressors_to_remove).T

        clean_ts_runs = np.append(clean_ts_runs, clean_ts, axis=1)

    return clean_ts_runs

def fsavTofsav5(sub,ses = 1,bids_folder='/Volumes/mrenkeED/data/ds-dnumrisk',task = 'magjudge'):
    # requires fsaverage and fsaverage5 directory in bids_folder/derivatives/freesurfer !
    runs = range(1,7)
    
    for run in runs:
        for hemi in ['L', 'R']:
            sxfm = SurfaceTransform(subjects_dir=op.join(bids_folder,'derivatives','freesurfer'))
            in_file = f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-fsaverage_hemi-{hemi}_bold.func.gii'
            in_file_path = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{sub}',f'ses-{ses}','func',in_file)
            out_file = f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-fsaverage5_hemi-{hemi}_bold.func.gii'
            out_file_path = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{sub}',f'ses-{ses}','func',out_file)

            sxfm.inputs.source_file = in_file_path
            sxfm.inputs.out_file = out_file_path

            sxfm.inputs.source_subject = 'fsaverage'
            sxfm.inputs.target_subject = 'fsaverage5'

            if hemi == 'L':
                sxfm.inputs.hemi = 'lh'
            elif hemi == 'R':
                sxfm.inputs.hemi = 'rh'

            r = sxfm.run()


def saveGradToNPFile(grad, sub,ses, specification='',bids_folder='/Users/mrenke/data/ds-dnumrisk'):
    target_dir = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses}')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    for g, n_grad  in enumerate(range(1,1+np.shape(grad)[0])):
        np.save(op.join(target_dir,f'grad{n_grad}{specification}.npy'), grad[g])

def npFileTofs5Gii(sub,ses, specification='',bids_folder='/Users/mrenke/data/ds-dnumrisk', gradient_Ns = [1,2,3], task = 'magjudge' ):
    target_dir = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses}')

    for n_grad in gradient_Ns:
        grad = np.load(op.join(target_dir, f'grad{n_grad}{specification}.npy'))
        grad = np.split(grad,2) # for i, hemi in enumerate(['L', 'R']): --> left first

        for h, hemi in enumerate(['L', 'R']):    

            gii_im_datar = nib.gifti.gifti.GiftiDataArray(data=grad[h])
            gii_im = nib.gifti.gifti.GiftiImage(darrays= [gii_im_datar])

            out_file = op.join(target_dir, f'sub-{sub}_ses-{ses}_task-{task}_space-fsaverage5_hemi-{hemi}_grad{n_grad}{specification}.surf.gii')
            gii_im.to_filename(out_file) # https://nipy.org/nibabel/reference/nibabel.spatialimages.html


def get_basic_mask():
    atlas = datasets.fetch_atlas_surf_destrieux()
    regions = atlas['labels'].copy()
    masked_regions = [b'Medial_wall', b'Unknown']
    masked_labels = [regions.index(r) for r in masked_regions]
    for r in masked_regions:
        regions.remove(r)
    labeling = np.concatenate([atlas['map_left'], atlas['map_right']])
    labeling_noParcel = np.arange(0,len(labeling),1,dtype = int)     # Map gradients to original parcels
    mask = ~np.isin(labeling, masked_labels)
    return mask, labeling_noParcel

def plot_GM12_from_sum_npfile(file = 'gm_av50_unfiltered_aligned-marg.npy',bids_folder='/Volumes/mrenkeED/data/ds-dnumrisk',grad_folder = 'derivatives/gradients', colorbar=False):

    fsaverage = fetch_surf_fsaverage()

    grad = np.load(op.join(bids_folder,grad_folder,file))
    

    grad1 = np.split(grad[0],2) # for i, hemi in enumerate(['L', 'R']): --> left first
    grad2 = np.split(grad[1],2)
    grad3 = np.split(grad[2],2)
    
    grad1_r = grad1[1] # 0 = left, 1 = right
    grad2_r = grad2[1]  
    grad3_r = grad3[1]

    figure, axes = plt.subplots(nrows=1, ncols=3, subplot_kw=dict(projection='3d'))
    nplt.plot_surf_stat_map(surf_mesh=fsaverage.infl_right, colorbar = colorbar,stat_map=grad1_r,cmap='viridis',view='medial',axes=axes[0])
    axes[0].set(title='grad 1')
    nplt.plot_surf_stat_map(surf_mesh=fsaverage.infl_right, colorbar = colorbar,stat_map=grad2_r,cmap='viridis',view='medial',axes=axes[1])
    axes[1].set(title='grad 2')
    nplt.plot_surf_stat_map(surf_mesh=fsaverage.infl_right, colorbar = colorbar,stat_map=grad3_r,cmap='viridis',view='medial',axes=axes[2])
    axes[2].set(title='grad 3')
    figure.suptitle(file)