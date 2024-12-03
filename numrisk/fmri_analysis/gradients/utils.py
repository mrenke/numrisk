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
from numrisk.fmri_analysis.gradients.utils_old import get_events_confounds,surfTosurf

def cleanTS(sub, ses =1, remove_task_effects = False, runs = range(1, 7),space = 'fsaverage5', bids_folder='/Users/mrenke/data/ds-dnumrisk', task = 'magjudge'):
    # load in data as timeseries and regress out confounds (for each run sepeprately)

    fmriprep_confounds_include = ['global_signal', 'dvars', 'framewise_displacement', 'trans_x',
                                    'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                    'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'cosine00', 'cosine01', 'cosine02'
                                    ] # 

    # check if {space}.gii exists, if not, perform fsavTofsav5 [fsnative should automatically have been produced during fmriprep]
    ex_file = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func', 
            f'sub-{sub}_ses-{ses}_task-{task}_run-1_space-{space}_hemi-L_bold.func.gii')   
    if (os.path.exists(ex_file) == False): #& (space == 'fsaverage5'):
        print(f'sub-{sub} {space}.gii missing, fsnativeTo{space}] will be performed')
        surfTosurf(sub,source_space='fsnative',target_space=space,ses=1,bids_folder=bids_folder)

    # get number of vertices
    if space == 'fsaverage5':
        number_of_vertex = 20484  # 'fsaverage5', 10242 * 2
    elif space == 'fsaverage':
        number_of_vertex = 327684  # 'fsaverage', 163842 * 2
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
        try:
            timeseries = [None] * 2
            for i, hemi in enumerate(['L', 'R']):
                filename = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func', 
                f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-{space}_hemi-{hemi}_bold.func.gii')        
                timeseries[i] = nib.load(filename).agg_data()
            timeseries = np.vstack(timeseries) # (20484, 135)

            fmriprep_confounds_file = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func', f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-confounds_timeseries.tsv')
            fmriprep_confounds = pd.read_table(fmriprep_confounds_file)[fmriprep_confounds_include] 
            #fmriprep_confounds= fmriprep_confounds.fillna(method='bfill') # deprecated
            fmriprep_confounds= fmriprep_confounds.bfill()

            if remove_task_effects:
                dm = get_events_confounds(sub, ses, run, bids_folder)
                regressors_to_remove = pd.concat([dm.reset_index(drop=True), fmriprep_confounds], axis=1)
            else:
                regressors_to_remove = fmriprep_confounds
            clean_ts = signal.clean(timeseries.T, confounds=regressors_to_remove).T

            clean_ts_runs = np.append(clean_ts_runs, clean_ts, axis=1)
        except:
            print(f'sub-{sub}, run-{run} makes problems (prob. confounds ts not there) \n skipping that run') # for sub 5,47,53,62

    return clean_ts_runs

def fit_correlation_matrix_unfiltered(sub,bids_folder):
    mask, labeling_noParcel = get_basic_mask()
    clean_ts = cleanTS(sub, bids_folder=bids_folder) # checks if fsav5-file exists, if not, creates it
    seed_ts = clean_ts[mask]

    from nilearn.connectome import ConnectivityMeasure
    correlation_measure = ConnectivityMeasure(kind='correlation')
    cm = correlation_measure.fit_transform([seed_ts.T])[0] #correlation_matrix_noParcel
    print(f'sub-{sub}: raw connectivity matrix estimated')    
    np.save(op.join(bids_folder, 'derivatives', 'correlation_matrices', f'sub-{sub}_unfiltered.npy'), cm)

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

def get_glasser_parcels(base_folder='/mnt_03/diverse_neuralData/atlases_parcellations', space='fsaverage'):
    atlas_left = nib.load(op.join(base_folder,f'lh_space-{space}.HCPMMP1.gii')).agg_data()
    atlas_right =  nib.load(op.join(base_folder,f'rh_space-{space}.HCPMMP1.gii')).agg_data()

    labeling = np.concatenate([(atlas_left+1000), (atlas_right+2000)]) # unique labels for left and right!
    mask = ~np.isin(labeling, [1000,2000]) # non-cortex region (unknow and medial wall) have label 0, hence 1000 & 2000 in my variation labels L/R
    # mask.sum() == len(labeling[(labeling != 1000) & (labeling != 2000)]) 
    return mask, labeling

def get_glasser_CAatlas_mapping(datadir = '/mnt_03/diverse_neuralData/atlases_parcellations/ColeAnticevicNetPartition'):
    glasser_CAatlas_mapping = pd.read_csv(op.join(datadir,'cortex_parcel_network_assignments.txt'),header=None)
    glasser_CAatlas_mapping.index.name = 'glasser_parcel'
    glasser_CAatlas_mapping = glasser_CAatlas_mapping.rename({0:'ca_network'},axis=1)

    CAatlas_names = pd.read_csv(op.join(datadir,'network_label-names.csv'),index_col=0)
    CAatlas_names = CAatlas_names.set_index('Label Number')
    CAatlas_names = CAatlas_names.sort_index(level='Label Number')
    
    return glasser_CAatlas_mapping, CAatlas_names
