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

def get_events_confounds(sub, ses, run, bids_folder='/Users/mrenke/data/ds-dnumrisk',task='magjudge' ):
    tr = 2.3 # repetition Time
    n = 188 # number of slices # adjust this important!!

    df_events = pd.read_csv(op.join(bids_folder, f'sub-{sub}', f'ses-{ses}', 'func', f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.tsv'.format(sub=sub, ses=ses)), sep='\t') # before run was ot interated over (run-1)
    
    stimulus1 = df_events.loc[df_events['trial_type'] == 'stimulus 1', ['onset', 'trial_nr', 'trial_type', 'n1']]
    stimulus1['duration'] = 0.6 + 0.8
    stimulus1['onset'] = stimulus1['onset'] - 0.8 # cause we want to take the onset of the piechart 
    stimulus1['stim_order'] = int(1)
    stimulus1_int = stimulus1.copy()
    stimulus1_int['trial_type'] = 'stimulus1_int'
    stimulus1_int['modulation'] = 1
    stimulus1_mod= stimulus1.copy()
    stimulus1_mod['trial_type'] = 'stimulus1_mod'
    stimulus1_mod['modulation'] = stimulus1['n1']

    #choices = df_events.xs('choice', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_nr', 'trial_type', 'n2']]
    choices = df_events.loc[df_events['trial_type'] == 'choice']

    #stimulus2 = df_events.xs('stimulus 2', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_nr', 'trial_type', 'n2']]
    stimulus2 = df_events.loc[df_events['trial_type'] == 'stimulus 2', ['onset', 'trial_nr', 'trial_type', 'n2']]
    stimulus2 = stimulus2.set_index('trial_nr')
    stimulus2['duration'] = choices.set_index('trial_nr')['onset']- stimulus2['onset'] + 0.6 # 0.6 + 0.6 ## looked at the data, is is different for stim 1 and 2... ?!!
    stimulus2['onset'] = stimulus2['onset'] - 0.6
    stimulus2['stim_order'] = int(2)
    stimulus2_int = stimulus2.copy()
    stimulus2_int['trial_type'] = 'stimulus2_int'
    stimulus2_int['modulation'] = 1
    stimulus2_mod= stimulus2.copy()
    stimulus2_mod['trial_type'] = 'stimulus2_mod'
    stimulus2_mod['modulation'] = stimulus2['n2']
    stimulus2_int.reset_index(inplace=True)
    stimulus2_mod.reset_index(inplace=True)

    events = pd.concat((stimulus1_int,stimulus1_mod, stimulus2_int, stimulus2_mod)).set_index(['trial_nr','stim_order'],append=True).sort_index()

    onsets = events[['onset', 'duration', 'trial_type', 'modulation']]
    onsets['onset'] = ((onsets['onset']+tr/2.) // 2.3) * 2.3

    frametimes = np.linspace(tr/2., (n - .5)*tr, n)

    from nilearn.glm.first_level import make_first_level_design_matrix
    dm = make_first_level_design_matrix(frametimes, onsets, 
                                        hrf_model='spm + derivative + dispersion', 
                                        oversampling=100.,drift_order=1, 
                                        drift_model=None).drop('constant', axis=1)
    dm /= dm.max()
    print('Design matrix created to remove task effects. shape:')
    print(dm.shape)
    return dm

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


def surfTosurf(sub,source_space, target_space,     # requires both space directories in bids_folder/derivatives/freesurfer !
                ses = 1, runs=range(1,7), bids_folder='/Volumes/mrenkeED/data/ds-dnumrisk',task = 'magjudge'):

    for run in runs:
        for hemi in ['L', 'R']:
            sxfm = SurfaceTransform(subjects_dir=op.join(bids_folder,'derivatives','freesurfer'))
            in_file = f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-{source_space}_hemi-{hemi}_bold.func.gii'
            in_file_path = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{sub}',f'ses-{ses}','func',in_file)
            out_file = f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-{target_space}_hemi-{hemi}_bold.func.gii'
            out_file_path = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{sub}',f'ses-{ses}','func',out_file)

            sxfm.inputs.source_file = in_file_path
            sxfm.inputs.out_file = out_file_path

            sxfm.inputs.source_subject = source_space if source_space != 'fsnative' else f'sub-{int(sub):02d}'
            sxfm.inputs.target_subject = target_space if target_space != 'fsnative' else f'sub-{int(sub):02d}'

            if hemi == 'L':
                sxfm.inputs.hemi = 'lh'
            elif hemi == 'R':
                sxfm.inputs.hemi = 'rh'

            r = sxfm.run()

def saveGradToNPFile(grad, sub, specification='',bids_folder='/Users/mrenke/data/ds-dnumrisk',
                     space='fsaverage5'):
    target_dir = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub}') # , f'ses-{ses}')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    for g, n_grad  in enumerate(range(1,1+np.shape(grad)[0])):
        np.save(op.join(target_dir,f'grad{n_grad}_space-{space}{specification}.npy'), grad[g])

def npFileTofs5Gii(sub, specification='',bids_folder='/Users/mrenke/data/ds-dnumrisk', gradient_Ns = [1,2,3], task = 'magjudge',space='fsaverage5' ): # ses=1
    target_dir = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub}') # , f'ses-{ses}'

    for n_grad in gradient_Ns:
        grad = np.load(op.join(target_dir, f'grad{n_grad}_space-{space}{specification}.npy'))
        grad = np.split(grad,2) # for i, hemi in enumerate(['L', 'R']): --> left first

        for h, hemi in enumerate(['L', 'R']):    

            gii_im_datar = nib.gifti.gifti.GiftiDataArray(data=grad[h].astype(np.float32)) #
            gii_im = nib.gifti.gifti.GiftiImage(darrays= [gii_im_datar])

            out_file = op.join(target_dir, f'sub-{sub}_task-{task}_space-{space}_hemi-{hemi}_grad{n_grad}{specification}.surf.gii') # _ses-{ses}
            gii_im.to_filename(out_file) # https://nipy.org/nibabel/reference/nibabel.spatialimages.html
            print(f'saved to {out_file}')


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


def plot_GM12_from_sum_npfile_old(file = 'gm_av50_unfiltered_aligned-marg.npy',bids_folder='/Volumes/mrenkeED/data/ds-dnumrisk',grad_folder = 'derivatives/gradients', colorbar=False):

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

def plot_GM12_from_sum_npfile(n_comp, file = '',bids_folder='',grad_folder = 'derivatives/gradients', 
                              cmap='viridis',colorbar=False,):

    fsaverage = fetch_surf_fsaverage()

    grad = np.load(op.join(bids_folder,grad_folder,file))
    
    figure, axes = plt.subplots(nrows=1, ncols=n_comp, subplot_kw=dict(projection='3d'))

    for i in range(0,n_comp):
        gm = np.split(grad[i],2) # for i, hemi in enumerate(['L', 'R']): --> left first
        gm_r = gm[1] # 0 = left, 1 = right

        nplt.plot_surf_stat_map(surf_mesh=fsaverage.infl_right, colorbar = colorbar,stat_map=gm_r,cmap=cmap,view='medial',axes=axes[i])
        axes[i].set(title=f'grad {i+1}')
