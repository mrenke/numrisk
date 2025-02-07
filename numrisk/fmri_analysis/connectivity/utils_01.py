from nilearn import image
import numpy as np
import os.path as op
import nibabel as nib
import os
from nilearn import signal
import pandas as pd


def get_cleanTS_run(sub, ses =1, run = 1,space = 'fsaverage5', bids_folder='/Users/mrenke/data/ds-dnumrisk', task = 'magjudge'):
    # load in data as timeseries and regress out confounds (for each run sepeprately)
    fmriprep_confounds_include = ['global_signal', 'dvars', 'framewise_displacement', 'trans_x',
                                    'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                    'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'cosine00', 'cosine01', 'cosine02'
                                    ] # 
    timeseries = [None] * 2
    for i, hemi in enumerate(['L', 'R']):
        fn = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func', 
        f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-{space}_hemi-{hemi}_bold.func.gii')        
        timeseries[i] = nib.load(fn).agg_data()
    timeseries = np.vstack(timeseries) # (20484, 135)

    fmriprep_confounds_file = op.join(bids_folder,'derivatives', 'fmriprep', f'sub-{sub}', f'ses-{ses}', 'func', f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-confounds_timeseries.tsv')
    fmriprep_confounds = pd.read_table(fmriprep_confounds_file)[fmriprep_confounds_include] 
    fmriprep_confounds= fmriprep_confounds.bfill()
    clean_ts = signal.clean(timeseries.T, confounds=fmriprep_confounds).T

    return clean_ts


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

    onsets = events[['onset', 'duration', 'trial_type', 'modulation']].copy() #  
    onsets['onset'] = ((onsets['onset']+tr/2.) // 2.3) * 2.3

    frametimes = np.linspace(tr/2., (n - .5)*tr, n)

    # Suppress specific warning messages did not work so far 
    from nilearn.glm.first_level import make_first_level_design_matrix
    dm = make_first_level_design_matrix(frametimes, onsets.dropna(), 
                                            hrf_model='spm + derivative + dispersion', 
                                            oversampling=100.,drift_order=1, 
                                            drift_model=None).drop('constant', axis=1)
                                  
    dm /= dm.max()
    #print('Design matrix created to remove task effects. shape:')
    #print(dm.shape)
    return dm


def get_NPC_mask(bids_folder, space='fsaverage5', hemi='both'):
    surf_mask_L = op.join(bids_folder, 'derivatives/surface_masks', f'desc-NPC_L_space-{space}_hemi-lh.label.gii')
    surf_mask_L = nib.load(surf_mask_L).agg_data()
    surf_mask_R = op.join(bids_folder, 'derivatives/surface_masks', f'desc-NPC_R_space-{space}_hemi-rh.label.gii')
    surf_mask_R = nib.load(surf_mask_R).agg_data()
    if hemi == 'both':
        nprf_r2 = np.concatenate((surf_mask_L, surf_mask_R))
    elif hemi == 'L':
        nprf_r2 = np.concatenate((surf_mask_L, np.zeros(len(surf_mask_R))))
    elif hemi == 'R':   
        nprf_r2 = np.concatenate((np.zeros(len(surf_mask_L)), surf_mask_R))
        
    return np.bool_(nprf_r2)

