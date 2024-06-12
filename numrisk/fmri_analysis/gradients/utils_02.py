from nilearn import image
import numpy as np
import os.path as op
import nibabel as nib
import os
from nilearn import signal
import pandas as pd
from utils import get_basic_mask
from nilearn import datasets
from brainspace.utils.parcellation import map_to_labels

import matplotlib.colors as colors
import matplotlib.pyplot as plt


def align_gradients_ROIdependant(gradients, reference): # this version does not bother about nan values
    gradients = np.array(gradients)
    reference_grad = np.array(reference)

    # get masks
    atlas = datasets.fetch_atlas_surf_destrieux()
    labeling = np.concatenate([atlas['map_left'], atlas['map_right']]) # table with regions and names: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2937159/

    motor_mask = np.isin(labeling, atlas.labels.index(b'G_precentral')) 
    occ_mask = np.isin(labeling, atlas.labels.index(b'Pole_occipital'))
    atlats_index_highOrder = [atlas.labels.index(b'S_temporal_sup'), atlas.labels.index(b'S_interm_prim-Jensen')] # temporal, TPJ
    atlats_index_lowOrder = [atlas.labels.index(b'G_precentral'), atlas.labels.index(b'Pole_occipital' )] # motor, visual
    highOrderRegions_mask = np.isin(labeling, atlats_index_highOrder)
    lowOrderRegions_mask = np.isin(labeling, atlats_index_lowOrder)

    difs = np.zeros(shape = (3,2))
    for grad_n in range(3):
        dif_forgrad1 = gradients[grad_n][highOrderRegions_mask].mean() - gradients[grad_n][lowOrderRegions_mask].mean()
        dif_forgrad2 = gradients[grad_n][motor_mask].mean() - gradients[grad_n][occ_mask].mean()
        difs[grad_n,:] = [np.abs(dif_forgrad1), np.abs(dif_forgrad2)]
    print(difs)
    max_difs_masks_indices = np.argmax(np.abs(difs), axis=0)
    print(max_difs_masks_indices) # should work from inspecting the difs * , max_difs

    # assign the missing gradient   
    all_numbers = set(range(3))
    missing_number = all_numbers - set(max_difs_masks_indices)
    max_difs_masks_indices = np.append(max_difs_masks_indices, missing_number.pop())
    print(max_difs_masks_indices)  

    max_difs_masks_indices = np.array(max_difs_masks_indices, dtype=int).flatten()
    # Reorder gradients based on the index of the reference gradient with the highest absolute dot product
    gradients = gradients[np.argsort(max_difs_masks_indices), :]
    dot_products = np.array([[np.nansum(g * r) for r in reference] for g in gradients]) # kind of np.dot, but ignoring nans
    
    print(dot_products)
    # Flip gradients where the dot product with the corresponding reference gradient is negative
    for i in range(np.shape(gradients)[0]):
        if dot_products[i,i] < 0:
            gradients[i] *= -1

    return gradients


def get_reference_gradient(file = 'gm_av50_unfiltered_aligned-marg_stressrisk.npy',
                           bids_folder='/Volumes/mrenkeED/data/ds-numrisk',
                           grad_folder = 'derivatives/gradients'):
    
    mask, labeling_noParcel = get_basic_mask()

    # reference 
    gm = np.load(op.join(bids_folder,grad_folder,file))
    grad = [None] * np.shape(gm)[1]
    for i, g in enumerate(gm.T):
        grad[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)

    reference_grad = grad
    return reference_grad

def get_GMmargulies_cmap(skewed=True): 
    # proportion of the two colormaps, defines how much space is taken by each
    first = int((128*2)-np.round(255*(1.-0.90)))
    second = (256-first)
    first = first if skewed else second
    colors2 = plt.cm.viridis(np.linspace(0.1, .98, first))
    colors3 = plt.cm.YlOrBr(np.linspace(0.25, 1, second))

    # combine them and build a new colormap
    cols = np.vstack((colors2,colors3))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', cols)
    return mymap
# from : /Users/mrenke/git/gradient_analysis/03_visualize_embeddings.ipynb

def get_events_confounds(sub, ses, run, bids_folder='/Users/mrenke/data/ds-dnumrisk',task='magjudge' ):
    tr = 2.3 # repetition Time
    n = 135  # number of slices
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
    stimulus2['duration'] = choices.set_index('trial_nr')['onset']- stimulus2.set_index('trial_nr')['onset'] + 0.6 # 0.6 + 0.6 ## looked at the data, is is different for stim 1 and 2... ?!!
    stimulus2['onset'] = stimulus2['onset'] - 0.6
    stimulus2['stim_order'] = int(2)
    stimulus2_int = stimulus2.copy()
    stimulus2_int['trial_type'] = 'stimulus2_int'
    stimulus2_int['modulation'] = 1
    stimulus2_mod= stimulus2.copy()
    stimulus2_mod['trial_type'] = 'stimulus2_mod'
    stimulus2_mod['modulation'] = stimulus2['n2']

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