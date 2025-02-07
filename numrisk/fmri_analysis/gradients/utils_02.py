from nilearn import image
import numpy as np
import os.path as op
import nibabel as nib
import os
from nilearn import signal
import pandas as pd
from numrisk.fmri_analysis.gradients.utils import get_basic_mask
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

def get_pval_colormap():
    skewed = True
    first = int((128*2)-np.round(255*(1.-0.90)))
    second = (256-first)
    first = first if skewed else second
    colors2 = plt.cm.cool(np.linspace(0.1, .98, first))
    colors3 = plt.cm.spring(np.linspace(0.25, 1, second))

    # combine them and build a new colormap
    cols = np.vstack((colors2,colors3))
    mymap = colors.LinearSegmentedColormap.from_list('my_colormap', cols[::-1])
    return mymap

def get_all_behavior(bids_folder):
    source_folder = op.join(bids_folder, 'derivatives/phenotype')
    magjudge_bauer_params = pd.read_csv(op.join(source_folder,f'magjudge_bauer-3_sds.csv')).set_index('subject')
    magjudge_probit_params = pd.read_csv(op.join(source_folder,'probit-2_all-subwise-params_appropSample.csv')).set_index('subject')
    magjudge_bauer_params_unbiased = pd.read_csv(op.join(source_folder,'bauer-3_sds-maps_unbiased.csv')).set_index('subject')
    magjudge_bauer_params_unbiased = magjudge_bauer_params_unbiased.rename(mapper={'memory_noise_sd':'memory_noise_sd_unbiased', 
                                                        'perceptual_noise_sd':'perceptual_noise_sd_unbiased'}, axis=1)
    vs_wm = pd.read_csv(op.join(source_folder, 'visio-spatial-WM_CBTtask-params.csv')).set_index('subject')
    pana = pd.read_csv(op.join(source_folder, 'ANSacuity_panamath.csv')).set_index('subject')
    pana['weber_frac_log'] = np.log(pana['weber_frac'])
    math_stuff = pd.read_csv(op.join(source_folder, 'math_skill&confidence&anxiety-means.csv')).set_index('subject')['skill_score']
    iq_scores = pd.read_csv(op.join(source_folder, 'iq-scores_ids2.csv')).set_index('subject').rename(mapper={'me':'visio-spatial IQ','kn':'verbal IQ'},axis=1)

    df_behave = magjudge_probit_params.join(pana).join(magjudge_bauer_params).join(vs_wm)
    df_behave = df_behave.join(magjudge_bauer_params_unbiased.drop(columns='group')).join(math_stuff).join(iq_scores)

    df_behave = df_behave.set_index('group', append=True)
    return df_behave

