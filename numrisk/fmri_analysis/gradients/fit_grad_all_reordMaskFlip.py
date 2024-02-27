# all steps of the gradient generation process combined,
# hence: needs freesurfer directory for 1. fsavTofsav5 (laoding in), then fsav5Tofsnative (save)
# 

import argparse
from nilearn.connectome import ConnectivityMeasure
from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import map_to_labels
import numpy as np
import nibabel as nib
from nilearn import datasets
import os.path as op
import os
from nilearn import signal
import pandas as pd
from scipy.sparse.csgraph import connected_components
from utils import get_basic_mask,cleanTS, saveGradToNPFile, npFileTofs5Gii #,fsav5Tofsnative
from utils_02 import align_gradients_ROIdependant, get_reference_gradient

def main(sub,ses,bids_folder,specification='_reordMaskFlip', n_components=3,save_cm=True):

    mask, labeling_noParcel = get_basic_mask()

    clean_ts = cleanTS(sub, ses,bids_folder=bids_folder) # checks if fsav5-file exists, if not, creates it
    seed_ts = clean_ts[mask]
    correlation_measure = ConnectivityMeasure(kind='correlation')

    # filter out nodes that are not connected to the rest
    graph = correlation_measure.fit_transform([seed_ts.T])[0] #correlation_matrix_noParcel
    if save_cm:
        target_folder = op.join(bids_folder,'derivatives','correlation_matrices') 
        np.save(op.join(target_folder,f'sub-{sub}_ses-{ses}_corrMatrix_fsav5_unfiltered.npy'),graph)

    cc = connected_components(graph)
    mask_cc = cc[1] == 0 # all nodes in 0 belong to the largest connected component, check #-components in cc[0]
    mask[mask == True] = mask_cc # mark nodes not in component 0  as False in mask
    print('mask with connected components created')

    seed_ts = clean_ts[mask]

    #now perform embedding on cleaned data
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([seed_ts.T])[0]
    print('filtered connectivity matrix estimated')

    gm = GradientMaps(n_components=n_components, random_state=0)
    gm.fit(correlation_matrix)

    grad = [None] * n_components
    for i, g in enumerate(gm.gradients_.T):
        grad[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)
    
    print('gradients generated')

    if specification == '_reordMaskFlip':
        reference = get_reference_gradient(bids_folder=bids_folder)
        gradients = align_gradients_ROIdependant(grad, reference)
        print('aligned gradients (based on ROI masks and max diff between 1) high/low and 2) visual/motor ROIs')

    saveGradToNPFile(gradients, sub,ses, specification,bids_folder=bids_folder)
    npFileTofs5Gii(sub,ses, specification,bids_folder=bids_folder)
    #fsav5Tofsnative(sub,ses,specification,bids_folder=bids_folder)   


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--session', default=1, type=int)  
    parser.add_argument('--bids_folder', default='/Volumes/mrenkeED/data/ds-dnumrisk')
    parser.add_argument('--specification', default='_reordMaskFlip')
    cmd_args = parser.parse_args()

    main(cmd_args.subject, cmd_args.session, cmd_args.bids_folder, cmd_args.specification)