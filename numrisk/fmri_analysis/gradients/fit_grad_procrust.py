# all steps of the gradient generation process combined,
# hence: needs freesurfer directory for 1. fsavTofsav5 (laoding in), then fsav5Tofsnative (save)
# OQ: 
# - save the whole gradient object? (lambdas - explained variance, etc.)
# - save filtered masks ?

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
from utils import get_basic_mask, cleanTS #saveGradToNPFile, npFileTofs5Gii,fsav5Tofsnative

def main(sub,ses,bids_folder,remove_task_effects, specification,save_cm_unfiltered, n_components=3):

    sub = '%02d' % int(sub)

    align_spec = '_align-procrustes'
    remove_task_effects_spec = '_removed-taskeffect' if remove_task_effects else ''
    print(remove_task_effects_spec)

    specification = remove_task_effects_spec + align_spec + specification

    mask, labeling_noParcel = get_basic_mask()

    clean_ts = cleanTS(sub, ses,remove_task_effects, bids_folder=bids_folder) # checks if fsav5-file exists, if not, creates it
    seed_ts = clean_ts[mask]
    correlation_measure = ConnectivityMeasure(kind='correlation')
    cm = correlation_measure.fit_transform([seed_ts.T])[0] #correlation_matrix_noParcel
    print(f'sub-{sub}: raw connectivity matrix estimated')    
    if save_cm_unfiltered:
        np.save(op.join(bids_folder, 'derivatives', 'correlation_matrices', f'sub-{sub}_unfiltered{remove_task_effects_spec}.npy'), cm)

    # filter out nodes that are not connected to the rest
    cc = connected_components(cm)
    mask_cc = cc[1] == 0 # all nodes in 0 belong to the largest connected component, check #-components in cc[0]
    mask[mask == True] = mask_cc # mark nodes not in component 0  as False in mask
    cm_filtered = cm[mask_cc, :][:, mask_cc]
    print('connected components derived')    

    # load in reference gradient and apply same filter
    g_ref = np.load(op.join(bids_folder,'derivatives', 'gradients','refGrad-av_task-risk_align-marg.npy')) # same labeling_noParcel as cm_unfiltered
    g_ref = g_ref[mask_cc]

    # now perform embedding on cleaned data + alignment
    g_align_fil = GradientMaps(n_components=n_components,alignment='procrustes') # defaults: approacch = 'dm', kernel = None
    g_align_fil.fit(cm_filtered,reference=g_ref)
    print(f'finished sub-{sub}: gradients generated')

    gm = g_align_fil.aligned_.T # !!!! take .algined_ and not .gradients (which also exists, but those are not aligend)

    grad = [None] * n_components
    for i, g in enumerate(gm): # gm.gradients_.T
        grad[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)
    
    target_dir = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub}')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    np.save(op.join(target_dir,f'sub-{sub}_gradients{specification}.npy'), grad) # save all together

    # saveGradToNPFile(grad, sub,ses, specification,bids_folder=bids_folder) # saves each gradient seperately
    #npFileTofs5Gii(sub,ses, specification,bids_folder=bids_folder)
    #fsav5Tofsnative(sub,ses,specification,bids_folder=bids_folder)   

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--session', default=1, type=int)  
    parser.add_argument('--bids_folder', default='/Volumes/mrenkeED/data/ds-dnumrisk')
    parser.add_argument('--specification', default='')
    parser.add_argument('--save_cm_unfiltered', action='store_true' ) #default=False, type=bool)
    parser.add_argument('--remove_task_effects', action='store_true' ) # default=False, type=bool)

    cmd_args = parser.parse_args()

    main(cmd_args.subject, cmd_args.session, cmd_args.bids_folder, 
        remove_task_effects = cmd_args.remove_task_effects,
         specification = cmd_args.specification, 
         save_cm_unfiltered = cmd_args.save_cm_unfiltered)
