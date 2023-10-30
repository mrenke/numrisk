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
from utils import cleanTS, get_basic_mask
#from utils import fsavTofsav5,cleanTS, saveGradToNPFile, npFileTofs5Gii,fsav5Tofsnative
ses=1

def main(sub,bids_folder, kernel, approach): #specification, 

    # Build Destrieux parcellation and mask
    mask, labeling_noParcel = get_basic_mask()

    # subject
    clean_ts = cleanTS(sub, ses,bids_folder=bids_folder) #does fsavTofsav5 if fsav5.gii does not exist
    seed_ts = clean_ts[mask]
    correlation_measure = ConnectivityMeasure(kind='correlation')
    print('raw connectivity matrix estimated')

     # filter out nodes that are not connected to the rest
    graph = correlation_measure.fit_transform([seed_ts.T])[0] #correlation_matrix
    cc = connected_components(graph)
    mask_cc = cc[1] == 0 # all nodes in 0 belong to the largest connected component, check #-components in cc[0]
    mask[mask == True] = mask_cc # mark nodes not in component 0  as False in mask

    print('mask with connected components created')

    seed_ts = clean_ts[mask]

    #now perform embedding on cleaned data
    correlation_measure = ConnectivityMeasure(kind='correlation')
    cm = correlation_measure.fit_transform([seed_ts.T])[0]
    target_folder = op.join(bids_folder,'derivatives','correlation_matrices') # save cm and mask for average_gm construction later
    np.save(op.join(target_folder,f'sub-{sub}_ses-{ses}_filtered_space-fsav5_kernel-{kernel}_approach-{approach}.npy'),np.array([cm, mask])) # 
    gm = GradientMaps(n_components=3, kernel = kernel, approach=approach,random_state=0)
    gm.fit(cm)

    grad = [None] * 2
    for i, g in enumerate(gm.gradients_.T):
        grad[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)
    
    print('gradients generated')

    target_folder = op.join(bids_folder,'derivatives','gradients')
    np.save(op.join(target_folder,f'sub-{sub}_ses-{ses}_filtered_space-fsav5_kernel-{kernel}_approach-{approach}.npy'),grad) # 

    #saveGradToNPFile(grad_, sub,ses, specification,bids_folder=bids_folder)
    #npFileTofs5Gii(sub,ses, specification,bids_folder=bids_folder)
    #fsav5Tofsnative(sub,ses,specification,bids_folder=bids_folder)   

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder', default='/Volumes/mrenkeED/data/ds-dnumrisk')
    parser.add_argument('--kernel', default=None) # 'normalized_angle',  #Kernel function. If None, only sparsify. Default is None.
    parser.add_argument('--approach', default='dm')# Embedding approach. Default is 'dm'

    cmd_args = parser.parse_args()

    main(cmd_args.subject, cmd_args.bids_folder, 
          cmd_args.kernel, cmd_args.approach,
          )