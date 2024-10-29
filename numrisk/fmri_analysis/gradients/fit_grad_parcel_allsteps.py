## not finished, first derive average gradient to align to, so first get all subs CMs 


import argparse
from nilearn.connectome import ConnectivityMeasure
from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import reduce_by_labels, map_to_labels
import numpy as np
import nibabel as nib
from nilearn import datasets
import os.path as op
import os
from nilearn import signal
import pandas as pd
from scipy.sparse.csgraph import connected_components
from utils import cleanTS, get_glasser_parcels
#from utils import fsavTofsav5,cleanTS, saveGradToNPFile, npFileTofs5Gii,fsav5Tofsnative

ses=1
n_components = 10
space = 'fsaverage'

def main(sub,bids_folder, kernel, approach): #specification, 
    target_folder_cm = op.join(bids_folder,'derivatives','correlation_matrices')
    target_folder_gm = op.join(bids_folder,'derivatives','gradients')

    # Get Glasser parcellation and mask
    mask, labeling = get_glasser_parcels()

    clean_ts = cleanTS(sub, ses,bids_folder=bids_folder, space=space) #does fsavTofsav5 if fsav5.gii does not exist
    seed_ts = reduce_by_labels(clean_ts[mask], labeling[mask], axis=1, red_op='mean',dtype=float)
    
    # CM
    correlation_measure = ConnectivityMeasure(kind='correlation')
    cm = correlation_measure.fit_transform([seed_ts.T])[0]
    np.save(op.join(target_folder_cm,f'sub-{sub}_ses-{ses}_glasserParcel-{space}.npy'),cm) # 
    print('raw connectivity matrix estimated')

    gm = GradientMaps(n_components=n_components, kernel = kernel, approach=approach,random_state=0)
    gm.fit(cm)

    grad = [None] * n_components
    for i, g in enumerate(gm.gradients_.T):
        grad[i] = map_to_labels(g, labeling, mask=mask, fill=np.nan)
    print('gradients generated')

    np.save(op.join(target_folder_gm,f'sub-{sub}_glasserParcel_gradients.npy'),grad) # -{space}_kernel-{kernel}_approach-{approach}
    np.save(op.join(target_folder_gm,f'sub-{sub}_glasserParcel_lambdas.npy'), gm.lambdas_) 
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