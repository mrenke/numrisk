import numpy as np
import nibabel as nib
from nilearn import datasets
import os.path as op
import os
from brainspace.gradient import GradientMaps
from utils import get_basic_mask
from  nilearn.datasets import fetch_surf_fsaverage
import nilearn.plotting as nplt
import matplotlib.pyplot as plt

bids_folder = '/mnt_03/ds-dnumrisk' 
plot_folder = op.join(bids_folder, 'plots_and_ims/gradient_stuff')

from os import listdir
subList = [f[4:6] for f in listdir(op.join(bids_folder)) if f[0:4] == 'sub-' and len(f)==6]

fsaverage = fetch_surf_fsaverage('fsaverage5') 

surf_mask_L = op.join(bids_folder, 'derivatives/surface_masks', 'desc-NPC_L_space-fsaverage5_hemi-lh.label.gii')
surf_mask_L = nib.load(surf_mask_L).agg_data()
surf_mask_R = op.join(bids_folder, 'derivatives/surface_masks', 'desc-NPC_R_space-fsaverage5_hemi-rh.label.gii')
surf_mask_R = nib.load(surf_mask_R).agg_data()
nprf_r2 = np.concatenate((surf_mask_L, surf_mask_R))

mask, labeling_noParcel = get_basic_mask()
nprf_r2_mask = nprf_r2[mask]

view = 'dorsal'

for sub in subList:
    cm_file = op.join(bids_folder, 'derivatives', 'correlation_matrices', f'sub-{sub}_unfiltered.npy')
    cm = np.load(cm_file)
    cm_NPC = cm[np.bool_(nprf_r2_mask),:]
    cm_NPC_cov = cm_NPC.dot(cm_NPC.T) # covariance matrix

    gm = GradientMaps(n_components=3) # defaults: approacch = 'dm', kernel = None
    gm.fit(cm_NPC_cov)
    print(f'finished sub-{sub}: gradients generated')
    
    figure, axes = plt.subplots(nrows=2, ncols=3,figsize = (15,8), subplot_kw=dict(projection='3d'))
    for n_grad in range(3):
        map = np.full(np.shape(mask), np.nan) #  np.zeros(np.shape(mask)) # 
        map[np.bool_(nprf_r2)] = gm.gradients_.T[n_grad,:]
        gms = np.split(map,2) 
        
        for n_hemi, hemi in enumerate(['L','R']):
            gm_ = gms[n_hemi]# right    
            surf_mesh = fsaverage.infl_left if hemi == 'L' else fsaverage.infl_right
            bg_map = fsaverage.sulc_left  if hemi == 'L' else fsaverage.sulc_right

            nplt.plot_surf(surf_mesh= surf_mesh, surf_map= gm_, 
                    view= view,cmap='jet', colorbar=False, title=f'grad {n_grad+1}',
                    bg_map=bg_map, bg_on_data=True,darkness=0.7, axes=axes[n_hemi, n_grad]) 
                
    figure.suptitle(f'sub-{sub}')
    plt.savefig(op.join(plot_folder, f'sub-{sub}_NPCmask-gradient.png'),bbox_inches='tight', dpi=300)


    