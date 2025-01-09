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

cor_type = 'cm-subset' #'covariance'
sparsity = 0.9
sparsity_name = str(sparsity).replace('.', '')
n_components = 6

plot = False
view = 'dorsal'

print(f'CM type: {cor_type},  sparsity:{sparsity}, plot:{plot}')
ref_grad = np.load(op.join(bids_folder, f'derivatives/gradients/gradients_av-NPCmask-{cor_type}_sparsity-{sparsity_name}_group-All.npy'))

for sub in subList:
    target_dir = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub}')
    #ex_file = op.join(target_dir,f'sub-{sub}_g-aligned_onlyNPC-{cor_type}.npy')
    #if  (os.path.exists(ex_file) == False):
    cm_file = op.join(bids_folder, 'derivatives', 'correlation_matrices', f'sub-{sub}_unfiltered.npy')
    cm = np.load(cm_file)
    if cor_type == 'covariance':
        cm_NPC_ = cm[np.bool_(nprf_r2_mask),:]
        cm_NPC = cm_NPC_.dot(cm_NPC_.T) # covariance matrix
    elif cor_type == 'cm-subset':
        cm_NPC = cm[np.bool_(nprf_r2_mask),:][:,np.bool_(nprf_r2_mask)]

    gm = GradientMaps(n_components=n_components,alignment='procrustes') # defaults: approacch = 'dm', kernel = None
    gm.fit(cm_NPC,reference=ref_grad)
    print(f'finished sub-{sub}: gradients generated')

    #sub = '%02d' % int(sub)
    target_dir = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub}')
    np.save(op.join(target_dir,f'sub-{sub}_g-aligned_onlyNPC-{cor_type}_sparsity-{sparsity_name}.npy'), gm.aligned_)
    np.save(op.join(target_dir,f'sub-{sub}_gradients_onlyNPC-{cor_type}_sparsity-{sparsity_name}.npy'), gm.gradients_)
    np.save(op.join(target_dir,f'sub-{sub}_lambdas_onlyNPC-{cor_type}_sparsity-{sparsity_name}.npy'), gm.lambdas_)

    if plot:
        figure, axes = plt.subplots(nrows=2, ncols=n_components,figsize = (20,8), subplot_kw=dict(projection='3d'))
        for n_grad in range(n_components):
            map = np.full(np.shape(mask), np.nan) #  np.zeros(np.shape(mask)) # 
            map[np.bool_(nprf_r2)] = gm.aligned_.T[n_grad,:]
            gms = np.split(map,2) 
            
            for n_hemi, hemi in enumerate(['L','R']):
                gm_ = gms[n_hemi]# right    
                surf_mesh = fsaverage.infl_left if hemi == 'L' else fsaverage.infl_right
                bg_map = fsaverage.sulc_left  if hemi == 'L' else fsaverage.sulc_right

                nplt.plot_surf(surf_mesh= surf_mesh, surf_map= gm_, 
                        view= view,cmap='jet', colorbar=False, title=f'grad {n_grad+1}',
                        bg_map=bg_map, bg_on_data=True,darkness=0.7, axes=axes[n_hemi, n_grad]) 
                    
        figure.suptitle(f'sub-{sub}')
        plt.savefig(op.join(plot_folder, f'sub-{sub}_NPCmask-{cor_type}_g-aligend.png'),bbox_inches='tight', dpi=300)
        plt.close('all')

