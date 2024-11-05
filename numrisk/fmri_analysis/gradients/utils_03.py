from nilearn import image
import numpy as np
import os.path as op
import nibabel as nib
import os
import pandas as pd
from neuromaps import transforms



def npFileTofsLRGii(sub, specification='',bids_folder='/Users/mrenke/data/ds-dnumrisk', gradient_Ns = [1,2,3], task = 'magjudge',
                    source_space='fsaverage5', target_space='fsLR_den-32k' ): # ses=1
    target_dir = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub}') # , f'ses-{ses}'

    for n_grad in gradient_Ns:
        grad = np.load(op.join(target_dir, f'grad{n_grad}_space-{source_space}{specification}.npy'))
        grad = np.split(grad,2) # for i, hemi in enumerate(['L', 'R']): --> left first

        for h, hemi in enumerate(['L', 'R']):    
            gii_im_datar = nib.gifti.gifti.GiftiDataArray(data=grad[h].astype(np.float32)) #
            gii_im_fsav = nib.gifti.gifti.GiftiImage(darrays= [gii_im_datar])

            gii_im_fslr = transforms.fsaverage_to_fslr(gii_im_fsav, '32k',hemi=hemi)
            out_file = op.join(target_dir, f'sub-{sub}_task-{task}_space-{target_space}_hemi-{hemi}_grad{n_grad}{specification}.surf.gii') # _ses-{ses}
            nib.save(gii_im_fslr[0],out_file)
            print(f'saved to {out_file}')