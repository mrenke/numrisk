# Fit gradient map from existing CM (unfiltered), with potential for many N_components
# so script makes it run via tmux on scienccloud without need to keep local computer  on
# weird naming with sub/group variable

import argparse
from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import map_to_labels
import numpy as np
import os.path as op
from utils import get_basic_mask
import os


def main(sub='All',bids_folder='/mnt_03/ds-dnumrisk', N_components=3):
    print(sub)
    source_folder = op.join(bids_folder,'derivatives','correlation_matrices')
    target_dir = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub}')
    os.makedirs(target_dir) if not op.exists(target_dir) else None

    specification = f'_N-{N_components}'

    mask, labeling_noParcel = get_basic_mask()
    N_vertices = len(np.where(mask==True)[0])

    n_components = int(N_components)
    cm = np.load(op.join(source_folder,f'cm_av_group-{sub}.npy'))
    gm = GradientMaps(n_components=n_components)
    gm.fit(cm)

    grad = [None] * n_components

    for i, g in enumerate(gm.gradients_.T): # 
        grad[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)

    np.save(op.join(target_dir,f'sub-{sub}_gradients{specification}.npy'), grad) # for plotting
    np.save(op.join(target_dir,f'sub-{sub}_gms{specification}.npy'), gm.gradients_) # for alignment reference! 
    np.save(op.join(target_dir,f'sub-{sub}_lambdas{specification}.npy'), gm.lambdas_) 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder', default='/mnt_03/ds-dnumrisk')
    parser.add_argument('--N_components', default=3)

    cmd_args = parser.parse_args()

    main(cmd_args.subject, cmd_args.bids_folder, cmd_args.N_components)