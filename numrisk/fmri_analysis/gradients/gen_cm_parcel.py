import argparse
from nilearn.connectome import ConnectivityMeasure
from brainspace.utils.parcellation import reduce_by_labels
import numpy as np
import os.path as op
from utils import cleanTS, get_glasser_parcels

ses=1
n_components = 10
space = 'fsaverage'

def main(subject,bids_folder): #specification, 
    target_folder_cm = op.join(bids_folder,'derivatives','correlation_matrices')
    sub = f'{int(subject):02d}'

    # Get Glasser parcellation and mask
    mask, labeling = get_glasser_parcels()

    clean_ts = cleanTS(sub, ses,bids_folder=bids_folder, space=space) #does fsavTofsav5 if fsav5.gii does not exist
    seed_ts = reduce_by_labels(clean_ts[mask], labeling[mask], axis=1, red_op='mean',dtype=float)
    
    # CM
    correlation_measure = ConnectivityMeasure(kind='correlation')
    cm = correlation_measure.fit_transform([seed_ts.T])[0]
    np.save(op.join(target_folder_cm,f'sub-{sub}_glasserParcel-{space}.npy'),cm) # 
    print(f'sub-{sub}: connectivity matrix estimated')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder', default='/mnt_03/ds-dnumrisk')

    cmd_args = parser.parse_args()

main(cmd_args.subject, cmd_args.bids_folder)