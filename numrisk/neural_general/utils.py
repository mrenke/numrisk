import pandas as pd
import os.path as op
import numpy as np
#from stress_risk.utils.data import get_all_behavior
#from tqdm.contrib.itertools import product
import matplotlib.pyplot as plt
#import pingouin
#import seaborn as sns

# E['sd'] = np.trapz(np.abs(E.values - pdf.columns.astype(float).values[np.newaxis, :]) * pdf, pdf.columns, axis=1)

def get_decoding_info(subject, session=1,n_stim=1,  bids_folder='/data/ds-dnumrisk',key = 'decoded_pdfs.volume', mask='NPC_R', n_voxels='select'): # 

    subject = f'{subject:02d}'
    
    key = f'decoded_pdfs_stim{n_stim}.volume.cv_vselect.denoise'
    fn = f'sub-{subject}_ses-{session}_mask-{mask}_space-T1w_pars.tsv'

    pdf = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func', fn)

    if op.exists(pdf):
        pdf = pd.read_csv(pdf, sep='\t', index_col=[0])
        pdf.columns = pdf.columns.astype(float)
        pdf = pdf.loc[:, np.log(5):np.log(28*4)] # restrict range to actually presensted numeroisities

        E = (pdf*pdf.columns.values[np.newaxis, :] / pdf.sum(1).values[:, np.newaxis]).sum(1)

        E = pd.concat((E,), keys=[(int(subject), int(session), mask, n_voxels)],
        names=['subject', 'session', 'mask', 'n_voxels']).to_frame('E')

        pdf /= np.trapz(pdf, pdf.columns.astype(float))[:, np.newaxis] #normalizing

        E['sd'] = np.trapz(np.abs(E.values - pdf.columns.astype(float).values[np.newaxis, :]) * pdf, pdf.columns, axis=1)
        #print('succesfully predicted')

        return E
    else:
        print(pdf)
        return pd.DataFrame(np.zeros((0, 0)))

from neuromaps import transforms
import nibabel as nib

def fsavTofsLR(source_folder, fn, hemi, source_space='fsaverage5', target_space='fsLR_den-32k'):
        source_fn =  op.join(source_folder, fn)
        target_fn = op.join(source_folder, fn.replace(source_space,target_space))
        target = transforms.fsaverage_to_fslr(source_fn, '32k',hemi=hemi)
        nib.save(target[0],target_fn) # 
        print(f'saved to {target_fn}')