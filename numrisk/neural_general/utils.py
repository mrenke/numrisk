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
        pdf = pdf.loc[:, np.log(5):np.log(28*2)] # restrict range to actually presensted numeroisities

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


def get_decoding2D_info(subject_id, bids_folder='/data/ds-dnumrisk',key = 'decoded_pdfs.volume', mask='NPC_R', n_voxels=100): # 

    subject = f'{subject_id:02d}'
    fn = f'sub-{subject}_mask-{mask}_n_voxels-{n_voxels}_pdf.tsv'

    pdf = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func', fn)
    pdf = pd.read_csv(pdf, sep='\t', index_col=[0])
    pdf = pdf.reset_index().rename(mapper={'x':'run','Unnamed: 1':'trial_nr' },axis=1).loc[2:]
    pdf['subject'] = subject_id
    pdf['run'] = pd.to_numeric(pdf['run'], errors='coerce').astype('Int64')  # Ensure integers, handle coercion
    pdf['trial_nr'] = pd.to_numeric(pdf['trial_nr'], errors='coerce').astype('Int64')
    pdf = pdf.set_index(['subject','run','trial_nr']) #make sure all indices values are of type int!

    df = pdf.copy()
    cleaned_columns = [float(col.split('.')[0] + '.' + col.split('.')[1]) if isinstance(col, str) and '.50' not in col else
        float(col.split('.')[0] + '.' + col.split('.')[1].split('.')[0]) if isinstance(col, str) else
        col for col in df.columns ]
    df.columns = cleaned_columns
    drop_idx = next(i for i in range(1, len(cleaned_columns)) if cleaned_columns[i] < cleaned_columns[i - 1])
    df_n1 = df.iloc[:, :drop_idx]
    df_n2 = df.iloc[:, drop_idx:]
    #####
    E_n1 = pd.DataFrame((df_n1*df_n1.columns.values[np.newaxis, :] / df_n1.sum(1).values[:, np.newaxis]).sum(1))
    E_n2 = pd.DataFrame((df_n2*df_n2.columns.values[np.newaxis, :] / df_n2.sum(1).values[:, np.newaxis]).sum(1))
    #E_n1['order']= 'n1'
    E_n1 = E_n1.rename(mapper={0:'E_n1'},axis=1)#.set_index('order', append=True)
    #E_n2['order']= 'n2'
    E_n2 = E_n2.rename(mapper={0:'E_n2'},axis=1)#.set_index('order', append=True)
    Es = pd.concat((E_n1,E_n2),axis=1)#.rename(mapper={0:'E'},axis=1)

    return Es


from neuromaps import transforms
import nibabel as nib

def fsavTofsLR(source_folder, fn, hemi, source_space='fsaverage5', target_space='fsLR_den-32k'):
        source_fn =  op.join(source_folder, fn)
        target_fn = op.join(source_folder, fn.replace(source_space,target_space))
        target = transforms.fsaverage_to_fslr(source_fn, '32k',hemi=hemi)
        nib.save(target[0],target_fn) # 
        print(f'saved to {target_fn}')