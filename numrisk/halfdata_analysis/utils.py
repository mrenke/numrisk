
import pandas as pd
import os.path as op
import numpy as np
from tqdm.contrib.itertools import product
import matplotlib.pyplot as plt
#import pingouin
import seaborn as sns
import scipy.stats as ss
import numpy as np

def invprobit(x):
    return ss.norm.ppf(x)

def extract_intercept_gamma(trace, model, data, group=False):

    fake_data = get_fake_data(data, group)

    pred = model.predict(trace, 'mean', fake_data, inplace=False, include_group_specific=not group)['posterior']['choice_mean']

    pred = pred.to_dataframe().unstack([0, 1])
    pred = pred.set_index(pd.MultiIndex.from_frame(fake_data))

    #print(pred)
    # return pred

    pred0 = pred.xs(0, 0, 'x')

    #print(pred0)
    intercept = pd.DataFrame(invprobit(pred0), index=pred0.index, columns=pred0.columns)
    gamma = invprobit(pred.xs(1, 0, 'x')) - intercept

    intercept = pd.concat((intercept.droplevel(0, 1),), keys=['intercept'], axis=1)
    gamma = pd.concat((gamma.droplevel(0, 1),), keys=['gamma'], axis=1)

    return intercept, gamma


def get_fake_data(data, group=False):

    data = data.reset_index()

    if group:
        permutations = [[1]]
    else:
        permutations = [data['subject'].unique()]

    permutations += [np.array([0., 1.])]
    names=['subject', 'x']


    fake_data = pd.MultiIndex.from_product(permutations, names=names).to_frame().reset_index(drop=True)

    return fake_data


def get_decoding_info(subject, session=1,  bids_folder='/data/ds-stressrisk', mask='NPC_R', n_voxels='select',
                      split_data = 'full'):

    subject = f'{subject:02d}' 
    split_data_fold_name = f'_{split_data}' if split_data != 'full' else ''

    if n_voxels =='select':
        key = f'decoded_pdfs.volume.cv_vselect{split_data_fold_name}.denoise'
        pdf = op.join(bids_folder, 'derivatives', key, f'sub-{subject}', 'func', f'sub-{subject}_ses-{session}_mask-{mask}_space-T1w_pars.tsv')
    else:
        pdf = op.join(bids_folder, 'derivatives', f'{key}.{n_voxels}voxels', f'sub-{subject}', 'func', f'sub-{subject}_ses-{session}_mask-{mask}_nvoxels-{n_voxels}_space-T1w_pars.tsv')

    if op.exists(pdf):
        pdf = pd.read_csv(pdf, sep='\t', index_col=[0])
        pdf.columns = pdf.columns.astype(float)
        pdf = pdf.loc[:, np.log(5):np.log(28*4)] # restrict range to actually presensted numeroisities

        E = (pdf*pdf.columns.values[np.newaxis, :] / pdf.sum(1).values[:, np.newaxis]).sum(1)

        E = pd.concat((E,), keys=[(int(subject), split_data)], #, int(session), mask, n_voxels)],
        names=['subject', 'data']).to_frame('E') #, 'session', 'mask', 'n_voxels']

        pdf /= np.trapz(pdf, pdf.columns.astype(float))[:, np.newaxis] #normalizing

        E['sd'] = np.trapz(np.abs(E.values - pdf.columns.astype(float).values[np.newaxis, :]) * pdf, pdf.columns, axis=1)
        #print('succesfully predicted')

        return E
    else:
        #print(pdf)
        return pd.DataFrame(np.zeros((0, 0)))
    

# from gilles, not needed so far (is for regresssion model)
def get_pars(idata, group=True):
    traces = {}

    pars_vertex = []
    pars_ips = []

    keys = ['n1_evidence_sd', 'n2_evidence_sd', 'n2_prior_mu', 'n2_prior_std']

    for key in keys:

        if group:
            key_ = key+'_mu'
        else:
            key_ = key
        
        if key_ in idata.posterior.keys():
            print(key_)

            traces[key] = idata.posterior[f'{key_}'].to_dataframe()

            ips_values = traces[key].xs('Intercept', 0, f'{key}_regressors')


            if 'stimulation_condition[T.vertex]' in traces[key].index.get_level_values(f'{key}_regressors'):
                vertex_values = ips_values + traces[key].xs('stimulation_condition[T.vertex]', 0, f'{key}_regressors')
            else:
                vertex_values = ips_values

            if (key in ['n1_evidence_sd', 'n2_evidence_sd']) and 'evidence_sd' in idata.posterior.keys():
                if group:
                    key__ = 'evidence_sd_mu'
                else:
                    key__ = 'evidence_sd'
                
                print('yo')
                ips_values += idata.posterior[key__].to_dataframe().xs('stimulation_condition[ips]', 0, f'evidence_sd_regressors').values
                vertex_values += idata.posterior[key__].to_dataframe().xs('stimulation_condition[vertex]', 0, f'evidence_sd_regressors').values


            if key in ['perceptual_noise_sd', 'memory_noise_sd', 'n1_evidence_sd', 'n2_evidence_sd', 'risky_prior_std', 'safe_prior_std']:
                ips_values = softplus_np(ips_values)
                vertex_values = softplus_np(vertex_values)

            pars_ips.append(ips_values)
            pars_vertex.append(vertex_values)


    pars_ips = pd.concat(pars_ips, axis=1)
    pars_vertex = pd.concat(pars_vertex, axis=1)

    pars = pd.concat((pars_ips, pars_vertex), keys=['IPS', 'Vertex'], names=['stimulation condition'])
    pars.columns.name = 'parameter'
    pars = pars.stack().to_frame('value')

    return pars
 