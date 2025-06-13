import os.path as op
import numpy as np
import pandas as pd

def get_all_behavior(bids_folder):
    source_folder = op.join(bids_folder, 'derivatives/phenotype')
    magjudge_bauer_params = pd.read_csv(op.join(source_folder,f'magjudge_bauer-3_sds.csv')).set_index('subject')
    magjudge_probit_params = pd.read_csv(op.join(source_folder,'probit-2_all-subwise-params_appropSample.csv')).set_index('subject')
    magjudge_bauer_params_unbiased = pd.read_csv(op.join(source_folder,'bauer-3_sds-maps_unbiased.csv')).set_index('subject')
    magjudge_bauer_params_unbiased = magjudge_bauer_params_unbiased.rename(mapper={'memory_noise_sd':'memory_noise_sd_unbiased', 
                                                        'perceptual_noise_sd':'perceptual_noise_sd_unbiased'}, axis=1)
    vs_wm = pd.read_csv(op.join(source_folder, 'visio-spatial-WM_CBTtask-params.csv')).set_index('subject')
    pana = pd.read_csv(op.join(source_folder, 'ANSacuity_panamath.csv')).set_index('subject')
    pana['weber_frac_log'] = np.log(pana['weber_frac'])
    math_stuff = pd.read_csv(op.join(source_folder, 'math_skill&confidence&anxiety-means.csv')).set_index('subject')['skill_score']
    iq_scores = pd.read_csv(op.join(source_folder, 'iq-scores_ids2.csv')).set_index('subject').rename(mapper={'me':'visio-spatial IQ','kn':'verbal IQ'},axis=1)

    df_behave = magjudge_probit_params.join(pana).join(magjudge_bauer_params).join(vs_wm)
    df_behave = df_behave.join(magjudge_bauer_params_unbiased.drop(columns='group')).join(math_stuff).join(iq_scores)

    df_behave = df_behave.set_index('group', append=True)
    return df_behave