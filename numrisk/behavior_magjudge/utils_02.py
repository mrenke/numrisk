import os.path as op
import os
import numpy as np
import pandas as pd
from bauer.models import DDMMagnitudeComparisonModel, MagnitudeComparisonRegressionModel, FlexibleNoiseComparisonRegressionModel, DDMMagnitudeComparisonRegressionModel

#from stress_risk.utils.data import get_all_behavior
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from bauer.utils.bayes import softplus

def get_subwise_params(idata, param_name, group_list=None):
    df_param= idata.posterior[param_name].to_dataframe()    
    df_param.columns.name = 'parameter'
    df_param.index = df_param.index.set_names(['chain','draw','subject','regressor']) 
    df_param = df_param.stack().to_frame('value')

    if group_list is not None:
        df_param = df_param.join(group_list, on='subject')
        regressors = df_param.index.get_level_values('regressor').unique()
        non_intercept = [r for r in regressors if r != 'Intercept']
        if len(non_intercept) != 1:
            raise ValueError(f"Expected exactly one non-Intercept regressor for {param_name}, got {list(non_intercept)}")
        group_regressor = non_intercept[0]  # e.g. 'group' (old traces) or 'group[T.dyscalc]' (bauer v0.4.0 traces)
        df_param_int = df_param.xs('Intercept', 0,'regressor')
        df_param_group = df_param.xs(group_regressor, 0,'regressor')
        df_result = df_param_int.copy()
        df_result.loc[df_result['group'] == 1] += df_param_group.loc[df_result['group'] == 1]
        df_param = df_result
    else:
        df_param = df_param.xs('Intercept', 0,'regressor')
    df_param = df_param.groupby(['subject'])[['value']].mean()
    df_param = df_param.rename(mapper={'value':param_name},axis=1)
    
    if 'sd' in param_name:
        df_param[param_name]= softplus(df_param[param_name])
    return df_param

def get_subwise_params_cellmeans(idata, param_name, group_map):
    """For cell-means traces (random_regressors={p: '0 + C(group)'}), e.g. the
    dyscalculic_ddm choice/DDM/RDM fits in derivatives/cogmodels_magjudge.
    Each subject's posterior has one entry per group level, but only the
    level matching that subject's actual group is informed by their data
    (the other is just the population prior) - so we have to pick it out
    per subject rather than averaging both, as get_subwise_params does for
    the reference-cell (Intercept + offset) design.

    group_map: dict subject -> group label string, matching the
    'C(group)[<label>]' regressor coordinates in idata.
    """
    da = idata.posterior[param_name]
    regressor_dim = f'{param_name}_regressors'
    values = {
        sub: da.sel(subject=sub, **{regressor_dim: f'C(group)[{group_map[sub]}]'}).mean(dim=['chain', 'draw']).item()
        for sub in da['subject'].values
    }
    df_param = pd.DataFrame.from_dict(values, orient='index', columns=[param_name])
    df_param.index.name = 'subject'

    if 'sd' in param_name:
        df_param[param_name] = softplus(df_param[param_name])
    return df_param

def build_model(model_label, df):
    if model_label == '1': # same priors = probit model ?!
        model = MagnitudeComparisonRegressionModel(df, 
                                    regressors = {'n1_evidence_sd':'group', 'n2_evidence_sd':'group'},
                                    fit_prior=False,
                                    fit_seperate_evidence_sd = True, 
                                    memory_model='independent',
                                    )
    if model_label == '2':
        model = MagnitudeComparisonRegressionModel(df, 
                                    regressors = {'n1_evidence_sd':'group', 'n2_evidence_sd':'group',
                                                  'prior_mu':'group','prior_sd':'group'},
                                    fit_prior=True,
                                    fit_seperate_evidence_sd = True, 
                                    memory_model='independent',
                                    )
    if model_label == '3': 
        model = MagnitudeComparisonRegressionModel(df, 
                                    regressors = {'perceptual_noise_sd':'group', 'memory_noise_sd':'group'},
                                                  #'prior_mu':'group','prior_sd':'group'},
                                    fit_prior=False,
                                    fit_seperate_evidence_sd = True, 
                                    memory_model='shared_perceptual_noise',
                                    )
    if model_label == '4': 
        model = MagnitudeComparisonRegressionModel(df, 
                                    regressors = {'perceptual_noise_sd':'group', 'memory_noise_sd':'group',
                                                  'prior_mu':'group','prior_sd':'group'},
                                    fit_prior=True,
                                    fit_seperate_evidence_sd = True, 
                                    memory_model='shared_perceptual_noise',
                                    )
    if model_label == '5': # number sense VS. memory noise
        model = MagnitudeComparisonRegressionModel(df, 
                                    regressors = {'prior_sd':'group'},
                                    fit_prior=True,
                                    fit_seperate_evidence_sd = True, 
                                    memory_model='shared_perceptual_noise',
                                    )
    if model_label == '6': # n1 vs n2 noise
        model = MagnitudeComparisonRegressionModel(df, 
                                    regressors = {'prior_sd':'group'},
                                    fit_prior=True,
                                    fit_seperate_evidence_sd = True, 
                                    memory_model='independent',
                                    )
    if model_label == 'flexNoiseReg1':
        model = FlexibleNoiseComparisonRegressionModel(df, {'n1_evidence_sd':'group', 'n2_evidence_sd':'group'},  
                                                        fit_seperate_evidence_sd=True,
                                                        fit_prior=False,
                                                        polynomial_order=5, 
                                                        memory_model='independent')
    if model_label == 'flexNoiseReg2':
        model = FlexibleNoiseComparisonRegressionModel(df, {'evidence_sd':'group'},  
                                                        fit_seperate_evidence_sd=False,
                                                        fit_prior=False,
                                                        polynomial_order=5)                                
                                                        #memory_model='independent')
    if model_label == 'flexNoiseReg3':
        model = FlexibleNoiseComparisonRegressionModel(df, {'prior_sd':'group'},  
                                                        fit_seperate_evidence_sd=True,
                                                        fit_prior=True,
                                                        polynomial_order=5,                                
                                                        memory_model='independent')
    if model_label == 'DDM1Reg1':
        model = DDMMagnitudeComparisonRegressionModel(df, 
                                        regressors = {'perceptual_noise_sd':'group', 'memory_noise_sd':'group'},
                                        fit_prior=False,
                                        fit_seperate_evidence_sd = True, 
                                        memory_model='shared_perceptual_noise',
                                        )
    if model_label == 'DDM0':
        model = DDMMagnitudeComparisonModel(df, 
                                        fit_prior=False,
                                        fit_seperate_evidence_sd = True, 
                                        memory_model='independent' # default
                                        )
    if model_label == 'DDM1':
        model = DDMMagnitudeComparisonModel(df, 
                                        fit_prior=False,
                                        fit_seperate_evidence_sd = True, 
                                        memory_model='shared_perceptual_noise',
                                        )
    return model                              