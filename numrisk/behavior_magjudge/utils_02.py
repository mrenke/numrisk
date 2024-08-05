import os.path as op
import os
import numpy as np
import pandas as pd
from bauer.models import MagnitudeComparisonRegressionModel, FlexibleNoiseComparisonRegressionModel

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
        df_param_int = df_param.xs('Intercept', 0,'regressor')
        df_param_group = df_param.xs('group', 0,'regressor')
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
    return model                              