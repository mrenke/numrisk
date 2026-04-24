import os.path as op
import os
import numpy as np
import pandas as pd
from bauer.models import RiskRegressionModel, FlexibleNoiseRiskRegressionModel, PowerLawNoiseRiskRegressionModel,  AffineNoiseRiskModel, PowerLawEncodingRiskModel, PowerLawEncodingRiskRegressionModel

import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss


def build_model(model_label, df):
    """
    Two model families:

    'risk'   — RiskRegressionModel: fix_prior_sd (free risky/safe prior mus),
               separate evidence SDs for safe (n1) and risky (n2) options.

    'power'  — PowerLawNoiseRiskRegressionModel: power-law magnitude-dependent noise,
               fix_prior_sd, separate noise intercepts per option.

    """
    if model_label == 'risk':
        model = RiskRegressionModel(df,
                                    regressors={},
                                    prior_estimate='fix_prior_sd',
                                    fit_seperate_evidence_sd=True,
                                    )
    elif model_label == 'risk_regression':
        model = RiskRegressionModel(df,
                                    regressors={'n1_evidence_sd':'group', 'n2_evidence_sd':'group', 
                                                'risky_prior_mu':'group','safe_prior_mu':'group', 
                                                'prior_sd':'group'},
                                    prior_estimate='fix_prior_sd',
                                    fit_seperate_evidence_sd=True,
                                    )
    elif model_label == '11breg': # FlexNoise SAME for safe (n1) & risky (n2) & 2 prior mus and SDs
        model = FlexibleNoiseRiskRegressionModel(df, 
                                    regressors = {'safe_prior_sd':'group'}, # 
                                    prior_estimate = 'full',
                                    polynomial_order=5, 
                                    fit_seperate_evidence_sd = False,
                                    )
        

    elif model_label == 'power':
        model = PowerLawNoiseRiskRegressionModel(df,
                                    regressors={},
                                    prior_estimate='fix_prior_sd',
                                    fit_seperate_evidence_sd=True,
                                    )
    elif model_label == 'power_regression':
        model = PowerLawNoiseRiskRegressionModel(df,
                                    regressors={ 'noise_exponent':'group', 'n1_evidence_sd':'group', 'n2_evidence_sd':'group', 
                                                #'risky_prior_mu':'group','safe_prior_mu':'group', 'prior_sd':'group'},
                                                },
                                    prior_estimate='fix_prior_sd',
                                    fit_seperate_evidence_sd=True,
                                    )
    elif model_label == 'affineNoise': # gilles version of Weber's law not holding...
        model = AffineNoiseRiskModel(df,
                                    #regressors={},
                                    prior_estimate='fix_prior_sd',
                                    fit_seperate_evidence_sd=True,
                                    )
    elif model_label == 'powerLawEncoding': # 
        model = PowerLawEncodingRiskModel(df,
                                    #regressors={},
                                    # prior_estimate='fix_prior_sd', defaults to fit_prior=False - otherwise dievergence issues
                                    fit_seperate_evidence_sd=True,
                                    )
    elif model_label == 'powerLawEncoding_regression0': # 
        model = PowerLawEncodingRiskRegressionModel(df,
                                    regressors={},
                                    fit_seperate_evidence_sd=True,
                                    )
    elif model_label == 'powerLawEncoding_regression': # no group alpha regression
        model = PowerLawEncodingRiskRegressionModel(df,
                                    regressors={'alpha': 'group','n1_evidence_sd': 'group', 'n2_evidence_sd': 'group'},
                                    fit_seperate_evidence_sd=True,
                                    fit_prior_mu=False,
                                    )
    elif model_label == 'powerLawEncoding_regression2': # no group alpha regression
        model = PowerLawEncodingRiskRegressionModel(df,
                                    regressors={'n1_evidence_sd': 'group', 'n2_evidence_sd': 'group'},
                                    fit_seperate_evidence_sd=True,
                                    fit_prior_mu=False,
                                    )
    elif model_label == 'powerLawEncoding2': # 
        model = PowerLawEncodingRiskModel(df,
                                     fit_prior_mu=False, # added this to differentiate from next one
                                    )
    elif model_label == 'powerLawEncoding3': # 
        model = PowerLawEncodingRiskModel(df,
                                     fit_prior_mu=True, 
                                     fit_seperate_evidence_sd=False,
                                    )
    elif model_label == 'powerLawEncoding3_regression0': # 
        model = PowerLawEncodingRiskRegressionModel(df,
                                    regressors={},
                                    fit_prior_mu=True, 
                                    fit_seperate_evidence_sd=False,
                                    )
    elif model_label == 'powerLawEncoding3_regression': # 
        model = PowerLawEncodingRiskRegressionModel(df,
                                    regressors={'alpha': 'group'},
                                     fit_prior_mu=True, 
                                     fit_seperate_evidence_sd=False,
                                    )
    elif model_label == 'powerLawEncoding4': # 
        model = PowerLawEncodingRiskModel(df,
                                    #regressors={'alpha': 'group', 'n1_evidence_sd': 'group', 'n2_evidence_sd': 'group'},
                                     fit_prior_mu=True, 
                                     fit_seperate_evidence_sd=True,
                                    )
    elif model_label == 'powerLawEncoding4_regression': # 
        model = PowerLawEncodingRiskRegressionModel(df,
                                    regressors={'alpha': 'group', 'n1_evidence_sd': 'group', 'n2_evidence_sd': 'group'},
                                     fit_prior_mu=True, 
                                     fit_seperate_evidence_sd=True,
                                    )
    elif model_label == 'powerLawEncoding4_regression2': # 
        model = PowerLawEncodingRiskRegressionModel(df,
                                    regressors={'n1_evidence_sd': 'group'},# , 'n2_evidence_sd': 'group' #'alpha': 'group', 'risky_prior_mu': 'group', 'safe_prior_mu': 'group'},
                                     fit_prior_mu=True, 
                                     fit_seperate_evidence_sd=True,
                                    )                        
    else:
        raise ValueError(f'Unknown model label: {model_label}')

    return model


def get_rnp(evidence_sd, prior_std, p=0.55):
    beta = prior_std**2 / (evidence_sd**2 + prior_std**2)
    return np.clip(np.exp(-(1./beta) * np.log(1./p)), 0, 1)
