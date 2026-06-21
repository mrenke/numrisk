import os.path as op
import os
import numpy as np
import pandas as pd
from bauer.models import RiskRegressionModel, FlexibleNoiseRiskModel, FlexibleNoiseRiskRegressionModel, PowerLawNoiseRiskRegressionModel,  AffineNoiseRiskModel, AffineNoiseRiskRegressionModel, PowerLawEncodingRiskModel, PowerLawEncodingRiskRegressionModel

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

    if model_label == 'riskKLW': #KLW model
        model = RiskRegressionModel(df, 
                                    regressors={},
                                    prior_estimate = 'klw',
                                    fit_seperate_evidence_sd = False,
                                    )
    elif model_label == 'risk':
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

        
# Natrual space - fixed!    
    elif model_label == 'powerLawEncNatSpace': # 
        model = PowerLawEncodingRiskModel(df, 
                                    fit_prior_mu=True,  # same as : prior_estimate = 'fix_prior_sd',   # 2 prior mus but fix prior sd - takes natural space into account                              
                                    fit_seperate_evidence_sd = True,  # different evidences for safe (n1) & risky (n2) options
                                    fixed_alpha=1.0
                                    )
    elif model_label == 'powerLawEncNatSpace_regression': # Natural Space! different evidences & prior_mus (but same prior_sd) for safe (n1) & risky (n2) options 
        model = PowerLawEncodingRiskModel(df, 
                                    regressors = {'n1_evidence_sd':'group', 'n2_evidence_sd':'group','risky_prior_mu':'group','safe_prior_mu':'group'},
                                    fit_prior_mu=True,
                                    fit_seperate_evidence_sd = True,  # different evidences for safe (n1) & risky (n2) options
                                    fixed_alpha = 1.0                # natural space specifciation!! (default =  False)
                                    )

# FlexNoise models - polynomial noise function of evidence magnitude, with different variants of what is fit and what is shared across conditions
    elif model_label == 'flexNoise1': # SAME FlexNoise  for safe (n1) & risky (n2) & 2 prior mus and SDs
        model = FlexibleNoiseRiskModel(df, 
                                    prior_estimate = 'full',
                                    polynomial_order=5, 
                                    fit_seperate_evidence_sd = False,
                                    )
    elif model_label == 'flexNoise2': # Different FlexNoise for safe (n1) & risky (n2) & 2 prior mus and SDs
        model = FlexibleNoiseRiskModel(df, 
                                    prior_estimate = 'full',
                                    polynomial_order=5, 
                                    fit_seperate_evidence_sd = True,
                                    )
    elif model_label == 'flexNoise1_regression': # FlexNoise SAME for safe (n1) & risky (n2) & 2 prior mus and SDs
        model = FlexibleNoiseRiskRegressionModel(df, 
                                     regressors = {'evidence_sd':'group', #'n1_evidence_sd':'group', 'n2_evidence_sd':'group',
                                                   'risky_prior_mu':'group','safe_prior_mu':'group',
                                                   'risky_prior_sd':'group','safe_prior_sd':'group'}, # 
                                    prior_estimate = 'full',
                                    polynomial_order=5, 
                                    fit_seperate_evidence_sd = False,
                                    )
    elif model_label == 'flexNoise1_regression2': # 11breg with old naming...
        model = FlexibleNoiseRiskRegressionModel(df, 
                                    regressors = {'safe_prior_sd':'group'}, # 
                                    prior_estimate = 'full',
                                    polynomial_order=5, 
                                    fit_seperate_evidence_sd = False,
                                    )
        
    elif model_label == 'affineNoise': # gilles version of Weber's law not holding...
        model = AffineNoiseRiskModel(df,
                                    #regressors={},
                                    prior_estimate='fix_prior_sd',
                                    fit_seperate_evidence_sd=True,
                                    )
    elif model_label == 'affineNoise_regression': # gilles version of Weber's law not holding...
        model = AffineNoiseRiskRegressionModel(df,
                                    regressors = {'n1_evidence_sd':'group', 'n2_evidence_sd':'group','risky_prior_mu':'group','safe_prior_mu':'group'}, # 
                                    prior_estimate='fix_prior_sd',
                                    fit_seperate_evidence_sd=True,
                                    )
        
# PowerLawEncoding models - power-law encoding of evidence, with different variants of what is fit and what is shared across conditions        
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
                                    fit_seperate_evidence_sd=False,
                                    )
    # allow prior_mus in representation space to be fit --
    elif model_label == 'powerLawEncoding3': # 
        model = PowerLawEncodingRiskModel(df,
                                     fit_prior_mu=True, 
                                     fit_seperate_evidence_sd=False,
                                    )
    elif model_label == 'powerLawEncoding3_regression0': # test...
        model = PowerLawEncodingRiskRegressionModel(df,
                                    regressors={},
                                    fit_prior_mu=True, 
                                    fit_seperate_evidence_sd=False,
                                    )
    elif model_label == 'powerLawEncoding3_regression': # 
        model = PowerLawEncodingRiskRegressionModel(df,
                                    regressors={'alpha': 'group', 'risky_prior_mu': 'group', 'safe_prior_mu': 'group'},
                                     fit_prior_mu=True, 
                                     fit_seperate_evidence_sd=False,
                                    )
    elif model_label == 'powerLawEncoding3_regression2': # 
        model = PowerLawEncodingRiskRegressionModel(df,
                                    regressors={'alpha': 'group'},
                                     fit_prior_mu=True, 
                                     fit_seperate_evidence_sd=False,
                                    )
    elif model_label == 'powerLawEncoding4': # 
        model = PowerLawEncodingRiskModel(df,
                                     fit_prior_mu=True, 
                                     fit_seperate_evidence_sd=True,
                                    )
    elif model_label == 'powerLawEncoding4_regression': # 
        model = PowerLawEncodingRiskRegressionModel(df,
                                    regressors={'alpha': 'group', 'n1_evidence_sd': 'group', 'n2_evidence_sd': 'group', 'risky_prior_mu': 'group', 'safe_prior_mu': 'group'},
                                     fit_prior_mu=True, 
                                     fit_seperate_evidence_sd=True,
                                    )
    elif model_label == 'powerLawEncoding4_regression1': # 
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
    
    elif model_label == 'powerLawEncoding5': # no prior!
        model = PowerLawEncodingRiskModel(df,
                                     flat_prior=True, 
                                     fit_seperate_evidence_sd=True,
                                    )
    
    
    else:
        raise ValueError(f'Unknown model label: {model_label}')

    return model


def get_rnp(evidence_sd, prior_std, p=0.55):
    beta = prior_std**2 / (evidence_sd**2 + prior_std**2)
    return np.clip(np.exp(-(1./beta) * np.log(1./p)), 0, 1)
