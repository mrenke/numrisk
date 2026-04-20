import os.path as op
import os
import numpy as np
import pandas as pd
from bauer.models import RiskRegressionModel, PowerLawNoiseRiskRegressionModel, AffineNoiseRiskModel
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
    
    else:
        raise ValueError(f'Unknown model label: {model_label}')

    return model


def get_rnp(evidence_sd, prior_std, p=0.55):
    beta = prior_std**2 / (evidence_sd**2 + prior_std**2)
    return np.clip(np.exp(-(1./beta) * np.log(1./p)), 0, 1)
