import os.path as op
import os
import numpy as np
import pandas as pd
from bauer.models import RiskRegressionModel, RiskLapseRegressionModel, FlexibleNoiseRiskRegressionModel
#from stress_risk.utils.data import get_all_behavior
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss



def build_model(model_label, df):
    if model_label == '1': # same priors = probit model ?!
        model = RiskRegressionModel(df, 
                                    regressors = {'prior_mu':'group','evidence_sd':'group'},
                                    prior_estimate = 'shared',
                                    fit_seperate_evidence_sd = False,
                                    )
    if model_label == '2': # seperate priors
        model = RiskRegressionModel(df,regressors={'risky_prior_mu':'group', 'risky_prior_std':'group',
                                    'safe_prior_mu':'group', 'safe_prior_std':'group',
                                    'evidence_sd':'group'},
                                     prior_estimate='full',
                                     fit_seperate_evidence_sd = False,
                                     )
    if model_label == '3': #KLW model
        model = RiskRegressionModel(df, 
                                    regressors = {'prior_sd':'group', # there is no prior_mu in the klw model
                                                  'evidence_sd':'group'},
                                    prior_estimate = 'klw',
                                    fit_seperate_evidence_sd = False,
                                    )
    if model_label == '4':
        model = RiskLapseRegressionModel(df, 
                                    regressors = {'p_lapse':'group',
                                                  'prior_sd':'group',
                                                  'evidence_sd':'group'},
                                    prior_estimate = 'klw',
                                    fit_seperate_evidence_sd = False,
                                    )
    if model_label == '5': # skeleton
        model = FlexibleNoiseRiskRegressionModel(df, 
                                    regressors = {'evidence_sd':'group','prior_sd':'group','prior_mu':'group'}, # 
                                    prior_estimate = 'shared',
                                    polynomial_order=5, 
                                    fit_seperate_evidence_sd = False,
                                    )
    return model


def get_rnp(evidence_sd, prior_std, p=0.55):
    beta = prior_std**2 / (evidence_sd**2 + prior_std**2)
    return np.clip(np.exp(-(1./beta) * np.log(1./p)), 0, 1)



    