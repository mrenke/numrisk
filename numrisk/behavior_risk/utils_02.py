import os.path as op
import os
import numpy as np
import pandas as pd
from bauer.models import RiskRegressionModel, RiskLapseRegressionModel
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
    if model_label == '3':
        model = RiskLapseRegressionModel(df, 
                                    regressors = {'prior_mu':'group','evidence_sd':'group'},
                                    prior_estimate = 'shared',
                                    fit_seperate_evidence_sd = False,
                                    )
    return model
    