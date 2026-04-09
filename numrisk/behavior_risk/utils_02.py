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
    elif model_label == '2': # seperate priors
        model = RiskRegressionModel(df,regressors={'risky_prior_mu':'group','safe_prior_mu':'group','evidence_sd':'group'},
                                     prior_estimate='full',
                                     fit_seperate_evidence_sd = False,
                                     )
        
    elif model_label == '2b': # seperate priors
        model = RiskRegressionModel(df,regressors={'risky_prior_sd':'group','safe_prior_sd':'group', 'evidence_sd':'group'},
                                     prior_estimate='full',
                                     fit_seperate_evidence_sd = False,
                                     )
        
    elif model_label == '3': #KLW model
        model = RiskRegressionModel(df, 
                                    regressors = {'prior_sd':'group', # there is no prior_mu in the klw model
                                                  'evidence_sd':'group'},
                                    prior_estimate = 'klw',
                                    fit_seperate_evidence_sd = False,
                                    )
    elif model_label == '4':
        model = RiskLapseRegressionModel(df, 
                                    regressors = {'p_lapse':'group',
                                                  'prior_sd':'group',
                                                  'evidence_sd':'group'},
                                    prior_estimate = 'klw',
                                    fit_seperate_evidence_sd = False,
                                    )
    elif model_label == '5': # skeleton
        model = FlexibleNoiseRiskRegressionModel(df, 
                                    regressors = {'evidence_sd':'group','prior_sd':'group','prior_mu':'group'}, # 
                                    prior_estimate = 'shared',
                                    polynomial_order=5, 
                                    fit_seperate_evidence_sd = False,
                                    )
    elif model_label == '6': # different evidences for safe (n1) & risky (n2) options
        model = RiskRegressionModel(df, 
                                    regressors = {},
                                    prior_estimate = 'shared',
                                    fit_seperate_evidence_sd = True,
                                    )
    elif model_label == '6reg': # different evidences for safe (n1) & risky (n2) options + regressors
        model = RiskRegressionModel(df, 
                                    regressors = {'n1_evidence_sd':'group', 'n2_evidence_sd':'group',
                                                  'prior_mu':'group','prior_sd':'group'},
                                    prior_estimate = 'shared',
                                    fit_seperate_evidence_sd = True,
                                    )
    elif model_label == '7': # different evidences for safe (n1) & risky (n2) options & objective prior!
        model = RiskRegressionModel(df, 
                                    regressors = {},
                                    prior_estimate = 'objective',
                                    fit_seperate_evidence_sd = True,
                                    )
    elif model_label == '7reg': # different evidences for safe (n1) & risky (n2) options & objective prior + regressors
        model = RiskRegressionModel(df, 
                                    regressors = {'n1_evidence_sd':'group', 'n2_evidence_sd':'group'},
                                    prior_estimate = 'objective',
                                    fit_seperate_evidence_sd = True,
                                    )
    elif model_label == '8': # different evidences for safe (n1) & risky (n2) options & 2 prior mus but fix prior sd
        model = RiskRegressionModel(df, 
                                    regressors = {},
                                    prior_estimate = 'fix_prior_sd',
                                    fit_seperate_evidence_sd = True,
                                    )
    elif model_label == '8reg': # different evidences for safe (n1) & risky (n2) options & 2 prior mus but fix prior sd
        model = RiskRegressionModel(df, 
                                    regressors = {'n1_evidence_sd':'group', 'n2_evidence_sd':'group','risky_prior_mu':'group','safe_prior_mu':'group'},
                                    prior_estimate = 'fix_prior_sd',
                                    fit_seperate_evidence_sd = True,
                                    )
    elif model_label == '9': # FlexNoise for safe (n1) & risky (n2) & & 2 prior mus but fix prior sd
        model = FlexibleNoiseRiskRegressionModel(df, 
                                    regressors = {}, # 
                                    prior_estimate = 'fix_prior_sd',
                                    polynomial_order=5, 
                                    fit_seperate_evidence_sd = True,
                                    )
    elif model_label == '9reg': # FlexNoise for safe (n1) & risky (n2) & & 2 prior mus but fix prior sd
        model = FlexibleNoiseRiskRegressionModel(df, 
                                    regressors = {'n1_evidence_sd':'group', 'n2_evidence_sd':'group','risky_prior_mu':'group','safe_prior_mu':'group'}, # 
                                    prior_estimate = 'fix_prior_sd',
                                    polynomial_order=5, 
                                    fit_seperate_evidence_sd = True,
                                    )
    elif model_label == '10': # skeleton
        model = FlexibleNoiseRiskRegressionModel(df, 
                                    regressors = {'evidence_sd':'group'}, # 
                                    prior_estimate = 'shared',
                                    polynomial_order=5, 
                                    fit_seperate_evidence_sd = False,
                                    )
    elif model_label == '11': # FlexNoise SAME for safe (n1) & risky (n2) & 2 prior mus and SDs
        model = FlexibleNoiseRiskRegressionModel(df, 
                                    regressors = {}, # 
                                    prior_estimate = 'full',
                                    polynomial_order=5, 
                                    fit_seperate_evidence_sd = False,
                                    )
    elif model_label == '11reg': # FlexNoise SAME for safe (n1) & risky (n2) & 2 prior mus and SDs
        # evidence_sd:'group' removed: probit model found no general group noise effect (p_bayes=0.267)
        # group regressors target prior SD asymmetry only, consistent with probit RNP:SS finding (p_bayes=0.979)
        model = FlexibleNoiseRiskRegressionModel(df,
                                    regressors = {'risky_prior_sd':'group','safe_prior_sd':'group' },
                                    prior_estimate = 'full',
                                    polynomial_order=5,
                                    fit_seperate_evidence_sd = False,
                                    )
    elif model_label == '11breg': # FlexNoise SAME for safe (n1) & risky (n2) & 2 prior mus and SDs
        model = FlexibleNoiseRiskRegressionModel(df, 
                                    regressors = {'safe_prior_sd':'group'}, # 
                                    prior_estimate = 'full',
                                    polynomial_order=5, 
                                    fit_seperate_evidence_sd = False,
                                    )
        
    elif model_label == '12': # like model-8 but Natural Space! different evidences & prior_mus (but same prior_sd) for safe (n1) & risky (n2) options 
        model = RiskRegressionModel(df, 
                                    regressors = {},
                                    prior_estimate = 'fix_prior_sd',   # 2 prior mus but fix prior sd - takes natural space into account                              
                                    fit_seperate_evidence_sd = True,  # different evidences for safe (n1) & risky (n2) options
                                    natural_space = True                # natural space specifciation!! (default =  False)
                                    )
    elif model_label == '12reg': # Natural Space! different evidences & prior_mus (but same prior_sd) for safe (n1) & risky (n2) options 
        model = RiskRegressionModel(df, 
                                    regressors = {'n1_evidence_sd':'group', 'n2_evidence_sd':'group','risky_prior_mu':'group','safe_prior_mu':'group'},
                                    prior_estimate = 'fix_prior_sd',   # 2 prior mus but fix prior sd - takes natural space into account                              
                                    fit_seperate_evidence_sd = True,  # different evidences for safe (n1) & risky (n2) options
                                    natural_space = True                # natural space specifciation!! (default =  False)
                                    )
    elif model_label == '13': # Normal RiskRegression + full prior (asymmetric prior SDs & mus for risky vs safe), shared evidence SD
        # Completes the 2x2 grid:
        #   evidence SD asymmetry (n1≠n2) → model-8 / 8reg
        #   prior SD asymmetry (risky≠safe) → model-13 / 13reg   ← this
        #   FlexNoise equivalents: model-9/9reg and model-11/11reg already exist
        model = RiskRegressionModel(df,
                                    regressors = {},
                                    prior_estimate = 'full',         # free risky_prior_mu, risky_prior_sd, safe_prior_mu, safe_prior_sd
                                    fit_seperate_evidence_sd = False, # shared evidence SD (asymmetry lives entirely in priors)
                                    )
    elif model_label == '13reg': # model-13 + group regressors on asymmetric prior SDs
        # evidence_sd:'group' omitted: probit model found no general group noise effect (p_bayes=0.267)
        # group regressors target prior SD asymmetry only, consistent with probit RNP:SS finding (p_bayes=0.979)
        model = RiskRegressionModel(df,
                                    regressors = {'risky_prior_sd': 'group',
                                                  'safe_prior_sd':  'group'},
                                    prior_estimate = 'full',
                                    fit_seperate_evidence_sd = False,
                                    )
    else :
        raise ValueError(f'Unknown model label: {model_label}')
    
    return model


def get_rnp(evidence_sd, prior_std, p=0.55):
    beta = prior_std**2 / (evidence_sd**2 + prior_std**2)
    return np.clip(np.exp(-(1./beta) * np.log(1./p)), 0, 1)



    