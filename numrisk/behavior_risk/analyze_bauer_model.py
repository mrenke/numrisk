from itertools import combinations
#import pingouin
import matplotlib.pyplot as plt
import seaborn
import os
import os.path as op
import argparse
import arviz as az
import numpy as np
import seaborn as sns
from bauer.utils.bayes import softplus
import pandas as pd
from os import listdir, remove

from utils import get_data
from utils_02 import build_model, get_rnp
from bauer.models import PowerLawNoiseRiskRegressionModel
# behav_fit3
# does only work when executed via terminal, not in interactive shell of VSC

def main(model_label, bids_folder='/Users/mrenke/data/ds-dnumrisk',format='non-symbolic',#col_wrap=5, AUC=False,E_dif=False, 
        plot_traces=True,
        remove_subjects = True, remove_sub_string = '32-40-45-46-50'):

    sns.set_context('talk')

    subject_list = [f[4:] for f in listdir(bids_folder) if f[0:3] == 'sub' and len(f) == 6]
    if remove_subjects:
        remove_sub_list = [f'{int(s):02d}' for s in remove_sub_string.split('-')]
        subject_list = [subject for subject in subject_list if subject not in remove_sub_list]
    
    df = get_data(bids_folder,subject_list)
    df = df.xs(format,0, level='format')
    model = build_model(model_label, df)
    model.build_estimation_model()

    if remove_subjects:
        model_label = f'{model_label}_rem-{remove_sub_string}'
    idata = az.from_netcdf(op.join(bids_folder, f'derivatives/cogmodels_risk/model-{model_label}_format-{format}_trace.netcdf'))

    target_folder = op.join(bids_folder, f'derivatives/cogmodels_risk/figures/{model_label}_format-{format}')
    if not op.exists(target_folder):
        os.makedirs(target_folder)

    if plot_traces:
        az.plot_trace(idata, var_names=['~p'])
        plt.savefig(op.join(target_folder, 'traces.pdf'))

    if model.prior_estimate == 'klw':
        idata.posterior['rnp'] = get_rnp(idata.posterior['evidence_sd'], idata.posterior['prior_sd'])
        idata.posterior['rnp_mu'] = get_rnp(idata.posterior['evidence_sd_mu'], idata.posterior['prior_sd_mu'])
        model.free_parameters['rnp'] = '' # appending to a dictionary

    for par in model.free_parameters:
        traces = idata.posterior[par+'_mu'].to_dataframe()

        par_helper = par if par != 'rnp' else 'evidence_sd'

        for regressor, t in traces.groupby(par_helper+'_regressors'):
            t = t.copy()
            print(regressor, t)
            if ('sd' in par) & (regressor == 'Intercept'): #  'risky_prior_std', 'safe_prior_std', 'n1_evidence_sd', 'n2_evidence_sd',
                t = softplus(t)

            plt.figure()
            sns.kdeplot(t, fill=True)
            if regressor != 'Intercept':
                plt.axvline(0.0, c='k', ls='--')
                txt = f'p({par} < 0.0) = {np.round((t.values < 0.0).mean(), 3)}'
                plt.xlabel(txt)

            else:
                if par == 'risky_prior_mu':
                    plt.axvline(np.log(df['n_risky']).mean(), c='k', ls='--')
                elif par == 'risky_prior_sd':
                    plt.axvline(np.log(df['n_risky']).std(), c='k', ls='--')
                elif par == 'safe_prior_mu':
                    for n_safe in np.log([7., 10., 14., 20., 28.]):
                        plt.axvline(n_safe, c='k', ls='--')

                    plt.axvline(np.log(df['n_safe']).mean(), c='k', ls='--', lw=2)
                elif par == 'safe_prior_sd':
                    plt.axvline(np.log(df['n_safe']).std(), c='k', ls='--')
            
            sns.despine()
            plt.savefig(op.join(target_folder, f'group_par-{par}.{regressor}.pdf'), bbox_inches='tight')
            plt.close()

    if isinstance(model, PowerLawNoiseRiskRegressionModel):
        plot_sd_curves(idata, df, target_folder)





def plot_sd_curves(idata, df, target_folder):
    """Plot SD(n) = exp(intercept) * n^noise_exponent for PowerLaw models.

    Log-log axes: the power law is a straight line, slope = noise_exponent.
    Shows group-level posterior mean + 94% HDI band, individual subject means,
    and Weber's law (slope=1) as a reference.
    """
    post = idata.posterior
    x = np.exp(np.linspace(np.log(df[['n1', 'n2']].min().min()),
                           np.log(df[['n1', 'n2']].max().max()), 100))

    def _group_curves(intercept_key):
        intercepts = post[intercept_key + '_mu'].values[..., 0].ravel()  # Intercept only
        exponents  = post['noise_exponent_mu'].values[..., 0].ravel()
        return np.exp(intercepts[:, None] + exponents[:, None] * np.log(x)[None, :])

    def _subject_curves(intercept_key):
        intercepts = post[intercept_key].values.mean(axis=(0, 1))[..., 0]  # (subject,)
        exponents  = post['noise_exponent'].values.mean(axis=(0, 1))[..., 0]
        return np.exp(intercepts[:, None] + exponents[:, None] * np.log(x)[None, :])

    sd_n1_group = _group_curves('n1_log_sd_intercept')
    sd_n2_group = _group_curves('n2_log_sd_intercept')
    sd_n1_subj  = _subject_curves('n1_log_sd_intercept')
    sd_n2_subj  = _subject_curves('n2_log_sd_intercept')

    exp_mean = float(post['noise_exponent_mu'].sel(noise_exponent_regressors='Intercept').values.mean())
    exp_hdi  = az.hdi(idata, var_names=['noise_exponent_mu'])['noise_exponent_mu'].sel(noise_exponent_regressors='Intercept').values

    fig, ax = plt.subplots(figsize=(6, 5))

    # individual subjects
    for curve in sd_n1_subj:
        ax.plot(x, curve, color='steelblue', alpha=0.15, lw=0.8)
    for curve in sd_n2_subj:
        ax.plot(x, curve, color='tomato', alpha=0.15, lw=0.8)

    # group HDI
    ax.fill_between(x, np.percentile(sd_n1_group, 3, axis=0), np.percentile(sd_n1_group, 97, axis=0),
                    color='steelblue', alpha=0.25)
    ax.fill_between(x, np.percentile(sd_n2_group, 3, axis=0), np.percentile(sd_n2_group, 97, axis=0),
                    color='tomato', alpha=0.25)

    # group mean
    ax.plot(x, sd_n1_group.mean(axis=0), color='steelblue', lw=2, label='safe (n1)')
    ax.plot(x, sd_n2_group.mean(axis=0), color='tomato',    lw=2, label='risky (n2)')

    # Weber's law reference anchored at midpoint of n1 group mean
    mid   = len(x) // 2
    anchor = sd_n1_group.mean(axis=0)[mid]
    ax.plot(x, anchor * (x / x[mid]) ** 1.0, 'k--', lw=1, alpha=0.5, label="Weber's law (slope=1)")

    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xlabel('Magnitude (n)')
    ax.set_ylabel('SD(n)')
    ax.set_title(f'noise_exponent = {exp_mean:.2f} [{exp_hdi[0]:.2f}, {exp_hdi[1]:.2f}]')
    ax.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(op.join(target_folder, 'sd_curves.pdf'), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/Users/mrenke/data/ds-dnumrisk')
    #parser.add_argument('--AUC', action='store_true')
    #parser.add_argument('--E_dif', action='store_true')
    parser.add_argument('--format', default='non-symbolic')
    parser.add_argument('--no_trace', dest='plot_traces', action='store_false')
    parser.add_argument('--keep_all_subjects', action='store_false', dest='remove_subjects')
    parser.add_argument('--remove_sub_string', default='32-40-45-46-50') # default='32-40-45-46-50'
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder, plot_traces=args.plot_traces, format=args.format, remove_subjects=args.remove_subjects, remove_sub_string=args.remove_sub_string) # , AUC=args.AUC, E_dif=args.E_dif