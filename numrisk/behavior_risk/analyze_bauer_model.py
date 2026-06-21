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
from bauer.models import PowerLawNoiseRiskRegressionModel, AffineNoiseRiskModel, PowerLawEncodingRiskModel, PowerLawEncodingRiskRegressionModel, FlexibleNoiseRiskModel
from bauer.core import RegressionModel
from bauer.utils.bayes import softplus as _softplus
# behav_fit3
# does only work when executed via terminal, not in interactive shell of VSC

def main(model_label, bids_folder='/Users/mrenke/data/ds-dnumrisk',format='symbolic',#col_wrap=5, AUC=False,E_dif=False, 
        plot_traces=False,
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

        regressors_key = par_helper + '_regressors'
        if regressors_key in traces.index.names:
            groups = traces.groupby(regressors_key)
        else:
            # No regressors dimension (e.g. affineNoise with regressors={}):
            # treat the whole posterior as a single "Intercept" group
            groups = [('Intercept', traces)]

        for regressor, t in groups:
            t = t.copy()
            print(regressor, t)
            par_transform = model.free_parameters.get(par, {}).get('transform', 'identity') if par != 'rnp' else 'identity'
            needs_softplus = (regressor == 'Intercept') and (
                ('sd' in par) or
                (par == 'alpha' and isinstance(model, RegressionModel) and par_transform == 'softplus')
            )
            if needs_softplus:
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

    if isinstance(model, (PowerLawNoiseRiskRegressionModel, AffineNoiseRiskModel, PowerLawEncodingRiskModel, FlexibleNoiseRiskModel)):
        plot_sd_curves(idata, df, target_folder, model)





def _flat(arr):
    """Flatten chain+draw dims, keeping any trailing dims (regressors, subjects)."""
    return arr.reshape(-1, *arr.shape[2:])


def _intercept(arr):
    """Select the Intercept regressor (last dim index 0) and flatten chain/draw."""
    return _flat(arr)[..., 0] if arr.ndim > 2 else _flat(arr)


def _grp_powerlawnoise(post, x, key):
    ic = _intercept(post[key + '_mu'].values)
    ex = _intercept(post['noise_exponent_mu'].values)
    return np.exp(ic[:, None] + ex[:, None] * np.log(x)[None, :])


def _sub_powerlawnoise(post, x, key):
    ic = _flat(post[key].values).mean(axis=0)[..., 0]
    ex = _flat(post['noise_exponent'].values).mean(axis=0)[..., 0]
    return np.exp(ic[:, None] + ex[:, None] * np.log(x)[None, :])


def _grp_affine(post, x_norm, key):
    b0 = _flat(post[f'{key}_spline1_mu'].values).ravel()
    b1 = _flat(post[f'{key}_spline2_mu'].values).ravel()
    return _softplus(b0[:, None] + b1[:, None] * x_norm[None, :])


def _grp_encoding(post, x, key, alpha_grp):
    sd_rep = _softplus(_intercept(post[key + '_mu'].values))
    return sd_rep[:, None] / (alpha_grp[:, None] * x[None, :] ** (alpha_grp[:, None] - 1))


def _sub_encoding(post, x, key, alpha_needs_softplus=False):
    def _sub_mean(v):
        m = _flat(v).mean(axis=0)
        return m[..., 0] if m.ndim > 1 else m  # select Intercept only when regressors dim exists

    alpha_raw = _sub_mean(post['alpha'].values)
    alpha_s   = _softplus(alpha_raw) if alpha_needs_softplus else alpha_raw
    sd_s      = _softplus(_sub_mean(post[key].values))
    return sd_s[:, None] / (alpha_s[:, None] * x[None, :] ** (alpha_s[:, None] - 1))


def _grp_flexnoise(model, post, x, key):
    """Group-level FlexNoise SD curve: (n_draws, len(x))."""
    labels1, labels2 = model._get_evidence_sd_spline_par_labels()
    base_labels = labels2 if 'n2' in key else labels1
    mu_labels = [f'{l}_mu' for l in base_labels]
    dm_var = key if key != 'evidence_sd' else 'n1_evidence_sd'
    dm = model.make_dm(x=x, variable=dm_var)
    pars = np.stack([_flat(post[l].values).ravel() for l in mu_labels], axis=1)
    return _softplus(pars @ dm.T)


def _sub_flexnoise(model, post, x, key):
    """Per-subject mean FlexNoise SD curves: (n_subjects, len(x))."""
    labels1, labels2 = model._get_evidence_sd_spline_par_labels()
    labels = labels2 if 'n2' in key else labels1
    dm_var = key if key != 'evidence_sd' else 'n1_evidence_sd'
    dm = model.make_dm(x=x, variable=dm_var)
    pars = np.stack([_flat(post[l].values).mean(axis=0) for l in labels], axis=1)
    return _softplus(pars @ dm.T)


def _draw_curves(x, curve_specs, ylabel, title, target_folder):
    """Shared plot: group HDI band + mean, optional subject thin lines, Weber ref.

    curve_specs: list of (grp_arr, sub_arr_or_None, color, label)
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    mid = len(x) // 2

    for grp, sub, color, label in curve_specs:
        if sub is not None:
            for curve in sub:
                ax.plot(x, curve, color=color, alpha=0.15, lw=0.8)
        ax.fill_between(x, np.percentile(grp, 3, axis=0), np.percentile(grp, 97, axis=0),
                        color=color, alpha=0.25)
        ax.plot(x, grp.mean(axis=0), color=color, lw=2, label=label)

    anchor = curve_specs[0][0].mean(axis=0)[mid]
    ax.plot(x, anchor * (x / x[mid]), 'k--', lw=1, alpha=0.5, label="Weber's law (slope=1)")

    ax.set_xlabel('Magnitude (n)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(op.join(target_folder, 'sd_curves.pdf'), bbox_inches='tight')
    plt.close()


def plot_sd_curves(idata, df, target_folder, model):
    """Dispatch SD-curve plotting for PowerLawNoise, AffineNoise, PowerLawEncoding."""
    post = idata.posterior
    n_min = df[['n1', 'n2']].min().min()
    n_max = df[['n1', 'n2']].max().max()
    separate = model.fit_seperate_evidence_sd

    if isinstance(model, PowerLawNoiseRiskRegressionModel):
        x = np.exp(np.linspace(np.log(n_min), np.log(n_max), 100))
        e_vals = _intercept(post['noise_exponent_mu'].values)
        e_mean, e_lo, e_hi = e_vals.mean(), np.percentile(e_vals, 3), np.percentile(e_vals, 97)
        title = f'PowerLaw: SD(n)=exp(c)·nᵉ,  e={e_mean:.2f} [{e_lo:.2f}, {e_hi:.2f}]'
        if separate:
            curve_specs = [
                (_grp_powerlawnoise(post, x, 'n1_log_sd_intercept'),
                 _sub_powerlawnoise(post, x, 'n1_log_sd_intercept'), 'steelblue', 'safe (n1)'),
                (_grp_powerlawnoise(post, x, 'n2_log_sd_intercept'),
                 _sub_powerlawnoise(post, x, 'n2_log_sd_intercept'), 'tomato', 'risky (n2)'),
            ]
        else:
            curve_specs = [
                (_grp_powerlawnoise(post, x, 'log_sd_intercept'),
                 _sub_powerlawnoise(post, x, 'log_sd_intercept'), 'steelblue', 'shared noise'),
            ]
        ylabel = 'SD(n)'

    elif isinstance(model, AffineNoiseRiskModel):
        x = np.linspace(n_min, n_max, 100)
        x_norm = (x - n_min) / (n_max - n_min)
        title = 'AffineNoise: σ(n) = softplus(β₀ + β₁·n̂)'
        if separate:
            curve_specs = [
                (_grp_affine(post, x_norm, 'n1_evidence_sd'), None, 'steelblue', 'safe (n1)'),
                (_grp_affine(post, x_norm, 'n2_evidence_sd'), None, 'tomato',    'risky (n2)'),
            ]
        else:
            curve_specs = [
                (_grp_affine(post, x_norm, 'evidence_sd'), None, 'steelblue', 'shared noise'),
            ]
        ylabel = 'σ(n)'

    elif isinstance(model, PowerLawEncodingRiskModel):
        x = np.exp(np.linspace(np.log(n_min), np.log(n_max), 100))
        _alpha_transform = model.free_parameters.get('alpha', {}).get('transform', 'identity')
        _alpha_raw = _intercept(post['alpha_mu'].values)
        # RegressionModel stores raw (pre-softplus) values; BaseModel stores already-transformed values
        alpha_grp = _softplus(_alpha_raw) if (isinstance(model, RegressionModel) and _alpha_transform == 'softplus') else _alpha_raw
        a_mean = float(alpha_grp.mean())
        title = f'PowerLawEncoding: r=nᵅ, const SD in rep-space,  α={a_mean:.2f}'
        if separate:
            curve_specs = [
                (_grp_encoding(post, x, 'n1_evidence_sd', alpha_grp),
                 _sub_encoding(post, x, 'n1_evidence_sd', _alpha_transform == 'softplus' and isinstance(model, RegressionModel)), 'steelblue', 'safe (n1)'),
                (_grp_encoding(post, x, 'n2_evidence_sd', alpha_grp),
                 _sub_encoding(post, x, 'n2_evidence_sd', _alpha_transform == 'softplus' and isinstance(model, RegressionModel)), 'tomato',    'risky (n2)'),
            ]
        else:
            curve_specs = [
                (_grp_encoding(post, x, 'evidence_sd', alpha_grp),
                 _sub_encoding(post, x, 'evidence_sd', _alpha_transform == 'softplus' and isinstance(model, RegressionModel)), 'steelblue', 'shared noise'),
            ]
        ylabel = 'Effective SD(n)'

    elif isinstance(model, FlexibleNoiseRiskModel):
        x = np.linspace(n_min, n_max, 100)
        poly = model.polynomial_order
        title = f'FlexNoise B-spline, order={poly}'
        if separate:
            curve_specs = [
                (_grp_flexnoise(model, post, x, 'n1_evidence_sd'),
                 _sub_flexnoise(model, post, x, 'n1_evidence_sd'), 'steelblue', 'safe (n1)'),
                (_grp_flexnoise(model, post, x, 'n2_evidence_sd'),
                 _sub_flexnoise(model, post, x, 'n2_evidence_sd'), 'tomato',    'risky (n2)'),
            ]
        else:
            curve_specs = [
                (_grp_flexnoise(model, post, x, 'evidence_sd'),
                 _sub_flexnoise(model, post, x, 'evidence_sd'), 'steelblue', 'shared noise'),
            ]
        ylabel = 'σ(n)'

    else:
        return

    _draw_curves(x, curve_specs, ylabel, title, target_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/Users/mrenke/data/ds-dnumrisk')
    #parser.add_argument('--AUC', action='store_true')
    #parser.add_argument('--E_dif', action='store_true')
    parser.add_argument('--format', default='symbolic')
    parser.add_argument('--trace', dest='plot_traces', action='store_true')
    parser.add_argument('--keep_all_subjects', action='store_false', dest='remove_subjects')
    parser.add_argument('--remove_sub_string', default='32-40-45-46-50') # default='32-40-45-46-50'
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder, plot_traces=args.plot_traces, format=args.format, remove_subjects=args.remove_subjects, remove_sub_string=args.remove_sub_string) # , AUC=args.AUC, E_dif=args.E_dif