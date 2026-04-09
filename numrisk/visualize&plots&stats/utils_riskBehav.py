from itertools import product
import numpy as np
import pandas as pd

def summarize_posterior(samples):
    """Compute mean, 95% CI, and P(>0) for a posterior array."""
    samples = np.array(samples, dtype=float)
    return {
        'mean': np.mean(samples),
        'ci_low': np.percentile(samples, 2.5),
        'ci_high': np.percentile(samples, 97.5),
        'p_gt_0': np.round(np.mean(samples > 0), 3)
    }

def fit_slope(rnp_sample, group=None, fmt=None):
    """Fit a line of RNP ~ n_safe for a given group/format subset. Returns (slope, intercept)."""
    tmp = rnp_sample.copy()
    if group is not None:
        tmp = tmp.xs(group, level='group')
    if fmt is not None:
        tmp = tmp.xs(fmt, level='format')
    #tmp = tmp.reset_index('n_safe')
    A = np.vstack([tmp['n_safe'], np.ones(len(tmp))]).T
    slope, intercept = np.linalg.lstsq(A, tmp['rnp'], rcond=None)[0]
    return slope, intercept


def compute_rnp_effects(rnp, model_name='model'):
    """
    Compute all RNP-related posterior effects and return as a tidy DataFrame.
    
    Parameters
    ----------
    rnp : xarray DataArray or similar, indexed by (chain, draw, subject, group, format, n_safe)
    model_name : str, label for the model version
    
    Returns
    -------
    pd.DataFrame with columns: model, effect, mean, ci_low, ci_high, p_gt_0
    """
    n_chains = rnp.index.unique('chain').values
    n_draws  = rnp.index.unique('draw').values

    # ------------------------------------------------------------------ #
    # 1. Simple contrasts (no n_safe slope fitting needed)
    # ------------------------------------------------------------------ #
    simple_effects = {
        'RNP':        rnp,
        'RNP:format': rnp.xs('non-symbolic', level='format') - rnp.xs('symbolic',    level='format'),
        'RNP:group':  rnp.xs('Dyscalculic',  level='group')  - rnp.xs('Control',     level='group'),
    }

    rows = []

    for name, posterior in simple_effects.items():
        stats = summarize_posterior(posterior.values.ravel())
        rows.append({'model': model_name, 'effect': name, **stats})

    # ------------------------------------------------------------------ #
    # 2. Slope-based effects (iterate over chain x draw)
    # ------------------------------------------------------------------ #
    # Storage for per-draw slopes — we need these for both direct reporting
    # and for computing higher-order slope differences
    slope_store = {k: [] for k in [
        'ctrl',           # RNP~n_safe | Control
        'dysc',           # RNP~n_safe | Dyscalculic
        'nonsym',         # RNP~n_safe | non-symbolic
        'sym',            # RNP~n_safe | symbolic
        'ctrl_nonsym',    # RNP~n_safe | Control x non-symbolic
        'ctrl_sym',       # RNP~n_safe | Control x symbolic
        'dysc_nonsym',    # RNP~n_safe | Dyscalculic x non-symbolic
        'dysc_sym',       # RNP~n_safe | Dyscalculic x symbolic
    ]}

    for chain, draw in product(n_chains, n_draws):
        sample = rnp.xs(chain, level='chain').xs(draw, level='draw').reset_index('n_safe')

        for key, (grp, fmt) in [
                            ('ctrl',        ('Control',      None)),
                            ('dysc',        ('Dyscalculic',  None)),
                            ('nonsym',      (None,           'non-symbolic')),
                            ('sym',         (None,           'symbolic')),
                            ('ctrl_nonsym', ('Control',      'non-symbolic')),
                            ('ctrl_sym',    ('Control',      'symbolic')),
                            ('dysc_nonsym', ('Dyscalculic',  'non-symbolic')),
                            ('dysc_sym',    ('Dyscalculic',  'symbolic')),
                            ]:
            slope, _ = fit_slope(sample, group=grp, fmt=fmt)
            slope_store[key].append(slope)
    # Convert lists to arrays
    slope_store = {k: np.array(v) for k, v in slope_store.items()}

    slope_effects = {
        'RNP:SS':              slope_store['ctrl'],                                                              # baseline SS slope
        'RNP:SS:group':        slope_store['dysc']     - slope_store['ctrl'],                                   # group modulation of SS slope
        'RNP:SS:format':       slope_store['sym']      - slope_store['nonsym'],                                 # format modulation of SS slope
        'RNP:SS:format:group': (slope_store['dysc_sym'] - slope_store['dysc_nonsym']) -                        # 3-way: group x format x SS
                               (slope_store['ctrl_sym'] - slope_store['ctrl_nonsym']),
    }

    for name, posterior in slope_effects.items():
        stats = summarize_posterior(posterior)
        rows.append({'model': model_name, 'effect': name, **stats})

    return pd.DataFrame(rows, columns=['model', 'effect', 'mean', 'ci_low', 'ci_high', 'p_gt_0'])


def save_rnp_effects(rnp, model_name, csv_path='rnp_effects.csv'):
    """
    Compute RNP effects and append to (or create) a CSV file.
    Existing rows for the same model_name are replaced.
    """
    df_new = compute_rnp_effects(rnp, model_name=model_name)

    try:
        df_existing = pd.read_csv(csv_path)
        df_existing = df_existing[df_existing['model'] != model_name]  # drop old rows for this model
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_out = df_new

    df_out.to_csv(csv_path, index=False)
    print(f"Saved {len(df_new)} effects for '{model_name}' → {csv_path}")
    return df_new


def save_posterior_effects(traces, model_name, csv_path='posterior_effects.csv'):
    """
    Extract all fixed-effect posterior parameters from traces and save to CSV.
    """
    rows = []
    posterior = traces.posterior

    for param in posterior.data_vars:
        samples = posterior[param].values.ravel()  # flatten chains x draws

        # skip random effects (usually 3D+: chain x draw x subject)
        if samples.size != posterior.dims['chain'] * posterior.dims['draw']:
            continue

        rows.append({
            'model':  model_name,
            'effect': param,
            **summarize_posterior(samples)
        })

    df_new = pd.DataFrame(rows, columns=['model', 'effect', 'mean', 'ci_low', 'ci_high', 'p_gt_0'])

    # append to / update CSV
    try:
        df_existing = pd.read_csv(csv_path)
        df_existing = df_existing[df_existing['model'] != model_name]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_out = df_new

    df_out.to_csv(csv_path, index=False)
    print(f"Saved {len(df_new)} parameters for '{model_name}' → {csv_path}")
    return df_new

import seaborn as sns

def plot_risky_bayerian_inference(mu_prior, std_prior,
                                    mu_n1, sd_n1, 
                                    mu_n2, sd_n2,
                                    mu_post_n1, sd_post_n1, 
                                    mu_post_n2, sd_post_n2,
                                    x = np.linspace(1.5, 5.5, 1000),
                                    palette = sns.color_palette('coolwarm', 4)#[::-1]
                                  ):
    import matplotlib.pyplot as plt
    import scipy.stats as ss

    sns.set_theme('paper', 'white', font='helvetica', font_scale=1.25, palette='tab10')

    def plot_dist(mu, sd, y=0.0,color=None, shade=True, **kwargs):
        plt.plot(x, y+ss.norm(loc=mu, scale=sd).pdf(x), color=color, **kwargs, alpha=.8)
        if shade:
            plt.fill_between(x,y,y+ ss.norm(loc=mu, scale=sd).pdf(x), alpha=0.3, color=color)
        sns.despine()

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))

    # likelihood and prior
    plot_dist(mu_prior, std_prior, color='black', label='Prior')
    plot_dist(mu_n1, sd_n1, color=palette[3], label='n1')
    plot_dist(mu_n2, sd_n2, color=palette[0], label='n2')
    plt.axis('off')

    # posterior beneath
    y = -2 # y-positions
    plot_dist(mu_post_n1, sd_post_n1, color=palette[3], label='n1', y=y)
    plot_dist(mu_post_n2, sd_post_n2, color=palette[0], label='n2', y=y)

    # likelihood - posterior connecting lines
    plt.plot([mu_n1, mu_post_n1], [0,y], color=palette[3], ls='--')
    plt.plot([mu_n2, mu_post_n2], [0,y], color=palette[0], ls='--')

    # annotations
    x_anot_offset = 0. # negative when likelihood larger than prior
    y_anot_offset = 1.4
    ax.annotate('Prior', (mu_prior, 0.65), ha='center', va='bottom')
    ax.annotate('safe', (mu_n1-x_anot_offset,y_anot_offset), ha='center', va='bottom', color=palette[3])
    ax.annotate('risky', (mu_n2-x_anot_offset, y_anot_offset), ha='center', va='bottom', color=palette[0])


    # arrow
    plt.annotate('', xytext=(mu_n1, 0), xy=(mu_post_n1, 0), arrowprops={"facecolor":palette[3], 'edgecolor':'black', "linewidth":1., 'shrink':0.05,'headlength':6}) #, 'headlength':6
    plt.annotate('', xytext=(mu_n2, 0), xy=(mu_post_n2, 0), arrowprops={"facecolor":palette[0], 'edgecolor':'black', "linewidth":1., 'shrink':0.05,'headlength':6}) #, 'headlength':6

    # gray baselines
    plt.axhline(0, color='grey', lw=2)
    plt.axhline(y, color='grey', lw=2)
    y2 = y-1.5
    plt.axhline(y2, color='grey', lw=2)

    ax.annotate(f'Payoff \nlikelihoods', (-2, 0.0+0.5), ha='left', va='center',fontsize=10)
    ax.annotate(f'Posteriors', (-2,y+0.5), ha='left', va='center',fontsize=10)
    ax.annotate('Expected \nvalue', (-2, y2+0.5), ha='left', va='center',fontsize=10)

    # last row 
    ax.annotate('* p=0.55', (mu_post_n2-0.2, -2.5),  ha='left', va='center', color=palette[0],fontsize=9)
    ax.annotate('* p=1.0', (mu_post_n1, -2.5),  ha='left', va='center', color=palette[3],fontsize=9)
    plt.plot([mu_post_n1,mu_post_n1 ], [y,y-1.5], color=palette[3], ls='-')
    plt.plot([mu_post_n2,mu_post_n2*0.55 ], [y,y-1.5], color=palette[0], ls='-')
    plt.scatter([mu_post_n1], [y2], color=palette[3], zorder=10)
    plt.scatter([mu_post_n2*0.55], [y2], color=palette[0], zorder=10)

    return fig, ax