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

def fit_slope(sample, col='rnp', group=None, fmt=None):
    """Fit a line of <col> ~ n_safe for a given group/format subset.

    Parameters
    ----------
    sample : DataFrame with columns including 'n_safe', <col>, and optionally
             'group' / 'format' as columns (after reset_index).
    col    : column to use as dependent variable ('rnp' or 'ind_point')
    group  : group label to filter on (None = no filter)
    fmt    : format string to filter on (None = no filter)

    Returns (slope, intercept).
    """
    tmp = sample.copy()
    if group is not None:
        tmp = tmp[tmp['group'] == group] if 'group' in tmp.columns else tmp.xs(group, level='group')
    if fmt is not None:
        tmp = tmp[tmp['format'] == fmt] if 'format' in tmp.columns else tmp.xs(fmt, level='format')
    A = np.vstack([tmp['n_safe'], np.ones(len(tmp))]).T
    slope, intercept = np.linalg.lstsq(A, tmp[col], rcond=None)[0]
    return slope, intercept


def compute_indpoint_effects(ind_point, model_name='model'):
    """
    Compute indifference-point-related posterior effects and return as a tidy DataFrame.

    Parameters
    ----------
    ind_point : DataFrame with column 'ind_point', indexed by at least
                (chain, draw, subject, group, n_safe).
                May optionally include a 'format' index level (both-format models).
    model_name : str

    Returns
    -------
    pd.DataFrame with columns: model, effect, mean, ci_low, ci_high, p_gt_0

    Effects always reported
    -----------------------
    ind_point             — overall mean indifference point
    ind_point:group       — Dyscalculic minus Control
    ind_point:SS          — slope of ind_point ~ n_safe (Control baseline)
    ind_point:SS:group    — group modulation of that slope (Dysc − Ctrl)

    Additional effects when 'format' index level is present
    -------------------------------------------------------
    ind_point:format          — non-symbolic minus symbolic
    ind_point:SS:format       — format modulation of SS slope
    ind_point:SS:format:group — 3-way interaction
    """
    # Work only on the numeric column — drop helpers like 'group_label' added by callers
    ind_point = ind_point[['ind_point']]

    n_chains   = ind_point.index.unique('chain').values
    n_draws    = ind_point.index.unique('draw').values
    has_format = 'format' in ind_point.index.names

    # Resolve group labels: support both numeric (0/1) and string ('Control'/'Dyscalculic')
    group_vals = ind_point.index.unique('group').tolist()
    if 0 in group_vals or 1 in group_vals:
        grp_ctrl, grp_dysc = 0, 1
    else:
        grp_ctrl, grp_dysc = 'Control', 'Dyscalculic'

    # ── 1. Simple contrasts ──────────────────────────────────────────────── #
    rows = []

    simple_effects = {'ind_point': ind_point}
    if has_format:
        simple_effects['ind_point:format'] = (
            ind_point.xs('non-symbolic', level='format')
            - ind_point.xs('symbolic',   level='format')
        )
    simple_effects['ind_point:group'] = (
        ind_point.xs(grp_dysc, level='group')
        - ind_point.xs(grp_ctrl, level='group')
    )

    for name, posterior in simple_effects.items():
        stats = summarize_posterior(posterior['ind_point'].values.ravel())
        rows.append({'model': model_name, 'effect': name, **stats})

    # ── 2. Slope-based effects (iterate over chain × draw) ───────────────── #
    slope_keys = ['ctrl', 'dysc']
    if has_format:
        slope_keys += ['nonsym', 'sym', 'ctrl_nonsym', 'ctrl_sym', 'dysc_nonsym', 'dysc_sym']
    slope_store = {k: [] for k in slope_keys}

    # which (grp, fmt) pairs to fit — format pairs only when level exists
    pairs = [
        ('ctrl', (grp_ctrl, None)),
        ('dysc', (grp_dysc, None)),
    ]
    if has_format:
        pairs += [
            ('nonsym',      (None,       'non-symbolic')),
            ('sym',         (None,       'symbolic')),
            ('ctrl_nonsym', (grp_ctrl,   'non-symbolic')),
            ('ctrl_sym',    (grp_ctrl,   'symbolic')),
            ('dysc_nonsym', (grp_dysc,   'non-symbolic')),
            ('dysc_sym',    (grp_dysc,   'symbolic')),
        ]

    for chain, draw in product(n_chains, n_draws):
        sample = (ind_point
                  .xs(chain, level='chain')
                  .xs(draw,  level='draw')
                  .reset_index())
        for key, (grp, fmt) in pairs:
            slope, _ = fit_slope(sample, col='ind_point', group=grp, fmt=fmt)
            slope_store[key].append(slope)

    slope_store = {k: np.array(v) for k, v in slope_store.items()}

    slope_effects = {
        'ind_point:SS':       slope_store['ctrl'],
        'ind_point:SS:group': slope_store['dysc'] - slope_store['ctrl'],
    }
    if has_format:
        slope_effects['ind_point:SS:format'] = (
            slope_store['sym'] - slope_store['nonsym']
        )
        slope_effects['ind_point:SS:format:group'] = (
            (slope_store['dysc_sym'] - slope_store['dysc_nonsym'])
            - (slope_store['ctrl_sym'] - slope_store['ctrl_nonsym'])
        )

    for name, posterior in slope_effects.items():
        stats = summarize_posterior(posterior)
        rows.append({'model': model_name, 'effect': name, **stats})

    return pd.DataFrame(rows, columns=['model', 'effect', 'mean', 'ci_low', 'ci_high', 'p_gt_0'])


def save_indpoint_effects(ind_point, model_name, csv_path='indpoint_effects.csv'):
    """
    Compute indifference-point effects and append to (or create) a CSV file.
    Existing rows for the same model_name are replaced.
    """
    df_new = compute_indpoint_effects(ind_point, model_name=model_name)

    try:
        df_existing = pd.read_csv(csv_path)
        df_existing = df_existing[df_existing['model'] != model_name]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_out = df_new

    df_out.to_csv(csv_path, index=False)
    print(f"Saved {len(df_new)} effects for '{model_name}' → {csv_path}")
    return df_new


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