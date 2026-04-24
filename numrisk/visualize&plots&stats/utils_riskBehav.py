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

def plot_risky_bayInf_diffDist(mu_prior, std_prior,
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

    y0 = 0
    y1 = -1.5
    y2 = -3
    y3 = -5

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))

    # y0: likelihood and prior 
    plot_dist(mu_prior, std_prior, color='black', label='Prior')
    plot_dist(mu_n1, sd_n1, color=palette[3], label='n1')
    plot_dist(mu_n2, sd_n2, color=palette[0], label='n2')
    plt.axis('off')
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
    plt.axhline(y1, color='grey', lw=2)
    plt.axhline(y2, color='grey', lw=2)
    ax.annotate(f'Payoff \nlikelihoods', (-2, 0.0+0.5), ha='left', va='center',fontsize=10)
    ax.annotate(f'Posteriors', (-2,y1+0.5), ha='left', va='center',fontsize=10)
    ax.annotate('Expected \nvalue', (-2, y2+0.5), ha='left', va='center',fontsize=10)


    # y0-y1:likelihood - posterior connecting lines
    plt.plot([mu_n1, mu_post_n1], [0,y1], color=palette[3], ls='--')
    plt.plot([mu_n2, mu_post_n2], [0,y1], color=palette[0], ls='--')

    # y1: posterior beneath
    plot_dist(mu_post_n1, sd_post_n1, color=palette[3], label='n1', y=y1)
    plot_dist(mu_post_n2, sd_post_n2, color=palette[0], label='n2', y=y1)


    # y1-y2: connecting
    #ax.annotate('* p=0.55', (mu_post_n2-0.2, -2.5),  ha='left', va='center', color=palette[0],fontsize=9)
    #ax.annotate('* p=1.0', (mu_post_n1, -2.5),  ha='left', va='center', color=palette[3],fontsize=9)
    #plt.plot([mu_post_n1,mu_post_n1 ], [y1,y1-1.5], color=palette[3], ls='-')
    #plt.plot([mu_post_n2,mu_post_n2*0.55 ], [y1,y1-1.5], color=palette[0], ls='-')


    # y2: posteriors in log-EV space
    mu_post_n2 = mu_post_n2 * 0.55 # R_RNP
    plot_dist(mu_post_n1, sd_post_n1, color=palette[3], label='n1', y=y2)
    plot_dist(mu_post_n2, sd_post_n2, color=palette[0], label='n2', y=y2)
    # overlay the overlap region
    pdf1 = ss.norm(loc=mu_post_n1, scale=sd_post_n1).pdf(x)
    pdf2 = ss.norm(loc=mu_post_n2, scale=sd_post_n2).pdf(x)
    plt.fill_between(x, y2, y2 + np.minimum(pdf1, pdf2), alpha=0.5, color='purple')
    # 
    plt.plot([mu_post_n1,mu_post_n1 ], [y2-0.2,y2+1.], color=palette[3], ls='--')
    plt.plot([mu_post_n2,mu_post_n2 ], [y2-0.2,y2+1.], color=palette[0], ls='--')

    # y3: difference distribution in log-EV space with risk-neutral reference point
    decision_reference = np.mean(x) # 
    diff_mu = (mu_post_n2 - mu_post_n1) + decision_reference# different scale for illustrative purposes
    diff_sd = np.sqrt(sd_post_n1**2 + sd_post_n2**2)
    # plot centrally 
    #plot_dist(diff_mu+decision_reference, diff_sd, color='grey', label='n2 - n1', y=y3)
    zoom_in_factor = 5
    pdf_diff = ss.norm(loc=diff_mu, scale=diff_sd).pdf(x)
    plt.plot(x, y3+pdf_diff, color='grey', alpha=.8)
    plt.fill_between(x, y3, y3 + pdf_diff,
                 where=(x >= decision_reference),
                 alpha=0.8, color='purple')  # n1 color
    plt.fill_between(x, y3, y3 + pdf_diff,
                 where=(x <= decision_reference),
                 alpha=0.8, color='yellow')  # n1 color
    plt.plot([decision_reference,decision_reference ], [y3,y3+1.5], color='grey', ls='-')
    plt.plot([diff_mu,diff_mu], [y3,y3+1.5], color='purple', ls='--')

    # gray baselines
    [plt.axhline(y, color='grey', lw=2) for y in [y0, y1, y2,y3]]
    ax.set_ylim(-6, 1.5)

    return fig, ax


def get_posterior(mu1, sd1, mu2, sd2):
    var1, var2 = sd1**2, sd2**2
    return mu1 + (var1/(var1+var2))*(mu2 - mu1), np.sqrt((var1*var2)/(var1+var2))

from scipy import stats as ss

def plot_probit_curves(ax, s_c_small, s_c_large, sigma_c, sigma_x, mu_p, sigma_p):
    """
    Psychometric (probit) curves: P(choose risky) vs log(x/c).

    The decision signal is:
        delta = mu_post_x - mu_post_c + log(p_risky)
    The log(p_risky) term shifts the RNP rightward (risk-neutral point > 0),
    because the risky option must compensate for p < 1.

    Symmetric noise (KLW / H1):
        delta = (1-w)*log_ratio + log(p)
        RNP   = -log(p) / (1-w)  =  log(1/0.55) / (1-w)
        → RNP shifts RIGHT as sigma increases (larger w → smaller 1-w)
        → slope (gamma) decreases with sigma
        → small/large stake curves still OVERLAP (no stake-size interaction)

    Asymmetric noise (H2, sigma_x > sigma_c):
        delta gains an extra term (w_c - w_x)*(s_c - mu_p) that flips sign
        across mu_p  →  curves SEPARATE for small vs large stakes
    """
    P_RISKY = 0.55    # probability of the risky outcome


    log_ratios = np.linspace(-1, 2.5, 300)
    log_p      = np.log(P_RISKY)          # ≈ -0.598

    # Posterior SDs depend only on sigma (not the mean)
    _, sd_pc = get_posterior(0, sigma_c, mu_p, sigma_p)
    _, sd_px = get_posterior(0, sigma_x, mu_p, sigma_p)
    sigma_delta = np.sqrt(sd_pc**2 + sd_px**2)

    # Shrinkage weights
    w_c = sigma_c**2 / (sigma_c**2 + sigma_p**2)
    w_x = sigma_x**2 / (sigma_x**2 + sigma_p**2)

    for s_c, ls, lbl, color in [(s_c_small, '-',  'Small stakes',  "#6B91C3" ),
                          (s_c_large, '--', 'Large stakes',  "#12396B" )]:
        mu_pc   = (1 - w_c) * s_c + w_c * mu_p
        delta_0 = (1 - w_x) * s_c + w_x * mu_p - mu_pc   # stake-size offset
        delta   = (1 - w_x) * log_ratios + delta_0 + log_p  # include p=0.55
        p_risky = ss.norm.cdf(delta / sigma_delta)

        ax.plot(log_ratios, p_risky, color=color, ls=ls, lw=3.0, label=lbl)

    ax.axhline(0.5, color='gray', ls=':', lw=0.8, alpha=0.7)
    ax.axvline(0.0, color='gray', ls=':', lw=0.8, alpha=0.7)
    ax.set_xlabel('log(risky / safe)', fontsize=9)
    ax.set_ylabel('P(choose risky)', fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-1, 2.5)
    ax.set_yticks([0, 0.5, 1])
    #ax.set_xticks([-1, 0, 1])
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc='upper left')
    sns.despine(ax=ax)