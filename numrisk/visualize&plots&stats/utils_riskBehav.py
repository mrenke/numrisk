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