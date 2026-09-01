"""Registry of subject-wise measures used in cross-measure correlation overviews.

Each entry maps a measure name to a loader function returning a DataFrame
indexed by `subject`, plus a `category` tag (used for grouping/coloring in
plots). To add a new measure: write a small `_load_*` function and register
it below — no need to touch the analysis notebook that consumes the registry.
"""
import os.path as op

import numpy as np
import pandas as pd

PHENOTYPE_DIR = '/Users/mrenke/data/ds-dnumrisk/derivatives/phenotype'
ADD_TABLES_DIR = '/Users/mrenke/data/ds-dnumrisk/add_tables'

# Some sources provide several redundant columns for the same underlying
# construct (a raw score and a transform of it, or a sub-score that feeds
# into a composite). CONFIG picks the representative column(s) used by
# default; pass `columns=...` to the loader directly to get the others.
CONFIG = {
    'panamath_columns': ['weber_transformed'],  # ANS-acuity measure; drop raw weber_frac and percent_correct
    'vs_wm_columns': ['erfassungsspanne'],  # span score; drop overall_score and basisscore
    'magjudge_bauer_variant':'v4_rdm', #'v4_choice',  # see MAGJUDGE_BAUER_FILES for available variants
    # Default to the 3 DAN patches that showed group differences (control vs.
    # dyscalc) in poster_OHBM26.ipynb; pass columns=[...] for the others
    # ('L_temporal', 'R_frontal-lateral', 'R_temporal').
    'dan_patch_columns': ['L_frontal-lateral', 'L_parietal-lateral', 'R_parietal-lateral'],
    # corr_diff (stim2 - stim1) is the measure of interest — how much the
    # neural-activation/gradient-3 correlation shifts between the two magjudge
    # stimuli; corr_1/corr_2 (the per-stimulus correlations) are available via
    # columns=['grad3_corr_stim1', 'grad3_corr_stim2', 'grad3_corr_diff'].
    'grad3_betastim_columns': ['grad3_corr_diff', 'grad3_corr_stim1', 'grad3_corr_stim2'],
    }


def _load_participants(dir_=ADD_TABLES_DIR):
    df = pd.read_csv(op.join(dir_, 'subjects_recruit_scan_scanned-final.csv'), header=0)
    df = (df.loc[:, ['subject ID', 'age', 'group', 'gender']]
            .rename(columns={'subject ID': 'subject'})
            .dropna()
            .astype({'subject': int, 'group': int})
            .set_index('subject'))
    df['group'] = np.where(df['group'] == 0, 'control', 'dyscalc')
    return df


def _load_magjudge_probit(dir_=PHENOTYPE_DIR):
    df = pd.read_csv(op.join(dir_, 'magjudge_probit-2_all-subwise-params.csv')).set_index('subject') # was '_appropSample' before!
    df = df.rename(columns={'gamma': 'gamma_magjudge', 'intercept': 'intercept_magjudge'})
    df['intercept_magjudge_abs'] = df['intercept_magjudge'].abs()
    return df


# filename per fitted model variant — add an entry here for any new bauer fit
MAGJUDGE_BAUER_FILES = {
    'biased': 'magjudge_bauer-3_sds.csv',
    'unbiased': 'magjudge_bauer-3_sds_unbiased.csv',
    'v4_choice': 'magjudge_bauerV4-choice_pm_hn_sds.csv',
    'v4_rdm' : 'magjudge_bauerV4-rdm_full_cont_hn_sds.csv'
}


def _load_magjudge_bauer(variant, dir_=PHENOTYPE_DIR, suffix_columns=True):
    df = pd.read_csv(op.join(dir_, MAGJUDGE_BAUER_FILES[variant])).set_index('subject')
    df = df.drop(columns=['group'], errors='ignore')
    # same column names across variants — suffix to compare them side by side if needed
    return df.add_suffix(f'_{variant}') if suffix_columns else df

def _load_magjudge_bauer_default(dir_=PHENOTYPE_DIR, variant=None):
    variant = variant if variant is not None else CONFIG['magjudge_bauer_variant']
    return _load_magjudge_bauer(variant, dir_, suffix_columns=False)


def _load_math_stuff(dir_=PHENOTYPE_DIR):
    return pd.read_csv(op.join(dir_, 'math_skill&confidence&anxiety-means.csv')).set_index('subject')


def _load_iq_scores(dir_=PHENOTYPE_DIR):
    return pd.read_csv(op.join(dir_, 'iq-scores_ids2.csv')).set_index('subject')


def _load_vs_wm(dir_=PHENOTYPE_DIR, columns=None):
    df = pd.read_csv(op.join(dir_, 'visio-spatial-WM_CBTtask-params.csv')).set_index('subject')
    columns = columns if columns is not None else CONFIG['vs_wm_columns']
    return df[columns]


def _load_panamath(dir_=ADD_TABLES_DIR, columns=None):
    df = pd.read_csv(op.join(dir_, 'panamath_AllRunsSummary.csv'))
    df = df.rename(columns={'Subject ID': 'subject'}).set_index('subject')
    df = df[['Weber Fraction', 'Percent Correct']].rename(
        columns={'Weber Fraction': 'weber_frac', 'Percent Correct': 'panamath_percent_correct'})
    df = df.drop(index=999, errors='ignore').sort_index()
    df['weber_transformed'] = np.log(df['weber_frac'])
    columns = columns if columns is not None else CONFIG['panamath_columns']
    return df[columns]


def _load_decode_r(dir_=PHENOTYPE_DIR):
    df = pd.read_csv(op.join(dir_, 'decoding_r.csv')).set_index('subject')
    return df.rename(columns={'r': 'neural_numsense_precision'})


def _load_npc_dispersion(dir_=PHENOTYPE_DIR):
    return pd.read_csv(op.join(dir_, 'NPC_dispersion_2D_final.csv')).set_index('subject').drop(columns=['group'])


def _load_grad3_betastim(dir_=PHENOTYPE_DIR, columns=None):
    df = pd.read_csv(op.join(dir_, 'grad3_betaStim-1-2_corr.csv')).set_index('subject')
    df = df.rename(columns={'corr_1': 'grad3_corr_stim1', 'corr_2': 'grad3_corr_stim2',
                             'corr_diff': 'grad3_corr_diff'})
    columns = columns if columns is not None else CONFIG['grad3_betastim_columns']
    return df[columns]


def _load_npc_pfm_net_area(dir_=PHENOTYPE_DIR): 
    # from parietal_patterns/nets_PFM/npc_net_ana.ipynb
    df = pd.read_csv(op.join(dir_, 'netsPFM_indArea_NPC.csv'))
    df = df.set_index(['subject', 'network']).unstack('network')
    df.columns = [f'{col[1]}_{col[0]}' for col in df.columns]
    return df

def _load_risk_probit(dir_=PHENOTYPE_DIR):
    df = pd.read_csv(op.join(dir_, 'risk_probit_format-symbolic_subwise_summary.csv')).set_index('subject')
    df = df.rename(columns={'ind_point_map': 'risk_indPoint', 
                            'ind_point_ss_slope': 'risk_indPoint_SSshift',
                            'gamma_map': 'risk_gamma', 
                            'gamma_ss_slope': 'risk_gamma_SSshift'})
                            
    return df.drop(columns='group_label')

def _load_risk_bauer(dir_=PHENOTYPE_DIR):
    df = pd.read_csv(op.join(dir_, 'risk_bauer_model-powerLawEncoding4_regression_format-symbolic.csv'))
    return df.set_index('subject').rename(columns={'alpha': 'risk_alpha', 'n_asym_ev_sd': 'risk_evSD_asym'})[['risk_alpha', 'risk_evSD_asym']]


def _load_eyetrack_dur_diff(format_='non-symbolic', dir_=PHENOTYPE_DIR):
    df = pd.read_csv(op.join(dir_, f'subwise_duration_option_difference_abs_{format_}.tsv')).set_index('subject')
    return df

def _load_everydaylifeQ(dir_=PHENOTYPE_DIR):
    df = pd.read_csv(op.join(dir_, 'everyday_life_numSituations_PCs.csv')).set_index('subject')
    return df

def _load_dan_patches(dir_=PHENOTYPE_DIR, columns=None, area_measure='ind_area'):
    # from parietal_patterns/nets_PFM/npc_net_ana.ipynb
    df = pd.read_csv(op.join(dir_, 'netsPFM_DANpatches.csv'))
    df = df.rename(columns={'sub_id': 'subject'}).set_index(['subject', 'patch'])
    df = df[area_measure].unstack('patch') / 100  # convert to cm^2
    columns = columns if columns is not None else CONFIG['dan_patch_columns']
    return df[columns].add_prefix('DANpatch_')


REGISTRY = {
    'participants': {'loader': _load_participants, 'category': 'demographics'},
    'magjudge_probit': {'loader': _load_magjudge_probit, 'category': 'behavioral_magjudge'},
    'magjudge_bauer': {'loader': _load_magjudge_bauer_default, 'category': 'behavioral_magjudge'},
    'math_stuff': {'loader': _load_math_stuff, 'category': {
        'skill_score': 'behavioral_cognitive',       # test-based
        'anx_mean': 'behavioral_questionnaire',       # self-report
        'conf_mean': 'behavioral_questionnaire',      # self-report
        }},
    'everydaylife_questionnaire': {'loader': _load_everydaylifeQ, 'category': 'behavioral_questionnaire'},
    'iq_scores': {'loader': _load_iq_scores, 'category': 'behavioral_cognitive'},
    'vs_wm': {'loader': _load_vs_wm, 'category': 'behavioral_cognitive'},
    'panamath': {'loader': _load_panamath, 'category': 'behavioral_cognitive'},
    'decode_r': {'loader': _load_decode_r, 'category': 'neural_encoding'},
    'npc_dispersion': {'loader': _load_npc_dispersion, 'category': 'neural_connectivity'},
    'grad3_betastim': {'loader': _load_grad3_betastim, 'category': 'neural_gradient_activation'},
    'npc_pfm_net_area': {'loader': _load_npc_pfm_net_area, 'category': 'neural_connectivity'},
    'dan_patches': {'loader': _load_dan_patches, 'category': 'neural_connectivity'},
    'risk_probit': {'loader': _load_risk_probit, 'category': 'behavioral_risk'},
    'risk_bauer': {'loader': _load_risk_bauer, 'category': 'behavioral_risk'},
    'eyetrack_dur_diff': {'loader': _load_eyetrack_dur_diff, 'category': 'behavioral_risk'},
}

# Risk-task measures are excluded by default; pass include_categories=['behavioral_risk']
# (or explicit `names`) to bring them into a given overview.
DEFAULT_EXCLUDE_CATEGORIES = {'behavioral_risk'}


def load_all(names=None, exclude_categories=DEFAULT_EXCLUDE_CATEGORIES, include_categories=None):
    """Load and outer-join registered measures into one wide DataFrame indexed by subject.

    `category` on a registry entry can be a single string (applies to all of
    that loader's columns) or a dict mapping column name -> category, for
    sources that mix e.g. task performance and questionnaire columns.

    Returns (df, col_to_category), where col_to_category maps each resulting
    column name to its category.
    """
    names = names if names is not None else list(REGISTRY.keys())
    exclude_categories = exclude_categories or set()

    df = None
    col_to_category = {}
    for name in names:
        entry = REGISTRY[name]
        measure_df = entry['loader']()
        cat = entry['category']
        if isinstance(cat, dict):
            col_to_category.update(cat)
        else:
            col_to_category.update({col: cat for col in measure_df.columns})
        df = measure_df if df is None else df.join(measure_df, how='outer')

    keep_cols = [c for c in df.columns
                 if col_to_category.get(c) not in exclude_categories
                 and (include_categories is None or col_to_category.get(c) in include_categories)]
    df = df[keep_cols]
    col_to_category = {c: col_to_category[c] for c in keep_cols}
    return df, col_to_category
