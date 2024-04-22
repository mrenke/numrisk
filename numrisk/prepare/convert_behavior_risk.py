import os
import os.path as op
import argparse
import pandas as pd
import numpy as np

def main(subject, session, bids_folder, max_rt=1.0):

    sourcedata = op.join(bids_folder, 'sourcedata')

    target_dir = op.join(bids_folder, f'sub-{subject}', f'ses-{session}', 'func')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    for format in ['non-symbolic', 'symbolic']:
        print(subject, session, format)

        behavior = pd.read_table(op.join(sourcedata, f'behavior_risk/sub-{subject}/ses-{session}/sub-{subject}_ses-{session}_task-risk_{format}_events.tsv'))
        behavior['trial_nr'] = behavior['trial_nr'].astype(int)

        print(behavior)

        #pulses = behavior[behavior.event_type == 'pulse'][['trial_nr', 'onset']] # I think only for when scanned simultanously

        stim = behavior[(behavior['event_type'] == 'stim') & (behavior['phase'] == 0)].copy()
        stim['trial_type'] = 'stimulus'

        choice = behavior[(behavior['event_type'] == 'choice')].copy()
        choice['trial_type'] = 'choice'

        events = pd.concat((stim, choice)).sort_index().reset_index(drop=True)
        # result['choice'] = result['choice'].astype(int)
        events = events[['trial_nr', 'onset', 'trial_type', 'prob1', 'prob2', 'n1', 'n2', 'choice']]

        fn = op.join(target_dir, f'sub-{subject}_ses-{session}_task-risk_{format}_events.tsv')
        print(fn)
        events.to_csv(fn, index=False, sep='\t')


def get_hazard(x, s=1.0, loc=0.0, scale=10, cut=30, use_cut=False):
    import scipy.stats as ss
    
    x = x / .7

    dist = ss.lognorm(s, loc, scale)
    
    if use_cut:
        sf = lambda x: 1 - (dist.cdf(x) / dist.cdf(cut))
    else:
        sf = dist.sf

    return np.clip(dist.pdf(x) / sf(x), 0, np.inf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--session', default=1)
    parser.add_argument('--bids_folder', default='/Users/mrenke/data/ds-dnumrisk')
    args = parser.parse_args()

    main(args.subject, args.session, args.bids_folder)
