import os.path as op
import pandas as pd
import numpy as np
#from riskeye.utils import get_header_length_csv
def get_header_length_csv(fn):
    return pd.read_csv(fn, delim_whitespace=True, nrows=0).shape[1]

def get_outliers():
    return ['58']

def remove_outliers(df):
    for outlier in get_outliers():
        df = df.drop(outlier, level='subject')

    return df

def get_all_subject_ids():
    return [f'{e:02d}' for e in range(1, 67)]

def get_all_subjects(bids_folder='/Users/mrenke/data/ds-dnumrisk'):
    return [Subject(e, bids_folder=bids_folder) for e in get_all_subject_ids()]

def get_all_behavior(include_no_responses=False, bids_folder='/Users/mrenke/data/ds-dnumrisk', exclude_outliers=True, include_eyedata=True, source='eyepos'):
    df = pd.concat([e.get_behavior(include_no_responses=include_no_responses) for e in get_all_subjects(bids_folder=bids_folder)])

    if exclude_outliers:
        df = remove_outliers(df)

    if include_eyedata:
        eyepos = get_all_eyepos_info(source=source, summarize=True, bids_folder=bids_folder)
        df = df.join(eyepos)
        df['seen_risky_first'] = True
        df['seen_risky_first']= df['seen_risky_first'].where(((df['first_saccade'] == 'left_option') & (df['p_left'] == 0.55)) | ((df['first_saccade'] == 'right_option') & (df['p_right'] == 0.55)) , False)
        df['seen_risky_first']= df['seen_risky_first'].where(~df['first_saccade'].isnull(), np.nan)

        df['fixation_duration'] = df['left_duration'] + df['right_duration']
        df['left_duration_prop'] = df['left_duration'] / df['fixation_duration']
        df['right_duration_prop'] = df['right_duration'] / df['fixation_duration']

        df['risky_duration'] = df['left_duration'].where(df['p_left'] == 0.55, df['right_duration'])
        df['safe_duration'] = df['left_duration'].where(df['p_left'] == 1.0, df['right_duration'])
        df['risky_duration_prop'] = df['risky_duration'] / df['fixation_duration']
        df['risky_duration_prop_split'] = df.groupby(['subject', 'exptype', 'n_safe', 'n_risky'], group_keys=False)['risky_duration_prop'].apply(lambda x: (x > np.nanmedian(x)).map({True:'high', False:'low'}) if ((len(x) > 1) and (not x.isnull().any())) else pd.Series([np.nan]*len(x), index=x.index))

        df['risky_duration_split'] = df.groupby(['subject', 'exptype', 'n_safe', 'n_risky'], group_keys=False)['risky_duration'].apply(lambda x: (x > np.nanmedian(x)).map({True:'high', False:'low'}) if ((len(x) > 1) and (not x.isnull().any())) else pd.Series([np.nan]*len(x), index=x.index))
        df['safe_duration_split'] = df.groupby(['subject', 'exptype', 'n_safe', 'n_risky'], group_keys=False)['safe_duration'].apply(lambda x: (x > np.nanmedian(x)).map({True:'high', False:'low'}) if ((len(x) > 1) and (not x.isnull().any())) else pd.Series([np.nan]*len(x), index=x.index))
        df['total_duration_split'] = df.groupby(['subject', 'exptype', 'n_safe', 'n_risky'], group_keys=False)['fixation_duration'].apply(lambda x: (x > np.nanmedian(x)).map({True:'high', False:'low'}) if ((len(x) > 1) and (not x.isnull().any())) else pd.Series([np.nan]*len(x), index=x.index))

    return df

def get_all_eyepos_info(source='eyepos', summarize=True, only_leftright=True, bids_folder='/Users/mrenke/data/ds-dnumrisk', exclude_outliers=True):
    if summarize:
        df =  pd.read_csv(op.join(bids_folder, 'derivatives', 'pupil', f'group_source-{source}_fixations_summary.tsv'),
                        index_col=['subject',    'trial', 'n_saccades'], sep='\t', dtype={'subject':str})

    else:
        df = pd.read_csv(op.join(bids_folder, 'derivatives', 'pupil', f'group_source-{source}_fixations.tsv'),
                        index_col=['subject',    'trial', 'n'], sep='\t', dtype={'subject':str})

    if exclude_outliers:
        df = remove_outliers(df) 

    return df


class Subject(object):


    def __init__(self, subject_id, bids_folder='/Users/mrenke/data/ds-dnumrisk'):

        if type(subject_id) is int:
            subject_id = f'{subject_id:02d}'

        self.subject_id = subject_id
        self.bids_folder = bids_folder


    def get_behavior(self, include_no_responses=False, format='symbolic'):

        df = pd.read_csv(op.join(self.bids_folder, f'sub-{self.subject_id}', 'ses-1','func',
                                    f'sub-{self.subject_id}_ses-1_task-risk_{format}_events.tsv'), sep='\t',)
                                    #index_col=['subject', 'trial'], dtype={'subject':str})
        df['trial'] = df['trial_nr'].astype(int) 
        df['subject'] = int(self.subject_id)   

        if not include_no_responses:
            df = df[~df['chose_risky'].isnull()]

        def get_risk_bin(d, n_risk_bins=6):
            try: 
                return pd.qcut(d, n_risk_bins, range(1, n_risk_bins+1))
            except Exception as e:
                n = len(d)
                ix = np.linspace(1, n_risk_bins+1, n, False)

                d[d.sort_values().index] = np.floor(ix)
                
                return d
        df['bin(risky/safe)'] = df.groupby(['subject'], group_keys=False)['log(risky/safe)'].apply(get_risk_bin)
        df['Experiment'] = df.index.get_level_values('exptype').map({'symbolic':'Symbols', 'non-symbolic':'Coin clouds'})

        return df

    def get_saccades_raw(self):
        saccades = pd.read_csv(op.join(self.bids_folder, 'derivatives', 'pupil', f'sub-{self.subject_id}', 'func', f'sub-{self.subject_id}_saccades.tsv'),
                           sep='\t')

        saccades['subject'] = self.subject_id
        return saccades.set_index(['subject', 'start_timestamp']).sort_index()

    def get_saccades(self, merge_saccades=True):
        saccades = self.get_saccades_raw()
        messages = self.get_eyetracker_timings().unstack('type')
        
        saccades_per_block = []

        for ix, row in messages.iterrows():
            s = saccades.loc[(ix[0], slice(row.stim, row.response)), :] #  Selects saccades for the current trial, between the times stimulus onset and response.
            saccades_per_block.append(s.droplevel([0]))

        saccades_per_block = pd.concat(saccades_per_block, keys=messages.index)

        def get_durations(df):
            durations =  df.iloc[1:]['end_timestamp'].values - df.iloc[:-1]['end_timestamp']
            durations.index = df.iloc[:-1].index

            # durations = durations.append(df.iloc[-1]['end_timestamp'], messages)
            last_trial = df.iloc[-1]
            durations = pd.concat((durations, pd.Series([messages.loc[last_trial.name[:2], 'response'] - last_trial['end_timestamp']], index=df.iloc[-1:].index)))
            return durations

        saccades_per_block['fixation_duration'] = saccades_per_block.groupby(['subject',    'trial'], group_keys=False).apply(get_durations)
        saccades_per_block = saccades_per_block[saccades_per_block.fixation_duration > 0.0]


        def get_fixation_targets(df):

            #median = df['end_x'].median()
            #bins = [-np.inf, median-350, median-100, median-75, median+75, median+100, median+350, np.inf]
            median = 1000 # df['end_x'].median()
            dist_to_median_out= 750
            dist_to_median = 250
            bins = [-np.inf, median-dist_to_median_out, median-dist_to_median, median-75, median+75, median+dist_to_median, median+dist_to_median_out, np.inf]

            return pd.cut(df['end_x'], bins=bins, labels=['outside_left', 'left_option', 'center_left', 'fixation', 'center_right', 'right_option', 'outside_right'])

        saccades_per_block['fixation_target'] = get_fixation_targets(saccades_per_block)

        if merge_saccades:

            def merge_fixations(d):
                d['previous_fixation_target'] = d['fixation_target'].shift(1)

                result = []

                n = 1
                duration = 0.0

                for ix, row in d.iterrows():
                    if (row.fixation_target == row['previous_fixation_target']) or (pd.isnull(row.previous_fixation_target)):
                        duration += row.fixation_duration
                    else:
                        result.append({'duration':duration, 'fixation_target':row.previous_fixation_target})
                        duration = row.fixation_duration
                        n += 1

                if duration != 0:
                    result.append({'duration':duration, 'fixation_target':row.fixation_target})

                return pd.DataFrame(result, index=pd.Index(np.arange(1, n+1), name='n'))

            saccades_per_block = saccades_per_block.groupby(['subject',    'trial']).apply(merge_fixations)
            saccades_per_block.index = saccades_per_block.index.set_levels(
                        saccades_per_block.index.levels[saccades_per_block.index.names.index('trial')].astype(int),
                        level='trial'
                    )
        return saccades_per_block.sort_index()

    def get_trialwise_saccade_info(self, source='saccades', summarize_trials=True):

        assert(source in ['saccades', 'eyepos'])

        def summarize_trial_saccades(d):
            d = d[np.in1d(d.fixation_target, ['left_option', 'right_option'])]

            durations = d.groupby('fixation_target')['duration'].sum() 
            # print(durations)

            if len(d) == 0:
                result = pd.DataFrame([{'n_saccades':0}])
            else:
                result = pd.DataFrame([{'n_saccades':len(d), 'first_saccade':d.iloc[0]['fixation_target'], 'last_saccade':d.iloc[-1]['fixation_target']}])
                result['left_duration'] = durations.loc['left_option'] if 'left_option' in durations.index else 0
                result['right_duration'] = durations.loc['right_option'] if 'right_option' in durations.index else 0

            return result

        if source == 'saccades':
            saccades = self.get_saccades()
        elif source == 'eyepos':
            eyepos = self.get_eyeposition(trialwise=True)
            eyepos['n'] = eyepos.groupby(['subject',    'trial'], group_keys=False)['fixation_target'].apply(lambda x: x.ne(x.shift(1)).cumsum())
            saccades = eyepos.groupby(['subject',    'trial', 'n', 'fixation_target']).apply(lambda d: d.iloc[-1].name[-1] - d.iloc[0].name[-1])
            saccades = (saccades.dt.total_seconds() * 1000.).to_frame('duration').reset_index(['fixation_target', 'n'])
            saccades = saccades[saccades.duration > 30.]

        if summarize_trials:
            return saccades.groupby(['subject',    'trial']).apply(summarize_trial_saccades).droplevel(-1)
        else:
            return saccades

    def get_eyetracker_timings(self):
        import re
        messages = pd.read_csv(op.join(self.bids_folder, 'derivatives', 'pupil', f'sub-{self.subject_id}', 'func', f'sub-{self.subject_id}_messages.tsv'),
                           sep='\t')
        messages['trial'] = messages['message'].map(lambda x: re.search(r'trial-(\d+)', x).group(1)) # was wrong before in prep_eyetrack_data.py
        messages['subject'] = self.subject_id

        messages = messages.set_index(['subject', 'trial', 'type'])['timestamp']
        messages = messages[~messages.index.duplicated()]
        return messages


    def get_eyeposition(self, trialwise=True): # needs runs... 

        fn = op.join(self.bids_folder, 'derivatives', 'pupil', f'sub-{self.subject_id}', 'func', f'sub-{self.subject_id}_gaze.tsv.gz')
        n_cols = get_header_length_csv(fn)
        if n_cols == 11:
            cols = ['time', 'L_gaze_X', 'L_gaze_Y', 'L_pupil_size', 'L_vel_X', 'L_vel_Y', 'null', 'target_x', 'target_y', 'target_distance', 'null2']
        elif n_cols == 12:
            cols = ['time', 'L_gaze_X', 'L_gaze_Y', 'L_pupil_size', 'R_gaze_X', 'R_gaze_Y', 'R_pupil_size', 'L_vel_X', 'L_vel_Y', 'R_vel_X', 'R_vel_Y', 'null2']
        elif n_cols == 7:
            cols = ['time', 'L_gaze_X', 'L_gaze_Y', 'L_pupil_size', 'L_vel_X', 'L_vel_Y', 'null']
        eyepos =pd.read_csv(fn,
                        delim_whitespace=True, index_col=0,
                        names=cols, na_values='.',
                        usecols=['time', 'L_gaze_X', 'L_gaze_Y'])
        eyepos.fillna(method='ffill')
        eyepos.index = pd.TimedeltaIndex(eyepos.index, unit='ms')
        eyepos = pd.DataFrame(eyepos)

        # define fixation targets
        median = 1000 # df['end_x'].median()
        dist_to_median_out= 750
        dist_to_median = 250
        #bins = [-np.inf, median-350, median-100, median-75, median+75, median+100, median+350, np.inf]
        bins = [-np.inf, median-dist_to_median_out, median-dist_to_median, median-75, median+75, median+dist_to_median, median+dist_to_median_out, np.inf]
        eyepos['fixation_target']  = pd.cut(eyepos['L_gaze_X'], bins=bins,
                                            labels=['outside_left', 'left_option', 'center_left', 'fixation', 'center_right', 'right_option', 'outside_right'])

        if trialwise:
            messages = self.get_eyetracker_timings()
            messages = pd.to_timedelta(messages, unit='ms').unstack('type')

            # non-responses take 4500 ms
            messages.loc[messages['response'].isnull(), 'response'] = messages.loc[messages['response'].isnull()] + pd.to_timedelta(4500, 'ms')

            trialwise_eyepos = []
            for ix, row in messages.iterrows():
                if pd.notnull(row.stim) and pd.notnull(row.response):
                    e = eyepos.loc[slice(row.stim, row.response)]
                    trialwise_eyepos.append(e)
                else:
                # handle or skip this case
                    continue
            trialwise_eyepos = pd.concat(trialwise_eyepos, keys=messages.index)
            trialwise_eyepos = trialwise_eyepos.reset_index()
            trialwise_eyepos['trial'] = trialwise_eyepos['trial'].astype(int)
            trialwise_eyepos = trialwise_eyepos.set_index(['subject','trial','time'])
            trialwise_eyepos = trialwise_eyepos[trialwise_eyepos.index.get_level_values('trial') != 0] # remove trial 0, which is not a real trial
            return trialwise_eyepos.sort_index()

        return eyepos