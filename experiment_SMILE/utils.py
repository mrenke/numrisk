import os.path as op
import argparse
import numpy as np
import scipy.stats as ss
import pandas as pd
from psychopy import logging
from itertools import product
import yaml

def run_experiment(session_cls, task, use_runs=True, subject=None, session=None, run=None, settings='default', n_runs=6, *args, **kwargs):

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=subject, nargs='?')
    parser.add_argument('session', default=session, nargs='?')
    parser.add_argument('run', default=run, nargs='?', type=int)
    parser.add_argument('--settings', default=settings, nargs='?')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--speedup', action='store_true')
    cmd_args = parser.parse_args()
    subject, session, run, settings = cmd_args.subject, cmd_args.session, cmd_args.run, cmd_args.settings

    print(f'session reveiced: {session}')
    if subject is None:
        subject = input('Subject? (999): ')
        subject = 999 if subject == '' else subject

    if session is None:
        session = input('Session? (1): ')
        session = 1 if session == '' else session

    if use_runs and (run is None):
        run = input('Run? (None): ')
        run = None if run == '' else run
    elif run == '0':
        run = None

    settings_fn = op.join(op.dirname(__file__), 'settings',
                       f'{settings}.yml')

    with open(settings_fn, 'r') as f_in:
        settings_ = yaml.safe_load(f_in)

    if 'eyetracker' in settings_.keys():
        eyetracker_on = True
        logging.warn("Using eyetracker")
    else:
        eyetracker_on = False
        logging.warn("Using NO eyetracker")

    logging.warn(f'Using {settings_fn} as settings')


    if run is None:
        runs = range(1, n_runs + 1)
    else:
        runs = [run] 

    for run in runs:
        output_dir, output_str = get_output_dir_str(subject, session, task, run)

        log_file = op.join(output_dir, output_str + '_log.txt')
        logging.warn(f'Writing results to: {log_file}')

        if (not cmd_args.overwrite) and op.exists(log_file):
            overwrite = input(
                f'{log_file} already exists! Are you sure you want to continue? ')
            if overwrite != 'y':
                raise Exception('Run cancelled: file already exists')
        #if (not cmd_args.
        session_object = session_cls(output_str=output_str,
                              output_dir=output_dir,
                              settings_file=settings_fn, subject=subject,session = session,
                              run=run,
                              eyetracker_on=eyetracker_on,
                              speedup = cmd_args.speedup, # stores as defualt False if not given??
                               *args, **kwargs)
        session_object.create_trials()
        logging.warn(f'Writing results to: {op.join(session_object.output_dir, session_object.output_str)}')
        session_object.run()
        session_object.close()

    return session


def sample_isis(n, s=1.0, loc=0.0, scale=10, cut=30):

    d = np.zeros(n, dtype=int)
    changes = ss.lognorm(s, loc, scale).rvs(n)
    changes = changes[changes < cut]

    ix = np.cumsum(changes).astype(int)
    ix = ix[ix < len(d)]
    d[ix] = 1

    return d


def create_stimulus_array_log_df(stimulus_arrays, index=None):

    stimuli = [pd.DataFrame(sa.xys, columns=['x', 'y'],
                            index=pd.Index(np.arange(1, len(sa.xys)+1), name='stimulus')) for sa in stimulus_arrays]

    stimuli = pd.concat(stimuli, ignore_index=True)

    if index is not None:
        stimuli.index = index

    return stimuli

def get_output_dir_str(subject, session, task, run):
    output_dir = op.join(op.dirname(__file__), 'logs', f'sub-{subject}')
    logging.warn(f'Writing results to  {output_dir}')

    if session:
        output_dir = op.join(output_dir, f'ses-{session}')
        output_str = f'sub-{subject}_ses-{session}_task-{task}'
    else:
        output_str = f'sub-{subject}_task-{task}'

    if run:
        output_str += f'_run-{run}'

    return output_dir, output_str



