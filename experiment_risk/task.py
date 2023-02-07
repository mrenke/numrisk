# gamble task outside the scanner

import argparse
from session import RiskPileSession
from utils import get_output_dir_str
import yaml
from psychopy import logging
import os.path as op

def run_experiment(session_cls, task, subject=None, session=None, settings='default', n_breaks = 4,  *args, **kwargs):

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=subject, nargs='?')
    parser.add_argument('session', default=session, nargs='?')
    parser.add_argument('--settings', default=settings, nargs='?')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--format', default='non-symbolic')
    parser.add_argument('--eyetracker', default='on')

    cmd_args = parser.parse_args()
    subject, session, settings, format, eyetracker = cmd_args.subject, cmd_args.session, cmd_args.settings, cmd_args.format, cmd_args.eyetracker


    if subject is None:
        subject = input('Subject? (999): ')
        subject = 999 if subject == '' else subject

    if session is None:
        session = input('Session? (1): ')
        session = 1 if session == '' else session

    settings_fn = op.join(op.dirname(__file__), 'settings',
                       f'{settings}.yml')

    with open(settings_fn, 'r') as f_in:
        settings_ = yaml.safe_load(f_in)

    if 'eyetracker' in settings_.keys() and eyetracker == 'on':
        eyetracker_on = True
        logging.warn("Using eyetracker")
    else:
        eyetracker_on = False
        logging.warn("Using NO eyetracker")

    logging.warn(f'Using {settings_fn} as settings')

    output_dir, output_str = get_output_dir_str(subject, session, task, format)

    log_file = op.join(output_dir, output_str + '_log.txt')
    logging.warn(f'Writing results to: {log_file}')

    if (not cmd_args.overwrite) and op.exists(log_file):
        overwrite = input(
            f'{log_file} already exists! Are you sure you want to continue? ')
        if overwrite != 'y':
            raise Exception('Run cancelled: file already exists') 
            
    session_object = session_cls(output_str=output_str,
                            output_dir=output_dir,
                            settings_file=settings_fn, subject=subject,
                            eyetracker_on=eyetracker_on, 
                            format = format,
                            n_breaks = n_breaks,
                            *args, **kwargs)
    
    # run 
    session_object.create_trials()
    logging.warn(f'Writing results to: {op.join(session_object.output_dir, session_object.output_str)}')
    session_object.run()
    session_object.close()

    return session

if __name__ == '__main__':
    
    session_cls = RiskPileSession
    task = 'risk'

    run_experiment(session_cls, task=task)


# make_design is called in: session --> class: RiskPileSession -->def:  create_design --> makeDesign
