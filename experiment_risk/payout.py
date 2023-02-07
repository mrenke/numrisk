import glob
import argparse
import pandas as pd
import os.path as op
import numpy as np
import re
from text import TextSession
from utils import get_output_dir_str
import numpy as np


def get_payout(subject, session, settings='macbook'):

    task = 'payout'

    logs_calibrate = glob.glob(op.abspath(op.join(
        'logs', f'sub-{subject}', f'ses-{session}', f'sub-{subject}_ses-{session}_task-calibration_run-*_events.tsv')))
    print(op.abspath(op.join('logs', f'sub-{subject}', f'ses-{session}',
                             f'sub-{subject}_ses-{session}_task-calibrate_run-*_events.tsv')))
    print(logs_calibrate)
    logs_task = glob.glob(op.abspath(op.join(
        'logs', f'sub-{subject}', f'ses-{session}', f'sub-{subject}_ses-{session}_task-risk_non-symbolic_events.tsv')))
    print(logs_task)

    reg = re.compile(
        '.*sub-(?P<subject>.+)_ses-(?P<session>.+)_task-(?P<task>.+)_events\.tsv')
    df = []

    for l in (logs_calibrate + logs_task):
        d = pd.read_table(l)
        for key, value in reg.match(l).groupdict().items():
            d[key] = value

        df.append(d)

    df = pd.concat(df)
    df = df.pivot_table(index=['task', 'trial_nr'], values=[
                        'choice',  'n1', 'n2', 'prob1', 'prob2'])

    row = df.sample().iloc[0]

    txt = 'Der Computer hat zufällig einen der Versuche ausgesucht' 
    'you made in this session\n\n'

    if np.isnan(row.choice):
        txt += f'In dem ausgewählt Versuch, hast du KEINE Antwort gegeben. ' \
            '\n\nDas bedeutet, du bekommst keinen Bonus'
        payout = 0
    else:
        txt += f'\n\nDu hast zwischen mit {int(np.round(row.prob1*100))}% Wahrscheinlichkeit ' \
            f' {int(row.n1)} CHF zu gewinnen, oder mit {int(np.round(row.prob2*100))}% Wahrscheinlichkeit ' \
            f' {int(row.n2)} CHF zu gewinnen gewählt.'

        if ((row.choice == 1) and (row['prob1'] == 1)):
            txt += f'\n\nDu hattest die sichere Option gewählt und erhältst nun {int(row.n1)} CHF Bonus'
            payout = row.n1

        if ((row.choice == 2) and (row.prob2 == 1)):
            txt += f'\n\nDu hattest die sichere Option gewählt und erhältst nun {int(row.n2)} CHF Bonus'
            payout = row.n2

        if ((row.choice == 2) and (row.prob1 == 1)):
            txt += '\n\nDu hattest die Option mit Risiko gewählt. '
            die = np.random.randint(100) + 1
            txt += f'Der digitale Würfel zeigt {die}. '
            if die > 55:
                txt += '\n\nDaher bekommst du leider keinen Bonus.'
                payout = 0
            else:
                txt += f'\n\nDaher bekommst du einen {int(row.n2)} CHF Bonus. '
                payout = row.n2

        if ((row.choice == 1) and (row.prob2 == 1)):
            txt += '\n\nDu hattest die Option mit Risiko gewählt. '
            die = np.random.randint(100) + 1
            txt += f'Der digitale Würfel zeigt {die}. '
            if die > 55:
                txt += f'\n\nDaher bekommst du leider keinen Bonus.'
                payout = 0
            else:
                txt += f'\n\nDaher bekommst du einen  {int(row.n1)} CHF Bonus.'
                payout = row.n1

    output_dir, output_str = get_output_dir_str(subject, session, task, None)

    settings_fn = op.join(op.dirname(__file__), 'settings',
                          f'{settings}.yml')

    payout_session = TextSession(txt=txt,
                                 output_str='payout',
                                 output_dir=output_dir,
                                 settings_file=settings_fn)

    payout_session.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None, nargs='?')
    parser.add_argument('session', default=None, nargs='?')
    parser.add_argument('--settings', default='default', nargs='?')

    args = parser.parse_args()

    get_payout(subject=args.subject, session=args.session,
               settings=args.settings)
