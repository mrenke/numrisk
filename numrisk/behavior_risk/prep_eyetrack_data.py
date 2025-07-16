import re
from io import StringIO
import argparse
import os.path as op
import os
import subprocess
import gzip

import glob
import pandas as pd
import shutil


def extract_data(subject, bids_folder='/data'):
    subject = int(subject)
    dir = op.join(bids_folder, 'sourcedata/eyetracking_risk_renamed')

    fn = op.join(dir, f'sub-{subject:02d}.edf')
    asc_fn = fn.replace('.edf', '.asc')
    edf2acs_cmd = '/home/ubuntu/git/riskeye/edf2asc'

    # get gaze
    cmd = f'{edf2acs_cmd}  -t -y -z -v -s -vel {fn}'# outputs sample data only
    gaze_target_fn = fn.replace('.edf', '.gaz.gz')

    subprocess.run(cmd, shell=True)

    with open(asc_fn, 'rb') as asc_file, gzip.open(gaze_target_fn, 'wb') as target_file:
            target_file.writelines(asc_file)
    os.remove(asc_fn)

    # get messages
    cmd = f'{edf2acs_cmd}   -t -y -z -v -e {fn}' # outputs event data only
    msg_target_fn = fn.replace('.edf', '.msg.gz')

    subprocess.run(cmd, shell=True)

    with open(asc_fn, 'rb') as asc_file, gzip.open(msg_target_fn, 'wb') as target_file:
            target_file.writelines(asc_file)
    os.remove(asc_fn)

def get_experimental_messages(subject, bids_folder='/data'):
    subject = int(subject)
    dir = op.join(bids_folder, 'sourcedata/eyetracking_risk_renamed')
    
    msg_fn=op.join(dir, f'sub-{subject:02d}.msg.gz')   # op.join(pati, f'{file}.edf')

    with gzip.open(msg_fn, 'rt') as mfd:
        message_string = mfd.read()

    pattern = re.compile(r'MSG\t(?P<timestamp>[0-9]+)\tstart_type-(?P<start_type>.+?)_phase-(?P<phase>[0-9]+)(?:_key-(?P<key>[a-zA-Z]+))?(?:_time-(?P<time>[0-9.]+))?')
    message_strings = pattern.findall(message_string)
    tmp = pd.DataFrame(message_strings, columns=['timestamp', 'message', 'phase', 'key', 'time']).astype({'timestamp': int, 'phase': int})
    tmp['trial'] = tmp['message'].map(lambda x: re.match(r'(.+)_trial', x).group(1))
    tmp['type'] = tmp['message'].map(lambda x: 'response' if 'response' in x else 'stim')

    return tmp

def get_saccades(subject, bids_folder='/data'):
    subject = int(subject)
    dir = op.join(bids_folder, 'sourcedata/eyetracking_risk_renamed')
    msg_fn=op.join(dir, f'sub-{subject:02d}.msg.gz')
    with gzip.open(msg_fn, 'rt') as mfd:
        message_string = mfd.read()
    message_strings = re.findall(re.compile('ESACC\t(?P<info>.+)'), message_string)
    csvString = ('\n'.join(message_strings))
    saccades = pd.read_csv(StringIO(csvString), sep='\t', names=['eye', 'start_timestamp', 'end_timestamp', 'duration', 'start_x', 'start_y', 'end_x', 'end_y', 'amp', 'peak_velocity'], na_values=['.', '   .'],)
                    #    dtype={'start_y':float})
    saccades.index.name = 'n'
    return saccades

def main(subject, bids_folder='/data'):
    target_dir = op.join(bids_folder, 'derivatives', 'pupil', f'sub-{subject:02d}', 'func')
    
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    extract_data(subject, bids_folder)
    messages = get_experimental_messages(subject, bids_folder)
    saccades = get_saccades(subject, bids_folder)

    messages.to_csv(op.join(target_dir, f'sub-{subject:02d}_messages.tsv'), sep='\t')
    saccades.to_csv(op.join(target_dir, f'sub-{subject:02d}_saccades.tsv'), sep='\t')

    #dir = get_subject_folder(subject, root_folder) 
    #gaze_fn = op.join(dir, f'Rs{subject:02d}rn{run:02d}.gaz.gz')
    #shutil.move(gaze_fn, op.join(target_dir, f'sub-{subject:02d}_run-{run}_gaze.tsv.gz'))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('subject', type=int)
    argparser.add_argument('--bids_folder', default='/mnt_03/ds-dnumrisk')
    args = argparser.parse_args()
    main(args.subject, args.bids_folder)