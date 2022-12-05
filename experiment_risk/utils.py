import os.path as op
import argparse
import numpy as np
import scipy.stats as ss
import pandas as pd
from psychopy import logging
from itertools import product
import yaml
from exptools2.core import Trial
from psychopy.visual import TextStim


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

def get_output_dir_str(subject, session, task):
    output_dir = op.join(op.dirname(__file__), 'logs', f'sub-{subject}')
    logging.warn(f'Writing results to  {output_dir}')

    if session:
        output_dir = op.join(output_dir, f'ses-{session}')
        output_str = f'sub-{subject}_ses-{session}_task-{task}'
    else:
        output_str = f'sub-{subject}_task-{task}'

    return output_dir, output_str


def run_break(session):
    self.session.win
    TextStim(self.session.win,
                                    text = f'{num2}',
                                    pos=pile2_pos,
                                    height=screen_adj_factor,
                                    wrapWidth=screen_adj_factor,
                                    color=(-1, -1, -1)
                                    )


class BreakPhase(Trial):
    def __init__(self, session, break_n, n_breaks, phase_durations=None, **kwargs):
        #phase_durations = [120]
        if phase_durations is None:
            max_break_time = 120
            phase_durations = [max_break_time]

        trial_nr = 999 # apparently i need this ?!

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        text = f"""
                you have done {break_n}/{n_breaks} of the experiment,        
                this is a short break.

                take as much time as you want,
                just press any button to continue.   
                """
        txt_height = self.session.settings['various'].get('text_height')
        txt_width = self.session.settings['various'].get('text_width')

        self.text = TextStim(self.session.win,
                                    text = text,
                                    height=txt_height, wrapWidth=txt_width,
                                    pos= (0,0),
                                    #height=0.5,
                                    #wrapWidth=0.5,
                                    color=(1, 1, 1)
                                    )
    def draw(self):
        if self.phase == 0:
            self.text.draw()

    def get_events(self):
        events = super().get_events() 

        if events:
            self.stop_phase()

    # run(self) defined in parent class       






