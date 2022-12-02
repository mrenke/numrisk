from exptools2.core import Session
from exptools2.core import Trial
from psychopy.visual import Pie, TextStim
from utils import run_experiment, create_stimulus_array_log_df
from stimuli import _create_stimulus_array, CertaintyStimulus, ProbabilityPieChart
import numpy as np
import os.path as op
import pandas as pd
from trial import InstructionTrial


class MagJudgeTrial(Trial):
    def __init__(self, session, trial_nr, phase_durations=None,
                 num1=10, num2=5,
                 jitter1=2.5, jitter2=4.0,
                 speedup = False, 
                 **kwargs):

        if phase_durations is None:
            # 0 1 2 3 4 5 6 7 8 9 --> delete 2 & 6 (piecharts)
            #phase_durations = [.25, .3, .3, .5, .6, jitter1, .3, .3, .6, jitter2]
            phase_durations = [.25, .75,  .6, jitter1, .6, jitter2]
        elif len(phase_durations) == 12:
            phase_durations = phase_durations
        else:
            raise Exception(
                "Don't directly set phase_durations for GambleTrial!")

        if speedup:
            phase_durations = [.25, .75,  .6, 1, .6, 2]
            
        super().__init__(session, trial_nr, phase_durations, **kwargs)

        self.parameters['n1'] = num1
        self.parameters['n2'] = num2

        self.buttons = self.session.settings['various'].get('buttons')

        # self.text1 = TextStim('Pie1: {int(

        self.pile1 = _create_stimulus_array(self.session.win,
                                            num1,
                                            self.session.settings['pile'].get(
                                                'aperture_radius'),
                                            self.session.settings['pile'].get(
                                                'dot_radius'),
                                            image=self.session.image1)

        self.pile2 = _create_stimulus_array(self.session.win,
                                            num2,
                                            self.session.settings['pile'].get(
                                                'aperture_radius'),
                                            self.session.settings['pile'].get(
                                                'dot_radius'),
                                            image=self.session.image1)
        self.stimulus_arrays = [self.pile1, self.pile2]

        self.choice_stim = TextStim(self.session.win)
        button_size = self.session.settings['various'].get('button_size')

        self.choice = None

        self.last_key_responses = dict(zip(self.buttons + [self.session.mri_trigger], [0.0] * 5))

    def draw(self):

        self.session.fixation_lines.draw()

        if self.phase == 0:
            self.session.fixation_lines.setColor((-1, 1, -1))
        elif self.phase == 1:
            self.session.fixation_lines.setColor((1, -1, -1))
        elif self.phase == 2:
            self.pile1.draw()
        elif self.phase == 4:
            self.pile2.draw()

        if self.phase == 5:
            if self.choice is not None:
                if (self.session.clock.getTime() - self.choice_time) < .5:
                    self.choice_stim.draw()

    def get_events(self):
        events = super().get_events()

        for key, t in events:
            if key not in self.last_key_responses:
                self.last_key_responses[key] = t - 0.6

            if t - self.last_key_responses[key] > 0.5:
                if self.phase > 3: ## !!!!
                    if self.choice is None:
                        if key in [self.buttons[0], self.buttons[1]]:
                            self.choice_time = self.session.clock.getTime()
                            if key == self.buttons[0]:
                                self.choice = 1
                            elif key == self.buttons[1]:
                                self.choice = 2
                            self.choice_stim.text = f'{self.choice}'

                            self.log(choice=self.choice)

            self.last_key_responses[key] = t

        return events

    def get_stimulus_array_log(self):

        n_dots1 = self.parameters['n1']
        n_dots2 = self.parameters['n2']

        # trial_ix, stim_array, stimulus_ix
        trial_ix = np.ones(n_dots1 + n_dots2) * self.trial_nr
        array_ix = [1] * n_dots1 + [2] * n_dots2
        stim_ix = np.hstack((np.arange(n_dots1) + 1, np.arange(n_dots2)+1))

        index = pd.MultiIndex.from_arrays([trial_ix, array_ix, stim_ix],
                                          names=('trial_nr', 'array_nr', 'stim_nr'))

        log = create_stimulus_array_log_df(self.stimulus_arrays, index=index)

        return log

    def log(self, choice=None):

        if (choice is not None):
            onset = self.session.clock.getTime()
            idx = self.session.global_log.shape[0]
            self.session.global_log.loc[idx, 'trial_nr'] = self.trial_nr
            self.session.global_log.loc[idx, 'onset'] = onset
            self.session.global_log.loc[idx, 'phase'] = self.phase
            self.session.global_log.loc[idx,
                                        'nr_frames'] = self.session.nr_frames

        if choice is not None:
            self.session.global_log.loc[idx, 'event_type'] = 'choice'
            self.session.global_log.loc[idx, 'choice'] = choice