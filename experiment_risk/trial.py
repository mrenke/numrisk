from exptools2.core import Trial
from stimuli import _create_stimulus_array, ProbabilityPieChart
from psychopy.visual import TextStim, Circle
import numpy as np
from utils import create_stimulus_array_log_df
import pandas as pd


class GambleTrial(Trial):
    def __init__(self, session, trial_nr, phase_durations=None, format = 'non-symbolic',
                 prob1=0.55, prob2=1.0, num1=10, num2=5,**kwargs):

        if phase_durations is None:
            phase_durations = [3., 1] # stimuli presentation, show response, break between trials?!

        super().__init__(session, trial_nr, phase_durations, **kwargs)

        self.trial_nr = trial_nr
        
        self.parameters['prob1'] = prob1
        self.parameters['prob2'] = prob2

        self.parameters['n1'] = num1
        self.parameters['n2'] = num2   

        
        self.buttons = self.session.settings['various'].get('buttons')
        piechart_width = self.session.settings['various'].get('piechart_width')
        
        text_width = 1.0
        center_l = -6
        center_r = 6
        piechart_pos1 = center_l, 5
        piechart_pos2 = center_r, 5

        self.piechart1 = ProbabilityPieChart(
            self.session.win, prob1, pos=piechart_pos1, size= piechart_width)
        self.piechart2 = ProbabilityPieChart(
            self.session.win, prob2, pos=piechart_pos2, size= piechart_width)

        y_pos = -1
        pile1_pos = center_l, y_pos
        pile2_pos = center_r, y_pos

        screen_adj_factor = 2
        circle_radius = self.session.settings['pile'].get('aperture_radius')
        circle_radius = circle_radius/screen_adj_factor
        dot_radius = self.session.settings['pile'].get('dot_radius')
        dot_radius = dot_radius/2

        if format == 'non-symbolic':


            self.pile1 = _create_stimulus_array(self.session.win,
                                                num1,
                                                circle_radius,
                                                dot_radius,
                                                pos=pile1_pos,
                                                image=self.session.image1)

            self.pile2 = _create_stimulus_array(self.session.win,
                                                num2,
                                                circle_radius,
                                                dot_radius,
                                                pos=pile2_pos,
                                                image=self.session.image1)
        elif format == 'symbolic':
            screen_adj_factor = screen_adj_factor / 2
            self.pile1 = TextStim(self.session.win,
                                    text = f'{num1}',
                                    pos=pile1_pos,
                                    height=screen_adj_factor,
                                    wrapWidth=screen_adj_factor,
                                    color=(-1, -1, -1)
                                    )
            self.pile2 = TextStim(self.session.win,
                                    text = f'{num2}',
                                    pos=pile2_pos,
                                    height=screen_adj_factor,
                                    wrapWidth=screen_adj_factor,
                                    color=(-1, -1, -1)
                                    )


        self.circle1 = Circle(self.session.win, radius = circle_radius + 2, pos = pile1_pos, lineColor= 'white' , fillColor = None, lineWidth = 4 )
        self.circle2 = Circle(self.session.win, radius = circle_radius + 2, pos = pile2_pos, lineColor= 'white' , fillColor = None, lineWidth = 4 )

        self.fixation_cross = TextStim(self.session.win, '+', color=(1., 1., 1.), pos = (0,y_pos))

        self.stimulus_arrays = [self.pile1, self.pile2]

        self.choice_stim = TextStim(self.session.win)
        button_size = self.session.settings['various'].get('button_size')

        self.choice = None

        self.last_key_responses = dict(zip(self.buttons + [self.session.mri_trigger], [0.0] * 5))        

    def draw(self):
        if self.phase == 0:
            self.fixation_cross.draw()
            self.piechart1.draw()
            self.piechart2.draw()
            self.pile1.draw()
            self.pile2.draw()
            self.circle1.draw()
            self.circle2.draw()
        elif self.phase == 1: # show them their reponse in phase 1
             if self.choice is not None:
                #if (self.session.clock.getTime() - self.choice_time) < .5:
                self.choice_stim.draw()           

    def get_events(self):
        events = super().get_events()

        for key, t in events:
            if key not in self.last_key_responses:
                self.last_key_responses[key] = t - 0.6

            if t - self.last_key_responses[key] > 0.5:
                if self.phase == 0: # reponse during phase 0
                    if self.choice is None:
                        if key in [self.buttons[0], self.buttons[1]]:
                            self.choice_time = self.session.clock.getTime()
                            if key == self.buttons[0]:
                                self.choice = 1
                            elif key == self.buttons[1]:
                                self.choice = 2
                            self.choice_stim.text = f'You chose pile {self.choice}'

                            self.log(choice=self.choice)
                            self.stop_phase()


            self.last_key_responses[key] = t

        return events

    def get_stimulus_array_log(self): # used in session.py

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

