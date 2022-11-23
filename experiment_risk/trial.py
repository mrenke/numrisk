from exptools2.core import Trial
from stimuli import _create_stimulus_array, CertaintyStimulus, ProbabilityPieChart
from psychopy.visual import TextStim
import numpy as np


class GambleTrial(Trial):
    def __init__(self, session, trial_nr, phase_durations=None,
                 prob1=0.55, prob2=1.0, num1=10, num2=5,**kwargs):

        if phase_durations is None:
            phase_durations = [10.]

        # print(f'**phase_durations**: {phase_durations}')
        super().__init__(session, trial_nr, phase_durations, **kwargs)

        self.parameters['prob1'] = prob1
        self.parameters['prob2'] = prob2

        self.parameters['n1'] = num1
        self.parameters['n2'] = num2   

        text_width = 1.0

        piechart_width = self.session.settings['various'].get('piechart_width')

        piechart_pos1 = -2, 5
        piechart_pos2 = 2, 5

        self.piechart1 = ProbabilityPieChart(
            self.session.win, prob1, pos=piechart_pos1, size= piechart_width)
        self.piechart2 = ProbabilityPieChart(
            self.session.win, prob2, pos=piechart_pos2, size= piechart_width)


        pile1_pos = -2, 0
        pile2_pos = 2, 0
        self.pile1 = _create_stimulus_array(self.session.win,
                                            num1,
                                            self.session.settings['pile'].get(
                                                'aperture_radius'),
                                            self.session.settings['pile'].get(
                                                'dot_radius'),
                                                pos=pile1_pos,
                                            image=self.session.image1)

        self.pile2 = _create_stimulus_array(self.session.win,
                                            num2,
                                            self.session.settings['pile'].get(
                                                'aperture_radius'),
                                            self.session.settings['pile'].get(
                                                'dot_radius'),
                                                pos=pile2_pos,
                                            image=self.session.image1)



    def draw(self):
        self.session.fixation_cross.draw()
        self.piechart1.draw()
        self.piechart2.draw()
        self.pile1.draw()
        self.pile2.draw()


    def get_events(self):

        events = super().get_events()

        if events:
            for key, t in events:
                print(f'{key} - {t}')
                if key == 'p':
                    self.stop_phase()
                if key == 'c':
                    print(self.session.fixation_cross.color)
                    if (self.session.fixation_cross.color == np.array([1., 1., 1.])).all():
                        self.session.fixation_cross.color = (-1. ,-1., -1.)
                    else:
                        self.session.fixation_cross.color = (1. ,1., 1.)



