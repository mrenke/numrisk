from exptools2.core import Session
from psychopy import visual, logging
import pandas as pd
import os.path as op
from psychopy import visual, logging
from trial import GambleTrial


class RiskPileSession(Session):
    """ Simple session with x trials. """
    def __init__(self, output_str, subject=None, output_dir=None, settings_file=None, run=None):
        """ Initializes TestSession object. """
        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file)

        self.image1 = visual.ImageStim(self.win,
                                       self.settings['pile'].get('image1'),
                                       texRes=32,
                                       size=self.settings['pile'].get('dot_radius'))

        self.fixation_cross = visual.TextStim(self.win, '+', color=(1., 1., 1.))

    def create_trials(self):
        self.trials = [GambleTrial(self, trial_nr=1, num1=5, num2=10, prob1=1, prob2=.55)]

    def run(self):
        self.start_experiment()

        for trial in self.trials:
            trial.run()

        self.close()