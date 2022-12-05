from exptools2.core import Session
from psychopy import visual, logging
import pandas as pd
import os.path as op
from psychopy import visual, logging
from trial import GambleTrial, InstructionTrial
from make_design import makeDesign
from utils import BreakPhase
import numpy as np


class RiskPileSession(Session):
    """ Simple session with x trials. """
    def __init__(self, output_str, subject=None, output_dir=None, settings_file=None, n_breaks = 4, format = 'non-symbolic'):
        """ Initializes TestSession object. """
        super().__init__(output_str, output_dir=output_dir, settings_file=settings_file)
        self.subject = subject
        self.image1 = visual.ImageStim(self.win,
                                       self.settings['pile'].get('image1'),
                                       texRes=32,
                                       size=self.settings['pile'].get('dot_radius'))

        self.format = format
        self.n_breaks = n_breaks

    def create_trials(self):

        makeDesign(self.subject) 
        
        task_settings_folder = op.abspath(op.join('settings', 'task'))
        fn = op.abspath(op.join(task_settings_folder,
                                f'sub-{self.subject}_ses_task-risk.tsv'))
                                #'sub-gilles_ses-task.tsv'))

        settings = pd.read_table(fn)
        self.trials = []
        
        for ix, row in settings.iterrows():
            self.trials.append(GambleTrial(self, row.trial, # self = session !
                                            prob1=row.p1, prob2=row.p2,
                                            num1=int(row.n1),
                                            num2=int(row.n2),
                                            format = self.format,
                                            ))
        #self.trials = [GambleTrial(self, trial_nr=1, num1=5, num2=10, prob1=1, prob2=.55)]

    def run(self):
        self.start_experiment()
        s = TaskInstructionTrial(self, trial_nr=0)
        s.run()

        break_n = 0
        n_breaks = self.n_breaks
        for trial in self.trials:
            trial.run()
            if (trial.trial_nr+1 % (len(self.trials)/n_breaks)) == 0 : 
                # len(self.trials)/self.n_breaks:
                break_n += 1
                b = BreakPhase(self,break_n,n_breaks)
                b.run()
                


        self.close()
    
    def close(self):
        super().close()

        array_log = []

        for trial in self.trials:
            if isinstance(trial, self.Trial):
                array_log.append(trial.get_stimulus_array_log())

        if len(array_log) > 0:
            array_log = pd.concat(array_log)

            array_log.to_csv(op.join(
                self.output_dir, self.output_str + '_stimarray_locations.tsv'), sep='\t')


class TaskInstructionTrial(InstructionTrial):
    
    def __init__(self, session, trial_nr, txt=None, phase_durations=[np.inf],
                 **kwargs):

        if txt is None:
            txt = f"""

            In this task, you will see two piles of Swiss Franc coins next to each other.
            Both piles are combined with a pie chart.
            The part of the pie chart that is lightly colored indicates
            the probability of a lottery you will gain the amount of
            Swiss Francs represented by the pile.
            Your task is to either select the left or the right lottery, by using your index (pile 1) or middle (pile 2) finger.

            NOTE: if you are to late in responding, or you do not 
            respond, you will gain no money for that trial. 

            There will be some breaks in between for which you can take as much time as you want.

            Press any of your buttons to continue.

            """

        super().__init__(session=session, trial_nr=trial_nr, phase_durations=phase_durations, txt=txt, **kwargs)
