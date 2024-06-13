
import argparse

from exptools2.core import Session # installed, Gilles version
from exptools2.core import Trial
from psychopy import logging
from psychopy.visual import TextStim, ImageStim

import numpy as np
import os
import os.path as op
import pandas as pd

from session import PileSession  # in folder
from trial import InstructionTrial # in folder
from trial_magJudge import MagJudgeTrial
from utils import get_output_dir_str
from make_trial_design import create_design_magJudge


class MagJudgeInstructTrial(MagJudgeTrial):

    def __init__(self, session, trial_nr, txt, bottom_txt=None, show_phase=0, keys=None, **kwargs):

        phase_durations = np.ones(6) * 1e-6
        phase_durations[show_phase] = np.inf
        self.keys = keys

        super().__init__(session, trial_nr, phase_durations=phase_durations, **kwargs)

        txt_height = self.session.settings['various'].get('text_height')
        txt_width = self.session.settings['various'].get('text_width')

        self.text = TextStim(session.win, txt,
                             pos=(0.0, 6.0), height=txt_height, wrapWidth=txt_width, color=(0, 1, 0))

        if bottom_txt is None:
            bottom_txt = "Drücken Sie eine Ihrer Tasten, um fortzufahren."

        self.text2 = TextStim(session.win, bottom_txt, pos=(
            0.0, -6.0), height=txt_height, wrapWidth=txt_width,
            color=(0, 1, 0))

    def get_events(self):

        events = Trial.get_events(self)

        if self.keys is None:
            if events:
                self.stop_phase()
        else:
            for key, t in events:
                if key in self.keys:
                    self.stop_phase()

    def draw(self):
        if self.phase != 0:
            self.session.fixation_lines.setColor((1, -1, -1))

        if self.phase < 9:
            super().draw()
        else:
            self.session.fixation_lines.draw()
            if self.phase == 10:
                self.choice_stim.draw()
            elif self.phase == 11:
                self.certainty_stim.draw()

        self.text.draw()
        self.text2.draw()


class InstructionSession(PileSession):

    Trial = MagJudgeTrial

    def __init__(self, output_str, subject=None, output_dir=None, settings_file=None):
        super().__init__(output_str, subject=subject,
                         output_dir=output_dir, settings_file=settings_file, eyetracker_on=False)

        #logging.warn(self.settings['run'])
        self.buttons = self.settings['various'].get('buttons')
        self.image2 = ImageStim(self.win,
                                       self.settings['pile'].get('image2'),
                                       texRes=32,
                                       size=self.settings['pile'].get('dot_radius'))

    def create_trials(self):
        self.trials = []

        txt = """
        Hi!

        Willkommen zu der Einführung des ersten Experimentes, welches Sie danach im Scanner machen werden.

        Zuersteinmal werden Sie immer 2 Tasten haben um Ihre Auswahl einzugeben. 

        Auf der Tastatur dieses Computers sind diese:
        * die J-Taset (rechter Zeigefinger)
        * die K-Taste (rechter Mittelfinger)

        TIPP: lassen Sie ihre Finger auf den Tasten liegen während des Experimentes. Bei diesem Computer hat die J-Taste eine kleine Erhöhung die man spüren kann.

        Im Scanner wird es auch 2 Tasten geben, die sie mit den gleichen Fingern Ihrer rechten Hand bedienen werden.

        Von nun an nennen wir sie Taste 1 (rechter Zeigefinger) und Taste 2 (rechter Mittelfinger)

        Drücke Taste 1 um fortzufahren.

        """

        self.trials.append(InstructionTrial(self, 1,
                                            txt=txt, keys=[self.buttons[0]]))

        txt = f"""
        Drücke Taste 2 um fortzufahren.
        """
        self.trials.append(InstructionTrial(self, 2,
                                            txt=txt, keys=[self.buttons[1]]))
        txt = """
        
        In diesem Experiment müssen Sie in jedem Versuch entscheiden, welche der zwei Stapel die sie nacheinander sehene, mehr Münzen enthält, also größer ist.
        
        Für jede korrekte Antwort bekommen sie einen kleinen Bonus, dessen Summe Ihnen am Ende zusätzlich gegeben wird.

        Falls Sie nicht antworten, erhalten Sie für diesen Versuch keinen Bonus.
        
        Drücke Taste 1 um fortzufahren.
        """

        self.trials.append(InstructionTrial(self, 3,
                                            txt=txt, keys=[self.buttons[0]]))

        txt = """

        Wir werden Sie nun durch alle Schritte des Versuches führen.

        Drücke Taste 1 um fortzufahren.
        """
        self.trials.append(InstructionTrial(self, 4,
                                            txt=txt, keys=[self.buttons[0]]))

        txt = """ Ein neuere Versuch startet immer mit einem grünen Fixierungs-Kreuz auf dem Bilschrim """

        self.trials.append(MagJudgeInstructTrial(
            self, 5, txt, num2=10, show_phase=0,   keys=[self.buttons[0], self.buttons[1]]))

        txt = """ Nach dem Fixierungs-Kreuz sehen Sie den ertsen Stapel Münzen 
                
                (Dessen Menge Sie sich so gut wie möglich merken sollten, jedoch nur als "Abschätzung", da später die Zeit nicht ausreicht um abzuzählen)
                
                """

        self.trials.append(MagJudgeInstructTrial(
            self, 6, txt, num1=10, show_phase=2, keys=[self.buttons[0], self.buttons[1]]))

        txt = """ Nun sehen Sie ein rotes Fixierungs-Kreuz für eine Weile """

        self.trials.append(MagJudgeInstructTrial(
            self, 7, txt, show_phase=3, keys=[self.buttons[0], self.buttons[1]]))

        txt = """ Dann sehen Sie den zweiten Stapel Münzen, dessen Menge Sie mit der des ersten Stapels des Versuches vergleichen müssen. 
        
        Nun sollten Sie direkt ihre Antwort abgeben.

        """

        bottom_txt = " Drücken Sie nun Taste 1 um anzugeben, dass die erste Menge größer war." 

        self.trials.append(MagJudgeInstructTrial(
            self, 8, txt, num2=5, show_phase=4, bottom_txt=bottom_txt, keys=[self.buttons[0]]))


        txt = "Nachdem Sie gewählt haben, wird Ihnen ihre Auswahl nochmal angezeigt"

        trial10 = MagJudgeInstructTrial(
            self, 16, txt, show_phase=5)
        trial10.choice = 1
        trial10.choice_time = 0.4
        trial10.choice_stim.text = f'{trial10.choice}'
        self.trials.append(trial10)

        txt = """
        Gut gemacht!!

        Sie werden nun 3 Probe-Versuche machen. 
        Diese werden automatisch voranschreiten. 

        Achtung: Wenn Sie nun den zweiten Stapel sehen, haben Sie auch nur begrenzt Zeit (2 sek.) um zu antworten .
        
        """
        self.trials.append(InstructionTrial(self, 11,
                                            txt=txt))

        frac = np.linspace(-6,6, 13)
        frac = np.delete(frac, 6)                     
        fractions = np.power(2,(frac/4))
        df = create_design_magJudge(fractions,  base = np.array([5, 7, 10, 14, 20]), repetitions=1, n_runs= 1)                             
        d = df.reset_index().loc[:2,:]
        for ix, row in d.iterrows():
                        self.trials.append(MagJudgeTrial(self, row.trial,
                                                        num1=int(row.n1),
                                                        num2=int(row.n2),
                                                        jitter1= 5, #row.jitter1,
                                                        jitter2= 2, #row.jitter2,
                                                        speedup = False))


        ### missing 10 example trials

        txt = f"""
        Super gemacht!

        Wir sind jetzt am Ende der Einführung.

        Nun sind Sie bereit das "offiziellen" Experiment im MRT-Scanner zu machen, in welchem ihre Antworten dann auch über die Bonus-Zahlung entscheiden.

        Falls irgendwas unklar ist, zögern Sie bitte nicht uns zu fragen!

        """

        self.trials.append(InstructionTrial(self, trial_nr=21, txt=txt))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default='instructee', nargs='?')
    parser.add_argument('--settings', default='instructions', nargs='?')

    cmd_args = parser.parse_args()

    subject, session, task, run = cmd_args.subject, 'instruction', 'instruction',  None
    output_dir, output_str = get_output_dir_str(subject, session, task, None ) #, run)

    log_file = op.join(output_dir, output_str + '_log.txt')
    logging.warn(f'Writing results to: {log_file}')

    settings_fn = op.join(op.dirname(__file__), 'settings',
                          f'{cmd_args.settings}.yml')

    session_object = InstructionSession(output_str=output_str,
                                        output_dir=output_dir,
                                        settings_file=settings_fn, 
                                        #eyetracker_on= False,
                                        subject=subject)

    session_object.create_trials()
    print(session_object.trials)
    logging.warn(
        f'Writing results to: {op.join(session_object.output_dir, session_object.output_str)}')
    session_object.run()
    session_object.close()