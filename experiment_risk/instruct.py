from session import RiskPileSession
#from gamble import IntroBlockTrial, GambleTrial
from utils import get_output_dir_str #, create_design
from make_design import create_design
from psychopy.visual import TextStim, ImageStim
from psychopy import logging
import os.path as op
import argparse
from trial import InstructionTrial, TaskInstructionTrial, GambleTrial
import numpy as np
from exptools2.core import Trial

class GambleInstructTrial(GambleTrial):

    def __init__(self, session, trial_nr, txt, bottom_txt=None, show_phase=0, keys=None, format = 'non-symbolic', **kwargs):

        phase_durations = np.ones(2) * 1e-6
        phase_durations[show_phase] = np.inf
        self.keys = keys

        super().__init__(session, trial_nr, phase_durations=phase_durations, format = format, **kwargs)

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

        super().draw()

        self.text.draw()
        self.text2.draw()


class InstructionSession(RiskPileSession):

    Trial = GambleTrial

    def __init__(self, output_str, subject=None, output_dir=None, settings_file=None):
        super().__init__(output_str, subject=subject,
                         output_dir=output_dir, settings_file=settings_file) #, eyetracker_on=False)

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

        Willkommen zu der Einführung des Experimentes. Hier erklären wir wie jeder einzelne Versuch abläuft.

        Zuersteinmal werden Sie immer 2 Tasten haben um Ihre Auswahl abzugeben

        Auf diesem Computer sind diese:
        * die J-Taset (linker Zeigefinger)
        * die K-Taste (rechter Zeigefinger)

        TIPP: lassen Sie ihre Finger auf den Tasten liegen während des Experimentes. Bei diesem Computer hat die J-Taste eine kleine Erhöhung die man spüren kann.

        Von nun an nennen wir sie Taste 1 (linker Zeigefinger) und Taste 2 rechter Zeigefinger)

        Drücke Taste 1 um fortzufahren.

        """

        self.trials.append(InstructionTrial(self, 1,
                                            txt=txt, keys=[self.buttons[0]]))

        txt = f"""
        Drücke Taste 2 um fortzufahren.
        """
        self.trials.append(InstructionTrial(self, 1,
                                            txt=txt, keys=[self.buttons[1]]))
        txt = """
        
        In diesem Experiment müssen Sie in jedem Versuch wählen zwischen: 
        * (i) eine gewissse Menge Geld sicher zu bekommen 
        oder 
        * (ii) an einer Lotterie teilzunehmen, bei der Sie mit 55% Wahrscheinlichkeit zu Gewinnen eine erheblich größere Menge Geld bekommen.

        Während dem Verlauf des Experimentes werden Sie viele Entscheidungen machen. 
        Nach dem Experiment werden wir zufällig einen dieser Versuche wählen. Wenn Sie in diesem Versuch die 55%- Option gewählt hatten, werden wir eine digitale Lotterie durchführen die bestimmt ob Sie den angebotenen Betrag bekommen.

        Drücke Taste 1 um fortzufahren.
        """

        self.trials.append(InstructionTrial(self, 1,
                                            txt=txt, keys=[self.buttons[0]]))

        txt = """
        Bitte notiern Sie: für den Fall dass Sie während eines Versuches nicht in der gegebenen Zeit antworten und dieser dann am Ende zufällig ausgewählt wird, erhalten sie 0 CHF.

        Drücke Taste 1 um fortzufahren.
        """

        self.trials.append(InstructionTrial(self, 1,
                                            txt=txt, keys=[self.buttons[0]]))

        txt = """

        Ein Beispiel: 
        Nehmen wir and während einem Versuch mussten Sie zwischen (1) 5 CHF mit 55% und (2) 1 CHF mit 100% zu gewinnen wählen. Sie haben nicht rechtzeitig geantwortet. Nun wird dieser Versuch für ihre Bonusbezahlung am Ende ausgewählt. Wieviel Geld bekommen Sie?

        1. Ich habe eine 55% Wahrscheinlichlkeit 5 CHF zu bekommen.
        2. Ich gewinne 0 CHF.

        Drücke die Taste die zu der korrekten Antwort gehört.

        """

        self.trials.append(InstructionTrial(self, 7,
                                            txt=txt, keys=[self.buttons[1]]))

        txt = """
        Wir werden Sie nun durch alle Schritte des Versuches führen.

        Grundsätzlich werden die Beträge in Schweizer Franken (CHF) die Sie gewinnen können ind er ersten Hälfte des Expeimentes als Stapel von Münzen und in der zweiten als Zahlen angezeigt.
        Es ändert sich aber nur das Anzeigeformat, sonst bleibt alles gleich.

        Nun zuerst ein Beispiel in dem der Betrag als Stapel von Münzen angezeigt wird.

        Drücke Taste 1 um fortzufahren.
        """
        self.trials.append(InstructionTrial(self, 1,
                                            txt=txt, keys=[self.buttons[0]]))

        txt = """
        Nun ist es ihre Zeit eine der beiden Optionen (Stapel) zu wählen
        Sie können Taste 1 für die erste und Taste 2 für die zweite Option drücken.
        """

        bottom_txt = " Drücken Sie Taste 2 um die zweite Option zu wählen" 

        self.trials.append(GambleInstructTrial(
            self, 15, txt, num2=10, show_phase=0, bottom_txt=bottom_txt, keys=[self.buttons[1]]))

        txt = "Nachdem Sie gewählt haben, wird Ihnen ihre Auswahl nochmal angezeigt"

        trial16 = GambleInstructTrial(
            self, 16, txt, show_phase=1)
        trial16.choice = 2
        trial16.choice_stim.text = f'Sie haben Option {trial16.choice} gewählt'
        self.trials.append(trial16)

        txt = """
        
        Wir gehen nun erneut durch einen Versuch (diesmal aber im Zahlen-Anzeigeformat !) und werden Ihnen am Ende Fragen dazu stellen, also passen Sie bitte auf!

        Drücken Sie Taste 2 um fortzufahren.

        """

        self.trials.append(InstructionTrial(self, 1,
                                            txt=txt, keys=[self.buttons[1]]))

        txt = 'Wählen Sie die erste Option'
        bottom_txt = '(drücken Sie Taste 1)'
        self.trials.append(GambleInstructTrial(
            self, 28, txt, show_phase=0, prob1=1., num1=2, prob2=.55, num2=4, bottom_txt=bottom_txt, keys=self.buttons[0], format = 'symbolic'))

        txt = ''
        trial29 = GambleInstructTrial(
            self, 29, txt, show_phase=1)
        trial29.choice = 1
        trial29.choice_stim.text = f'Sie haben Option {trial29.choice} gewählt'
        self.trials.append(trial29)

        txt = """
        Wählen Sie dir korrekte Antwortmöglichkeit:

        1. Ich  musste zwischen sicheren 4 CHF und einer Lotterie, 4 CHF mit 55 % Wahrscheinlichkeit zu gewinnen, wählen.

        2. Ich  musste zwischen sicheren 2 CHF und einer Lotterie, 4 CHF mit 55 % Wahrscheinlichkeit zu gewinnen, wählen.

       Drücken Sie die Taste die zu der korrekten Antwort gehört.

        """

        self.trials.append(InstructionTrial(self, 32,
                                            txt=txt, keys=[self.buttons[1]]))


        txt = """
        Gut gemacht!!

        Sie werden nun 10 Probe-Versuche machen. Diese werden automatisch voranschreiten.

        """

        self.trials.append(InstructionTrial(self, 33,
                                            txt=txt))



        frac = np.linspace(1,8,8)                     
        fractions = np.power(2,(frac/4))
        trial_settings = create_design([.55, 1.], [1., .55], fractions=fractions) #, n_runs=1)
        trial_settings = trial_settings.sample(n=10)


        trial_nr = 34

        for (p1, p2), d2 in trial_settings.groupby(['p1', 'p2'], sort=False):
            n_trials_in_miniblock = len(d2)
            self.trials.append(TaskInstructionTrial(session=self, trial_nr=trial_nr))
                                               #n_trials=n_trials_in_miniblock,
                                               #prob1=p1,
                                               #prob2=p2))
            trial_nr += 1

            for ix, row in d2.iterrows():
                self.trials.append(GambleTrial(self, trial_nr,
                                               prob1=row.p1, prob2=row.p2,
                                               num1=int(row.n1),
                                               num2=int(row.n2),
                                               ))
                trial_nr += 1

        txt = f"""
        Super gemacht!

        Wir sind jetzt am Ende der Einführung.

        Nun sind Sie bereit das "offiziellen" Experiment zu starten, in welchem ihre Antworten dann auch über die Bonus-Zahlung entscheiden.

        Falls irgendwas unklar ist, zögern Sie bitte nicht uns zu fragen!

        """

        self.trials.append(InstructionTrial(self, trial_nr=trial_nr, txt=txt))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default='instructee', nargs='?')
    parser.add_argument('--settings', default='instructions', nargs='?')
    cmd_args = parser.parse_args()

    subject, session, task, run = cmd_args.subject, 'instruction', 'instruction',  None
    output_dir, output_str = get_output_dir_str(subject, session, task ) #, run)

    log_file = op.join(output_dir, output_str + '_log.txt')
    logging.warn(f'Writing results to: {log_file}')

    settings_fn = op.join(op.dirname(__file__), 'settings',
                          f'{cmd_args.settings}.yml')

    session_object = InstructionSession(output_str=output_str,
                                        output_dir=output_dir,
                                        settings_file=settings_fn, subject=subject)

    session_object.create_trials()
    print(session_object.trials)
    logging.warn(
        f'Writing results to: {op.join(session_object.output_dir, session_object.output_str)}')
    session_object.run()
    session_object.close()
