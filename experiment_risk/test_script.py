#%%
import os
import yaml
import collections
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from psychopy import core
from psychopy.sound import Sound
from psychopy.hardware.emulator import SyncGenerator
from psychopy.visual import Window, TextStim
from psychopy.event import waitKeys, Mouse
from psychopy.monitors import Monitor
from psychopy import logging
from psychopy import prefs as psychopy_prefs

#%%
class Session:

    def __init__(self, output_str, output_dir=None, settings_file=None):

        self.output_str = output_str
        self.output_dir = op.join(os.getcwd(), 'logs') if output_dir is None else output_dir 
        self.settings_file = settings_file
        self.clock = core.Clock()
        self.timer = core.Clock()
        self.exp_start = None
        self.exp_stop = None
        self.current_trial = None
        self.global_log = pd.DataFrame(columns=['trial_nr', 'onset', 'event_type', 'phase', 'response', 'nr_frames'])
        self.nr_frames = 0  # keeps track of nr of nr of frame flips
        self.first_trial = True
        self.closed = False

        # Initialize
        self.settings = self._load_settings()
        self.monitor = self._create_monitor()
        self.win = self._create_window()
        self.mouse = Mouse(**self.settings['mouse'])
        self.logfile = self._create_logfile()
        #self.default_fix = create_circle_fixation(self.win, radius=0.075, color=(1, 1, 1))
        #self.mri_trigger = None  # is set below
        #self.mri_simulator = self._setup_mri()

    def _load_settings(self):
        """ Loads settings and sets preferences. """

        with open(self.settings_file, 'r', encoding='utf8') as f_in:   
            settings = yaml.safe_load(f_in)
        
        # Write settings to sub dir
        if not op.isdir(self.output_dir):
            os.makedirs(self.output_dir) 

        settings_out = op.join(self.output_dir, self.output_str + '_expsettings.yml')
        with open(settings_out, 'w') as f_out:  # write settings to disk
            yaml.dump(settings, f_out, indent=4, default_flow_style=False)

        exp_prefs = settings['preferences']  # set preferences globally
        for preftype, these_settings in exp_prefs.items():
            for key, value in these_settings.items():
                pref_subclass = getattr(psychopy_prefs, preftype)
                pref_subclass[key] = value
                setattr(psychopy_prefs, preftype, pref_subclass)

        return settings

    def _create_monitor(self):
        """ Creates the monitor based on settings and save to disk. """
        monitor = Monitor(**self.settings['monitor'])
        monitor.setSizePix(self.settings['window']['size'])
        monitor.save()  # needed for iohub eyetracker
        return monitor

    def _create_window(self):
        """ Creates a window based on the settings and calculates framerate. """
        win = Window(monitor=self.monitor.name, **self.settings['window'])
        win.flip(clearBuffer=True)
        self.actual_framerate = win.getActualFrameRate()
        if self.actual_framerate is None:
            logging.warn("framerate not measured, substituting 60 by default")
            self.actual_framerate = 60.0
        t_per_frame = 1. / self.actual_framerate
        logging.warn(f"Actual framerate: {self.actual_framerate:.5f} "
                     f"(1 frame = {t_per_frame:.5f})")
        return win
    def _set_exp_stop(self):
        """ Called on last win.flip(); timestamps end of exp. """
        self.exp_stop = self.clock.getTime()



# %%
from stimuli import _create_stimulus_array, CertaintyStimulus, ProbabilityPieChart
import os.path as op

settings_fn = op.join('/Users/mrenke/git/numrisk/experiment_risk/', 'settings', 'macbook.yml')

toy = Session(settings_file = settings_fn,output_str = '.tsv' )
core.quit()

piechart_pos1 = 0.2 , 0.2
#piechart_pos2 = .5 * -text_width - .25 * piechart_width, -1.5 * piechart_width
piechart_width = 1
piechart1 = ProbabilityPieChart(toy.win, prob1 = 0.55, pos=piechart_pos1, size= piechart_width)
piechart1.draw()



# %%
from psychopy import visual, core
from stimuli import ProbabilityPieChart

win = visual.Window([400,400])

piechart_pos1 = 0.2 , 0.2
piechart_width = 1
prob1 = 0.55
piechart1 = ProbabilityPieChart(win, prob1, pos=piechart_pos1, size= piechart_width)



message = visual.TextStim(win, text='hello')
message.autoDraw = True  # Automatically draw every frame
win.flip()
core.wait(2.0)
message.text = 'world'  # Change properties of existing stim
win.flip()
core.wait(2.0)
piechart1.draw()
win.flip()
core.wait(2.0)
win.close()



# %%
win = visual.Window([400,400], monitor = 1)
message = visual.TextStim(win, text='hello')
win.flip()
core.wait(2.0)
win.close()

# %%
import yaml
import os.path as op
from psychopy.monitors import Monitor


settings_fn = op.join('/Users/mrenke/git/numrisk/experiment_risk/', 'settings', 'macbook.yml')
settings = yaml.safe_load(settings_fn)
monitor = Monitor(settings['monitor'])

# %%
import os

# %%
