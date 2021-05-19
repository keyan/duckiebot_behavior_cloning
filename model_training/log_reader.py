"""
This module provides the Reader() class to read log files and extract
image and linear/angular velocity data. It is slightly modified from
the module in the AIDO template:
    https://github.com/duckietown/challenge-aido_LF-baseline-behavior-cloning/blob/master/duckieTrainer/log_reader.py
"""
import pickle
from log_schema import Episode, Step, SCHEMA_VERSION

class Reader:
    def __init__(self, log_file):
        self._log_file = open(log_file, 'rb')

    def read(self):
        episode_data = None
        episode_index = 0
        end = False
        Observation=[]
        Linear=[]
        Angular=[]
        while True:
            if episode_data is None:
                try:
                    episode_data = pickle.load(self._log_file)
                    episode_index = 0
                except EOFError:
                    print("End of log file!")
                    print("Size: ",len(Observation)," ",len(Linear)," ",len(Angular))
                    return Observation,Linear,Angular
            try:
                step = episode_data.steps[episode_index]
                episode_index+=1
                Observation.append(step.obs)
                Linear.append(step.action[0])
                Angular.append(step.action[1])
            except IndexError:
                episode_data=None
                continue

    def close(self):
        self._log_file.close()
