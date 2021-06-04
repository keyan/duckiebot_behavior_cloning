import pickle

from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from log_schema import Episode, Step

SCHEMA_VERSION = "1.0.0"


class Logger:
    def __init__(self, log_file):
        self.episode = Episode(version=SCHEMA_VERSION)
        self.episode_count = 0

        self._log_file = log_file  # open(log_file, 'wb')
        # we log the data in a multithreaded fashion
        # self._multithreaded_recording = ThreadPoolExecutor(4)
        # self.recording = []

    def log(self, step: Step, info: Dict):
        if self.episode.metadata is None:
            self.episode.metadata = info
        self.episode.steps.append(step)

    def reset_episode(self):
        self.episode = Episode(version=SCHEMA_VERSION)

    def on_episode_done(self):
        print(f"episode {self.episode_count} done, writing to file")
        # The next file cause all episodes to be written to the same pickle FP. (Overwrite first?)
        # self._multithreaded_recording.submit(lambda: self._commit(self.episode))
        self._commit(self.episode)
        self.episode = Episode(version=SCHEMA_VERSION)
        self.episode_count += 1

    def _commit(self, episode):
        # we use pickle to store our data
        # pickle.dump(self.recording, self._log_file)
        with open(f"{self._log_file}_episode_{self.episode_count}.log", "wb") as f:
            pickle.dump(episode, f)
        # self._log_file.flush()
        # del self.recording[:]
        # self.recording.clear()

    def close(self):
        pass
        # self._multithreaded_recording.shutdown()
        # self._log_file.close()