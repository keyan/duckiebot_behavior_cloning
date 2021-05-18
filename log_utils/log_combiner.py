import pickle
import argparse
from log_schema import Episode, Step
SCHEMA_VERSION = "1.0.0"
class Combiner:
    def __init__(self,log1,log2,output):
        self._log1 = open(log1,'rb')
        self._log2 = open(log2,'rb')
        self._output = open(output,'wb')
        self.episode_counter = 0
        self.combine()

    def combine(self):
        episode_data = None
        episode_index = 0
        while True:
            try:
                episode_data = pickle.load(self._log1)
                episode_index += 1
            except EOFError:
                print("Captured total {} episodes".format(episode_index))
                print("End of log file!")
                break
            self.commit_episode(episode_data)
            self.episode=Episode(version=SCHEMA_VERSION)
            self.episode_counter+=1

        while True:
            try:
                episode_data = pickle.load(self._log2)
                episode_index += 1
            except EOFError:
                print("Captured total {} episodes".format(episode_index))
                print("End of log file!")
                break
            self.commit_episode(episode_data)
            self.episode=Episode(version=SCHEMA_VERSION)
            self.episode_counter+=1
        self.close()


    def commit_episode(self,episode):
        pickle.dump(episode,self._output)
        self._output.flush()

    def close(self):
        self._log1.close()
        self._log2.close()
        self._output.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log1",default=None)
    parser.add_argument("--log2",default=None)
    parser.add_argument("--output",default=None)
    args = parser.parse_args()

    try:
        assert args.log1 is not None or args.log2 is not None or args.output is not None
    except  Exception:
        print("Please provide all inputs! 3 should be given.")

    Combiner(args.log1,args.log2,args.output)


