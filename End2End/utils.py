import os
import logging
import numpy as np
import yaml
import datetime
import pickle


def create_logging(log_dir, filemode):
    os.makedirs(log_dir, exist_ok=True)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode,
    )

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


def float32_to_int16(x: np.ndarray):
    assert np.max(np.abs(x)) <= 2.0
    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x: np.ndarray):
    return (x / 32767.0).astype(np.float32)


def read_yaml(config_yaml):
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


def note_to_freq(piano_note):
    return 2 ** ((piano_note - 39) / 12) * 440


def get_pitch_shift_factor(pitch_shift):
    return 2 ** (pitch_shift / 12)


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path

        self.backup_statistics_path = "{}_{}.pkl".format(
            os.path.splitext(self.statistics_path)[0],
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )

        self.statistics_dict = {"train": [], "test": []}

    def append(self, steps, statistics, split):
        statistics["steps"] = steps
        self.statistics_dict[split].append(statistics)

    def dump(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, "wb"))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, "wb"))
        logging.info("    Dump statistics to {}".format(self.statistics_path))
        logging.info("    Dump statistics to {}".format(self.backup_statistics_path))

    def load_state_dict(self, resume_steps):
        self.statistics_dict = pickle.load(open(self.statistics_path, "rb"))

        resume_statistics_dict = {"train": [], "test": []}

        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics["steps"] <= resume_steps:
                    resume_statistics_dict[key].append(statistics)

        self.statistics_dict = resume_statistics_dict
