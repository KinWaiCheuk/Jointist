import logging
import os
import pathlib
from typing import List

import pickle
import numpy as np
import h5py
from pytorch_lightning.utilities import rank_zero_only
import torch.distributed as dist


class End2EndSegmentSampler:
    def __init__(
        self,
        hdf5s_dir: str,
        split: str,
        segment_seconds: float,
        hop_seconds: float,
        batch_size: int,
        steps_per_epoch: int,
        evaluation: bool,
        max_evaluation_steps: int = -1,
        random_seed: int = 1234,
        mini_data: bool = False,
    ):
        r"""Sampler is used to sample segments for training or evaluation.

        Args:
          hdf5s_dir: str
          split: 'train' | 'validation' | 'test'
          segment_seconds: float, e.g., 10.0
          hop_seconds: float, e.g., 1.0
          batch_size: int, e.g., 16
          evaluation: bool, set to True in training, and False in evaluation
          max_evaluation_steps: only activate when evaluation=True
          random_seed: int
          mini_data: bool, sample a small amount of data for debugging
        """
        assert split in ['train', 'validation', 'test']

        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.evaluation = evaluation
        self.max_evaluation_steps = max_evaluation_steps

        # paths
        split_hdf5s_dir = os.path.join(hdf5s_dir, split)

        # Traverse directory
        hdf5_paths = sorted([str(path) for path in pathlib.Path(split_hdf5s_dir).rglob('*.h5')])

        self.segment_list = []

        n = 0
        for hdf5_path in hdf5_paths:
            try:
                with h5py.File(hdf5_path, 'r') as hf:
                    if hf.attrs['split'].decode() == split:

                        audio_name = '{}.h5'.format(os.path.splitext(hf.attrs['audio_name'])[0].decode())
                        self.segment_list.append([hf.attrs['split'].decode(), audio_name,])
            except:
                from IPython import embed; embed(using=False); os._exit(0)
        # self.segment_list looks like:
        # [['train', 'Track01122.h5', 0],
        #  ['train', 'Track01122.h5', 1.0],
        #  ['train', 'Track01122.h5', 2.0],
        #  ...]

        if evaluation:
            logging.info('Mini-batches for evaluating {} set: {}'.format(split, max_evaluation_steps))

        else:
            logging.info('Training segments: {}'.format(len(self.segment_list)))

        self.pointer = 0
        self.segment_indexes = np.arange(len(self.segment_list))

        if len(self.segment_indexes) == 0:
            error_msg = 'No training data found in {}! Please set up your workspace and data path properly!'.format(split_hdf5s_dir)
            raise Exception(error_msg)


        # Both training and evaluation shuffle segment_indexes in the begining.
        self.random_state = np.random.RandomState(random_seed)
        self.random_state.shuffle(self.segment_indexes)

    def __iter__(self):
        r"""Get batch meta.

        Returns:
            batch_segment_list: list of list, e.g.,
                [['train', 'Track00255.h5', 4.0],
                 ['train', 'Track00894.h5', 53.0],
                 ['train', 'Track01422.h5', 77.0],
                 ...]
        """
        if self.evaluation:
            return self.iter_eval()
        else:
            return self.iter_train()

    def iter_train(self):
        r"""Get batch meta for training.

        Returns:
            batch_segment_list: list of list, e.g.,
                [['train', 'Track00255.h5', 4.0],
                 ['train', 'Track00894.h5', 53.0],
                 ['train', 'Track01422.h5', 77.0],
                 ...]
        """
        while True:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[self.pointer]
                self.pointer += 1

                if self.pointer >= len(self.segment_indexes):
                    self.random_state.shuffle(self.segment_indexes)
                    self.pointer = 0

                batch_segment_list.append(self.segment_list[index])
                i += 1

            yield batch_segment_list

    def iter_eval(self):
        r"""Get batch meta for evaluation.

        Returns:
            batch_segment_list: list of list, e.g.,
                [['train', 'Track00255.h5', 4.0],
                 ['train', 'Track00894.h5', 53.0],
                 ['train', 'Track01422.h5', 77.0],
                 ...]
        """
        _pointer = 0
        _steps = 0

        while _pointer < len(self.segment_indexes):

            if _steps == self.max_evaluation_steps:
                break

            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[_pointer]
                _pointer += 1

                if _pointer >= len(self.segment_indexes):
                    break

                batch_segment_list.append(self.segment_list[index])
                i += 1

            _steps += 1

            yield batch_segment_list

    def __len__(self):
        return self.steps_per_epoch

    def state_dict(self):
        state = {'pointer': self.pointer, 'segment_indexes': self.segment_indexes}
        return state

    def load_state_dict(self, state):
        self.pointer = state['pointer']
        self.segment_indexes = state['segment_indexes']

    @rank_zero_only
    def log(self, str):
        logging.info(str)



