"""This module loads mixing secrets vocal stems and their pitch contour estimated by crepe.
Run `./scripts/dataset-mixing-secrets/get-vocal-stems-to-local.sh` to copy them to your local folder.

Details of CREPE: https://arxiv.org/abs/1802.06182

The mixing secrets dataset doesn't have any split. Based on this 10-second audio files, I decided that
the first 19613 files are training set, 19613:20850 is for validation, 21850:21890 is for testing.
In this way, there's no artist/track overlapping across the sets.

"""
import os
import glob

import torch
import numpy as np
import soundfile as sf

NUM_VOCAL_STEMS = 21890  # 19613, 20850,


class MixingSecretDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, wav_path: str, npy_path: str):
        """

        Args:
            split: 'train', 'valid' ,'test'

            wav_path: directory path of the wav files.
            See ./scripts/dataset-mixing-secrets/get-vocal-stems-to-local.sh for more information.

            npy_path: directory path of the npy files.
        """
        self.split = split
        self.wav_path = wav_path
        self.npy_path = npy_path

        wav_filenames = sorted(glob.glob(os.path.join(self.wav_path, '*.wav')))
        npy_filenames = sorted(glob.glob(os.path.join(self.npy_path, '*.npy')))

        if len(wav_filenames) != len(npy_filenames) or len(wav_filenames) != NUM_VOCAL_STEMS:
            raise RuntimeError(f'{len(wav_filenames)} != {len(npy_filenames)}. They all should be {NUM_VOCAL_STEMS}.')

        self.fiilenames = [fn[:-4] for fn in wav_filenames]
        if self.split == 'train':
            self.filenames = self.filenames[:19613]
        elif self.split == 'valid':
            self.filenames = self.filenames[19613:20850]
        elif self.split == 'test':
            self.filenames = self.filenames[20850:]
        else:
            raise ValueError(f'self.split is unexpected --> {self.split}')

        self.sample_rate = 16000

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        """

        Args:
            index (int): an integer that is < 21890

        Returns:
            audio_signal (float): [-1.0, 1.0] normalized audio signal. 10 second. shape: (160000, )
            pitch (float): (1001=time, 360=pitch) shaped. prediction was made every 10ms. a pitch bin covers 20 cents.
            its range is C1 to B7.
        """

        audio_signal, sr = sf.read(os.path.join(self.wav_path, self.filenames[index] + '.wav'))
        assert sr == self.sample_rate
        pitch = np.load(os.path.join(self.npy_path, self.filenames[index] + '.activation.npy'))

        return audio_signal, pitch
