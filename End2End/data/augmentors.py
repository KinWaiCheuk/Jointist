import logging
import numpy as np
import sox

from End2End.constants import SAMPLE_RATE


class Augmentor:
    def __init__(self, augmentation: str):
        r"""Data augmentor.

        Args:
            augmentation: str, 'none' | 'aug'
        """

        self.augmentation = augmentation
        self.random_state = np.random.RandomState(1234)
        self.sample_rate = SAMPLE_RATE

    def __call__(self, x):
        r"""Do augmentation.

        Args:
            x: ndarray, (audio_length,)

        Returns:
            ndarray, (audio_length)
        """

        if self.augmentation == 'none':
            return x

        elif self.augmentation == 'aug':
            return self.aug(x)

    def aug(self, x):
        # Todo
        clip_samples = len(x)

        logger = logging.getLogger('sox')
        logger.propagate = False

        tfm = sox.Transformer()
        tfm.set_globals(verbosity=0)

        tfm.pitch(self.random_state.uniform(-0.1, 0.1, 1)[0])
        tfm.contrast(self.random_state.uniform(0, 100, 1)[0])

        tfm.equalizer(
            frequency=self.loguniform(32, 4096, 1)[0],
            width_q=self.random_state.uniform(1, 2, 1)[0],
            gain_db=self.random_state.uniform(-30, 10, 1)[0],
        )

        tfm.equalizer(
            frequency=self.loguniform(32, 4096, 1)[0],
            width_q=self.random_state.uniform(1, 2, 1)[0],
            gain_db=self.random_state.uniform(-30, 10, 1)[0],
        )

        tfm.reverb(reverberance=self.random_state.uniform(0, 70, 1)[0])

        aug_x = tfm.build_array(input_array=x, sample_rate_in=self.sample_rate)
        aug_x = pad_truncate_sequence(aug_x, clip_samples)

        return aug_x

    def loguniform(self, low, high, size):
        return np.exp(self.random_state.uniform(np.log(low), np.log(high), size))
