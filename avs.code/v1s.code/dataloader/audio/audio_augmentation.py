import numpy


class Augmentation(object):
    """Audio pre-step used by training/inference: int16 waveform -> float in [-1, 1].

    The previous audiomentations-based transforms were commented out and never applied;
    behavior is unchanged: only scaling by 1/32768.
    """

    def __init__(self, mono=True):
        self.mono = mono

    def train_aug(self, x_, sr_):
        x_ = x_ / 32768.0
        return x_

    def test_process(self, x_):
        x_ = x_ / 32768.0
        return x_

    def __call__(self, x, sr, split):
        return self.train_aug(x, sr) if split == "train" else self.test_process(x)
