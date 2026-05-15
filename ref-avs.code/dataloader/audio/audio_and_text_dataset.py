"""Load REFAVS audio (log-mel) and pass through referring expression strings for the text encoder."""
import os

import numpy
import soundfile
import torch

from dataloader.audio.preprocess_vgg.vggish_input import waveform_to_examples


class AudioAndText(torch.utils.data.Dataset):
    def __init__(self, augmentation, directory_path, split):
        self.augmentation = augmentation
        self.directory_path = directory_path
        self.split = split

    def load_audio_wave(self, file_index, text_expression):
        audio_path = os.path.join(self.directory_path, 'media', file_index, 'audio.wav')
        wav_data, sample_rate = soundfile.read(audio_path, dtype='int16')
        assert wav_data.dtype == numpy.int16, 'Bad sample type: %r' % wav_data.dtype
        wav_data = self.augmentation(wav_data, sample_rate, self.split)
        if len(wav_data.shape) < 2:
            wav_data = wav_data[:, numpy.newaxis]
            wav_data = numpy.repeat(wav_data, axis=-1, repeats=2)

        audio_log_mel = torch.cat([
            waveform_to_examples(wav_data[:, 0], sample_rate, True).detach(),
            waveform_to_examples(wav_data[:, 1], sample_rate, True).detach(),
        ], dim=1)

        # VGGish expects at least 5 temporal segments.
        if audio_log_mel.shape[0] < 5:
            pad = audio_log_mel[-1].unsqueeze(0).repeat(5 - audio_log_mel.shape[0], 1, 1, 1)
            audio_log_mel = torch.cat([audio_log_mel, pad])
        return audio_log_mel, text_expression
