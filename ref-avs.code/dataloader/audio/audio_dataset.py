import torch
import numpy
import os
from dataloader.audio.preprocess_vgg.vggish_input import waveform_to_examples
import soundfile


class Audio(torch.utils.data.Dataset):
    def __init__(self, augmentation, directory_path, split):
        # temporarily set no augmentation.
        self.augmentation = augmentation
        self.directory_path = directory_path
        self.split = split

    def load_audio_wave(self, file_index, file_index_mix):
        audio_path = os.path.join(file_index, 'audio.wav')
        wav_data, sample_rate = soundfile.read(audio_path, dtype='int16')
        assert wav_data.dtype == numpy.int16, 'Bad sample type: %r' % wav_data.dtype

        if file_index_mix is not None:
            audio_path2 = os.path.join(file_index_mix, 'audio.wav')
            wav_data2, _ = soundfile.read(audio_path2, dtype='int16')
            mix_lambda = numpy.random.beta(10, 10)
            min_length = min(wav_data.shape[0], wav_data2.shape[0])
            wav_data = wav_data[:min_length] * mix_lambda + wav_data2[:min_length] * (1-mix_lambda)

        wav_data = self.augmentation(wav_data, sample_rate, self.split)
        audio_log_mel = torch.cat([waveform_to_examples(wav_data[:, 0], sample_rate, True).detach(),
                                   waveform_to_examples(wav_data[:, 1], sample_rate, True).detach()], dim=1)

        # for the vgg preprocess, we will need 5 seconds audio log.
        if audio_log_mel.shape[0] < 5:
            audio_log_mel = torch.cat([audio_log_mel,
                                       audio_log_mel[-1].unsqueeze(0).repeat(5-audio_log_mel.shape[0], 1, 1, 1)])
        return audio_log_mel

    def __len__(self):
        return len(self.audio_list)
