"""Fused audio-visual dataset for AVSBench-style indexing."""
import os
import random
import PIL.Image
import numpy
import torch
from dataloader.visual.visual_dataset import Visual
from dataloader.audio.audio_dataset import Audio
import pandas


class AV(torch.utils.data.Dataset):
    """Pairs video frames + labels from `Visual` with log-mel spectrograms from `Audio` via `metadata.csv`."""

    def __init__(self, split, augmentation, param, root_path='', data_name='find'):
        self.visual_dataset = Visual(augmentation['visual'], os.path.join(root_path, data_name), split, param.image_size, param.image_embedding_size)
        self.audio_dataset = Audio(augmentation['audio'], os.path.join(root_path, data_name), split)
        self.augment = augmentation
        self.split = split
        self.file_path = self.organise_files(self.split, root_path, data_name, csv_name_='avss_index/metadata.csv')

    def __getitem__(self, index):
        mixing_prob = 0. # we omit this option.
        other_index = random.randint(1, self.__len__()) - 1 if random.random() < mixing_prob and self.split == 'train' else None
        frame, label, prompts = self.visual_dataset.load_data(self.file_path[index])
        if other_index is not None:
            other_frame, other_label, other_prompts = self.visual_dataset.load_data(self.file_path[other_index])
            frame, label, prompts = self.visual_mix(frame, other_frame, label, other_label, prompts, other_prompts)
            audio_mel = self.audio_dataset.load_audio_wave(self.file_path[index], self.file_path[other_index])
        else:
            audio_mel = self.audio_dataset.load_audio_wave(self.file_path[index], None)

        assert other_index is None if self.split == 'test' else 1, print('no mix in validation.')

        return {'frame': frame, 'label': label, 'spectrogram': audio_mel, 'id': self.file_path[index],
                'prompts': prompts}

    def __len__(self):
        return len(self.file_path)

    @staticmethod
    def organise_files(split_, root_path_, data_name_, csv_name_):
        """Read rows from `csv_name_` under `root_path_` matching split and dataset label."""
        total_files = pandas.read_csv(os.path.join(root_path_, csv_name_))
        files_info = total_files[(total_files["split"] == split_) & (total_files["label"] == data_name_)]['uid']

        files_path = [os.path.join(root_path_, data_name_, files_name) for files_name in files_info]
        del total_files, files_info
        return files_path

    @staticmethod
    def visual_mix(frame1, frame2, label1, label2, prompts1, prompts2):
        mix_frame = frame1.clone()
        mix_label = label1.clone()
        bbx1, bby1, bbx2, bby2 = 0, 0, mix_label.shape[1] - 1, mix_label.shape[2] - 1

        for i in range(0, mix_frame.shape[0]):
            label_canvas_foreground = label2[i, bbx1:bbx2, bby1:bby2] > 0.
            mix_frame[i, :, bbx1:bbx2, bby1:bby2][:, label_canvas_foreground] = (
                    frame2[i, :, bbx1:bbx2, bby1:bby2][:, label_canvas_foreground])
            mix_label[i, bbx1:bbx2, bby1:bby2][label_canvas_foreground] = (
                    label2[i, bbx1:bbx2, bby1:bby2][label_canvas_foreground])

        return mix_frame, mix_label, prompts1



