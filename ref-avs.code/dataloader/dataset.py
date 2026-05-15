"""Ref-AVS dataset: frames, masks, log-mel audio, and referring expressions."""
import os
import numpy
import torch
import pandas

from dataloader.visual.visual_dataset import Visual
from dataloader.audio.audio_and_text_dataset import AudioAndText


class AV(torch.utils.data.Dataset):
    """Pairs ``Visual`` with ``AudioAndText`` via REFAVS ``metadata.csv``."""

    def __init__(self, split, augmentation, param, root_path=''):
        self.visual_dataset = Visual(
            augmentation['visual'], root_path, split,
            param.image_size, param.image_embedding_size,
        )
        self.audio_and_text_dataset = AudioAndText(augmentation['audio'], root_path, split)
        self.split = split
        self.file_path = self.organise_files(self.split, root_path, csv_name_='metadata.csv')

    def __getitem__(self, index):
        vid, fid, exp, _ = self.file_path[index]
        frame, label, prompts = self.visual_dataset.load_data(vid, fid)
        audio_mel, text_feature = self.audio_and_text_dataset.load_audio_wave(vid, exp)
        return {
            'frame': frame,
            'label': label,
            'spectrogram': audio_mel,
            'text': text_feature,
            'id': self.file_path[index],
            'prompts': prompts,
        }

    def __len__(self):
        return len(self.file_path)

    @staticmethod
    def organise_files(split_, root_path_, csv_name_):
        total_files = pandas.read_csv(os.path.join(root_path_, csv_name_))
        if split_ == 'test_n':
            rows = zip(
                total_files[total_files['split'] == split_]['uid'],
                total_files[total_files['split'] == split_]['fid'],
                total_files[total_files['split'] == split_]['exp'],
            )
            return [
                [name.rsplit('_', 2)[0], object_id, expression, 0]
                for name, object_id, expression in rows
            ]

        rows = zip(
            total_files[total_files['split'] == split_]['vid'],
            total_files[total_files['split'] == split_]['fid'],
            total_files[total_files['split'] == split_]['exp'],
        )
        file_path = [[vid, fid, expression, 0] for vid, fid, expression in rows]

        if split_ == 'train':
            null_uids = list(total_files[total_files['split'] == split_]['uid'])
            assert len(null_uids) == len(file_path)
            for idx, row in enumerate(file_path):
                if 'null_' in null_uids[idx]:
                    row[0] = null_uids[idx].rsplit('_', 2)[0]
                row[-1] = null_uids[idx].rsplit('_', 2)[1]

        return file_path
