import os
import re
import PIL.Image
import matplotlib.pyplot as plt
import numpy
import torch
import pandas
import torchvision


class Visual(torch.utils.data.Dataset):
    def __init__(self, augmentation, directory_path, split, image_size, image_embedding_size):
        self.augment = augmentation
        self.directory_path = directory_path
        self.split = split
        self.image_size = image_size
        self.embedding_size = image_embedding_size

    def get_frame_and_label(self, file_prefix, object_id):
        # if self.split == 'null':
        #     frame_path = os.path.join(self.directory_path, 'media_cross', file_prefix, 'frames')
        #     frame_path = [os.path.join(frame_path, i) for i in os.listdir(frame_path)]
        #     frame_path.sort(key=lambda x: tuple(map(int, x.split('/')[-1].split("_")[-1].split('.jpg')[0])))
        #     # dummy empty label.
        #     frame = [PIL.Image.open(i) for i in frame_path]
        #     label = [PIL.Image.new('L', frame[0].size)] * len(frame)
        # else:
        frame_path = os.path.join(self.directory_path, 'media', file_prefix, 'frames')
        label_path = os.path.join(self.directory_path, 'gt_mask', file_prefix, 'fid_{}'.format(str(object_id)))
        frame_path = [os.path.join(frame_path, i) for i in os.listdir(frame_path)]
        label_path = [os.path.join(label_path, i) for i in os.listdir(label_path)]
        frame_path.sort(key=lambda x: tuple(map(int, x.split('/')[-1].split("_")[-1].split('.jpg')[0])))
        label_path.sort(key=lambda x: tuple(map(int, x.split('/')[-1].split("_")[-1].split('.png')[0])))
        frame = [PIL.Image.open(i) for i in frame_path]
        label = [PIL.Image.open(i).convert('L') for i in label_path]
        return frame, label

    def load_data(self, file_prefix, object_id):
        frame, label = self.get_frame_and_label(file_prefix, object_id)
        label_idx = torch.tensor(list([1] * 10), dtype=torch.bool)

        prompts = {}
        image_batch = [None]*len(frame)
        label_batch = [None]*len(frame)
        
        if self.split == 'train':
            # apply sam2 augmentation.
            frame, label = self.augment(frame, label)

        for i in range(len(frame)):
            if 'test_' in self.split:
                # note: there is no augmentation in here.
                curr_frame, curr_label = self.augment(frame[i], label[i], split=self.split)
            else:
                curr_frame, curr_label = frame[i], label[i]

            curr_label[curr_label > 0.] = 1.
            image_batch[i], label_batch[i] = curr_frame, curr_label

            # image_batch[i], label_batch[i] = self.augment(frame[i], label[i], split=self.split)
            # note: we simply convert the code to binary mask in v1s, v1m;
            # to some reason, we failed to load the label in `L' format and had to hardcoding here.
            # label_batch[i][label_batch[i] > 0.] = 1.

            # prompts['box_coords'][i], prompts['masks'][i] = self.receive_other_prompts(label_batch[i])

        # organise the prompts
        # prompts.update({'masks': torch.stack(prompts['masks'], dim=0)})
        # prompts.update({'box_coords': torch.stack(prompts['box_coords'], dim=0)})
        # prompts.update({'point_labels': torch.stack(prompts['point_labels'], dim=0)})
        prompts.update({'label_index': label_idx})
        return torch.stack(image_batch, dim=0), torch.stack(label_batch, dim=0), prompts

    def receive_other_prompts(self, y_):
        # y_ = torch.zeros_like(y_)
        if len(torch.unique(y_)) > 1:
            # foreground point
            points_foreground = torch.stack(torch.where(y_ > 0)[::-1], dim=0).transpose(1, 0)

            # bbox prompt (left-top corner & right-bottom corner)
            bbox_one = torch.min(points_foreground[:, 0]), torch.min(points_foreground[:, 1])
            bbox_fou = torch.max(points_foreground[:, 0]), torch.max(points_foreground[:, 1])
            bbox_coord = torch.tensor(bbox_one + bbox_fou, dtype=torch.float)
            bbox_coord = self.transform_coords(bbox_coord, orig_hw=y_.squeeze().shape)
            # mask prompt
            low_mask = torchvision.transforms.functional.resize(y_.clone(), [self.embedding_size*4, self.embedding_size*4],
                                                                torchvision.transforms.InterpolationMode.NEAREST)
        else:
            # for the pure background situation.
            bbox_coord = torch.zeros([4], dtype=torch.float).fill_(float('nan'))
            low_mask = torch.zeros([1, self.embedding_size*4, self.embedding_size*4], dtype=torch.float).fill_(float('nan'))

        return bbox_coord, low_mask

    # we transfer the coords to SAM's input resolution (1024, 1024).
    def transform_coords(self, coords: torch.Tensor, orig_hw=None) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. The coordinates can be in absolute image or normalized coordinates,
        If the coords are in absolute image coordinates, normalize should be set to True and original image size is required.

        Returns
            Un-normalized coordinates in the range of [0, 1] which is expected by the sam2 model.
        """
        h, w = orig_hw
        coords = coords.clone().reshape(-1, 2, 2)
        coords[..., 0] = coords[..., 0] / w
        coords[..., 1] = coords[..., 1] / h
        coords = coords * self.image_size  # unnormalize coords
        return coords.reshape(4)



