# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transforms and data augmentation for both image + bbox.
"""

import logging

import random
from typing import Iterable

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as Fv2
from PIL import Image as PILImage
# from docutils.nodes import label
import numpy
from torchvision.transforms import InterpolationMode

# from utils.data_utils import VideoDatapoint


def hflip(frames, labels, index):
    # print(index)
    # print(len(frames), frames[index].size, type(frames[index]))
    # print(len(labels), labels[index].size, type(labels[index]))
    frames[index] = F.hflip(frames[index])
    labels[index] = F.hflip(labels[index])
    # for obj in frames[index].objects:
    #     if obj.segment is not None:
    #         obj.segment = F.hflip(obj.segment)

    return frames, labels


def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = max_size * min_original_size / max_original_size

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = int(round(size))
        oh = int(round(size * h / w))
    else:
        oh = int(round(size))
        ow = int(round(size * w / h))

    return (oh, ow)


def resize(frames, labels, index, size, max_size=None, square=False, v2=False):
    # size can be min_size (scalar) or (w, h) tuple
    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    if square:
        size = size, size
    else:
        raise NotImplementedError
        # cur_size = (
        #     frames[index].data.size()[-2:][::-1]
        #     if v2
        #     else frames[index].data.size
        # )
        # size = get_size(cur_size, size, max_size)

    # old_size = (
    #     frames[index].data.size()[-2:][::-1]
    #     if v2
    #     else frames[index].data.size
    # )
    if v2:
        frames[index].data = Fv2.resize(
            frames[index].data, size, antialias=True
        )
    else:
        frames[index] = F.resize(frames[index], size)
        labels[index] = F.resize(labels[index], size)
    # new_size = (
    #     frames[index].data.size()[-2:][::-1]
    #     if v2
    #     else frames[index].data.size
    # )

    # for obj in frames[index].objects:
    #     if obj.segment is not None:
    #         obj.segment = F.resize(obj.segment[None, None], size).squeeze()

    # h, w = size
    # frames[index].size = (h, w)
    return frames, labels


def pad(frames, index, padding, v2=False):
    old_h, old_w = frames[index].size
    h, w = old_h, old_w
    if len(padding) == 2:
        # assumes that we only pad on the bottom right corners
        frames[index].data = F.pad(
            frames[index].data, (0, 0, padding[0], padding[1])
        )
        h += padding[1]
        w += padding[0]
    else:
        # left, top, right, bottom
        frames[index].data = F.pad(
            frames[index].data,
            (padding[0], padding[1], padding[2], padding[3]),
        )
        h += padding[1] + padding[3]
        w += padding[0] + padding[2]

    frames[index].size = (h, w)

    for obj in frames[index].objects:
        if obj.segment is not None:
            if v2:
                if len(padding) == 2:
                    obj.segment = Fv2.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = Fv2.pad(obj.segment, tuple(padding))
            else:
                if len(padding) == 2:
                    obj.segment = F.pad(obj.segment, (0, 0, padding[0], padding[1]))
                else:
                    obj.segment = F.pad(obj.segment, tuple(padding))
    return frames


class RandomHorizontalFlip:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform

    def __call__(self, frames, labels, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for i in range(len(frames)):
                    frames, labels = hflip(frames, labels, i)
            return frames, labels
        for i in range(len(frames)):
            if random.random() < self.p:
                    frames, labels = hflip(frames, labels, i)
        return frames, labels


class RandomResizeAPI:
    def __init__(
        self, sizes, consistent_transform, max_size=None, square=False, v2=False
    ):
        if isinstance(sizes, int):
            sizes = (sizes,)
        assert isinstance(sizes, Iterable)
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square
        self.consistent_transform = consistent_transform
        self.v2 = v2

    def __call__(self, frames, labels):
        if self.consistent_transform:
            size = random.choice(self.sizes)
            for i in range(len(frames)):
                frames, labels = resize(
                    frames, labels, i, size, self.max_size, square=self.square, v2=self.v2
                )
            return frames, labels
        for i in range(len(frames)):
            size = random.choice(self.sizes)
            frames, labels = resize(
                frames, labels, i, size, self.max_size, square=self.square, v2=self.v2
            )
        return frames, labels


class ToTensorAPI:
    def __init__(self, v2=False):
        self.v2 = v2

    def __call__(self, frames, labels, **kwargs):
        for img_idx in range(len(frames)):
            if self.v2:
                raise NotImplementedError
                # frames[img_idx] = Fv2.to_tensor(frames[img_idx])
            else:
                frames[img_idx] = F.to_tensor(frames[img_idx])
                labels[img_idx] = torch.tensor(numpy.array(labels[img_idx]), dtype=torch.float)
        return frames, labels


class NormalizeAPI:
    def __init__(self, mean, std, v2=False):
        self.mean = mean
        self.std = std
        self.v2 = v2

    def __call__(self, frames, labels, **kwargs):
        for img_idx in range(len(frames)):
            # if self.v2:
            #     img.data = Fv2.convert_image_dtype(img.data, torch.float32)
            #     img.data = Fv2.normalize(img.data, mean=self.mean, std=self.std)
            # else:
            frames[img_idx] = F.normalize(frames[img_idx], mean=self.mean, std=self.std)

        return frames, labels

'''
    <dataloader.sam2_dataset.transforms.RandomHorizontalFlip object at 0x75c815561b40>
    <dataloader.sam2_dataset.transforms.RandomAffine object at 0x75c815561bd0>
    <dataloader.sam2_dataset.transforms.RandomResizeAPI object at 0x75c815561c60>
    <dataloader.sam2_dataset.transforms.ColorJitter object at 0x75c815561cc0>
    <dataloader.sam2_dataset.transforms.RandomGrayscale object at 0x75c815561cf0>
    <dataloader.sam2_dataset.transforms.ColorJitter object at 0x75c815561de0>
    <dataloader.sam2_dataset.transforms.ToTensorAPI object at 0x75c815507280>
    <dataloader.sam2_dataset.transforms.NormalizeAPI object at 0x75c815507490>
'''
class ComposeAPI:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, frames, labels, **kwargs):
        for t in self.transforms:
            frames, labels = t(frames, labels, **kwargs)
        return frames, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomGrayscale:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform
        self.Grayscale = T.Grayscale(num_output_channels=3)

    def __call__(self, frames, labels, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for img_idx in range(len(frames)):
                    frames[img_idx] = self.Grayscale(frames[img_idx])
            return frames, labels
        for img_idx in range(len(frames)):
            if random.random() < self.p:
                frames[img_idx] = self.Grayscale(frames[img_idx])
        return frames, labels


class ColorJitter:
    def __init__(self, consistent_transform, brightness, contrast, saturation, hue):
        self.consistent_transform = consistent_transform
        self.brightness = (
            brightness
            if isinstance(brightness, list)
            else [max(0, 1 - brightness), 1 + brightness]
        )
        self.contrast = (
            contrast
            if isinstance(contrast, list)
            else [max(0, 1 - contrast), 1 + contrast]
        )
        self.saturation = (
            saturation
            if isinstance(saturation, list)
            else [max(0, 1 - saturation), 1 + saturation]
        )
        self.hue = hue if isinstance(hue, list) or hue is None else ([-hue, hue])

    def __call__(self, frames, labels, **kwargs):
        if self.consistent_transform:
            # Create a color jitter transformation params
            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = T.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        for img in frames:
            if not self.consistent_transform:
                (
                    fn_idx,
                    brightness_factor,
                    contrast_factor,
                    saturation_factor,
                    hue_factor,
                ) = T.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue
                )
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)
        return frames, labels


class RandomAffine:
    def __init__(
        self,
        degrees,
        consistent_transform,
        scale=None,
        translate=None,
        shear=None,
        image_mean=(123, 116, 103),
        label_fill_value=0.,
        log_warning=True,
        num_tentatives=1,
        image_interpolation="bicubic",
    ):
        """
        The mask is required for this transform.
        if consistent_transform if True, then the same random affine is applied to all frames and masks.
        """
        self.degrees = degrees if isinstance(degrees, list) else ([-degrees, degrees])
        self.scale = scale
        self.shear = (
            shear if isinstance(shear, list) else ([-shear, shear] if shear else None)
        )
        self.translate = translate
        self.fill_img = image_mean
        self.fill_label = label_fill_value
        self.consistent_transform = consistent_transform
        self.log_warning = log_warning
        self.num_tentatives = num_tentatives
        assert self.num_tentatives >= 1., 'must have at least one if we utilise the augmentation.'

        if image_interpolation == "bicubic":
            self.image_interpolation = InterpolationMode.BICUBIC
        elif image_interpolation == "bilinear":
            self.image_interpolation = InterpolationMode.BILINEAR
        else:
            raise NotImplementedError

    def __call__(self, frames, labels, **kwargs):
        for _tentative in range(self.num_tentatives):
            res_img, res_labels = self.transform_frames(frames, labels)
            # if res is not None:
        return res_img, res_labels

        # raise NotImplementedError
        # if self.log_warning:
        #     logging.warning(
        #         f"Skip RandomAffine for zero-area mask in first frame after {self.num_tentatives} tentatives"
        #     )
        # return frames

    def transform_frames(self, frames, labels):
        _, height, width = F.get_dimensions(frames[0])
        img_size = [width, height]

        if self.consistent_transform:
            # Create a random affine transformation
            affine_params = T.RandomAffine.get_params(
                degrees=self.degrees,
                translate=self.translate,
                scale_ranges=self.scale,
                shears=self.shear,
                img_size=img_size,
            )

        for img_idx, img in enumerate(frames):
            if not self.consistent_transform:
                # if not consistent we create a new affine params for every frame&mask pair Create a random affine transformation
                affine_params = T.RandomAffine.get_params(
                    degrees=self.degrees,
                    translate=self.translate,
                    scale_ranges=self.scale,
                    shears=self.shear,
                    img_size=img_size,
                )
            frames[img_idx] = F.affine(
                img,
                *affine_params,
                interpolation=self.image_interpolation,
                fill=self.fill_img,
            )
            labels[img_idx] = F.affine(
                labels[img_idx],
                *affine_params,
                # default: interpolation='nearest',
                fill=self.fill_label,
            )
        return frames, labels


'''
def random_mosaic_frame(
    datapoint,
    index,
    grid_h,
    grid_w,
    target_grid_y,
    target_grid_x,
    should_hflip,
):
    # Step 1: downsize the images and paste them into a mosaic
    image_data = datapoint.frames[index].data
    is_pil = isinstance(image_data, PILImage.Image)
    if is_pil:
        H_im = image_data.height
        W_im = image_data.width
        image_data_output = PILImage.new("RGB", (W_im, H_im))
    else:
        H_im = image_data.size(-2)
        W_im = image_data.size(-1)
        image_data_output = torch.zeros_like(image_data)

    downsize_cache = {}
    for grid_y in range(grid_h):
        for grid_x in range(grid_w):
            y_offset_b = grid_y * H_im // grid_h
            x_offset_b = grid_x * W_im // grid_w
            y_offset_e = (grid_y + 1) * H_im // grid_h
            x_offset_e = (grid_x + 1) * W_im // grid_w
            H_im_downsize = y_offset_e - y_offset_b
            W_im_downsize = x_offset_e - x_offset_b

            if (H_im_downsize, W_im_downsize) in downsize_cache:
                image_data_downsize = downsize_cache[(H_im_downsize, W_im_downsize)]
            else:
                image_data_downsize = F.resize(
                    image_data,
                    size=(H_im_downsize, W_im_downsize),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,  # antialiasing for downsizing
                )
                downsize_cache[(H_im_downsize, W_im_downsize)] = image_data_downsize
            if should_hflip[grid_y, grid_x].item():
                image_data_downsize = F.hflip(image_data_downsize)

            if is_pil:
                image_data_output.paste(image_data_downsize, (x_offset_b, y_offset_b))
            else:
                image_data_output[:, y_offset_b:y_offset_e, x_offset_b:x_offset_e] = (
                    image_data_downsize
                )

    datapoint.frames[index].data = image_data_output

    # Step 2: downsize the masks and paste them into the target grid of the mosaic
    for obj in datapoint.frames[index].objects:
        if obj.segment is None:
            continue
        assert obj.segment.shape == (H_im, W_im) and obj.segment.dtype == torch.uint8
        segment_output = torch.zeros_like(obj.segment)

        target_y_offset_b = target_grid_y * H_im // grid_h
        target_x_offset_b = target_grid_x * W_im // grid_w
        target_y_offset_e = (target_grid_y + 1) * H_im // grid_h
        target_x_offset_e = (target_grid_x + 1) * W_im // grid_w
        target_H_im_downsize = target_y_offset_e - target_y_offset_b
        target_W_im_downsize = target_x_offset_e - target_x_offset_b

        segment_downsize = F.resize(
            obj.segment[None, None],
            size=(target_H_im_downsize, target_W_im_downsize),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,  # antialiasing for downsizing
        )[0, 0]
        if should_hflip[target_grid_y, target_grid_x].item():
            segment_downsize = F.hflip(segment_downsize[None, None])[0, 0]

        segment_output[
            target_y_offset_b:target_y_offset_e, target_x_offset_b:target_x_offset_e
        ] = segment_downsize
        obj.segment = segment_output

    return datapoint


class RandomMosaicVideoAPI:
    def __init__(self, prob=0.15, grid_h=2, grid_w=2, use_random_hflip=False):
        self.prob = prob
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.use_random_hflip = use_random_hflip

    def __call__(self, frames, **kwargs):
        if random.random() > self.prob:
            return datapoint

        # select a random location to place the target mask in the mosaic
        target_grid_y = random.randint(0, self.grid_h - 1)
        target_grid_x = random.randint(0, self.grid_w - 1)
        # whether to flip each grid in the mosaic horizontally
        if self.use_random_hflip:
            should_hflip = torch.rand(self.grid_h, self.grid_w) < 0.5
        else:
            should_hflip = torch.zeros(self.grid_h, self.grid_w, dtype=torch.bool)
        for i in range(len(datapoint.frames)):
            datapoint = random_mosaic_frame(
                datapoint,
                i,
                grid_h=self.grid_h,
                grid_w=self.grid_w,
                target_grid_y=target_grid_y,
                target_grid_x=target_grid_x,
                should_hflip=should_hflip,
            )

        return datapoint
'''