import random

import matplotlib.pyplot as plt
import numpy
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


class Augmentation(object):
    def __init__(self, image_mean, image_std, image_width, image_height, scale_list, ignore_index=255):
        self.image_size = (image_height, image_width)
        # self.image_norm = (image_mean, image_std)
        # self.get_crop_pos = transforms.RandomCrop(self.image_size)
        self.color_jitter = transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.25)
        self.gaussian_blurring = transforms.GaussianBlur((3, 3))
        self.scale_list = scale_list

        self.normalise = transforms.Normalize(mean=image_mean, std=image_std)
        self.to_tensor = transforms.ToTensor()

        self.ignore_index = ignore_index

        # self.normalise = transforms.Normalize(mean=image_mean, std=image_std)

        # if setup == "avs" or setup == "avss" or setup == "avss_binary":
        #     # AVS
        #     self.scale_list = [.5, .75, 1.]
        #     self.color_jitter = None
        # else:
        #     # COCO
        #     # self.scale_list = [.75, 1., 1.25, 1.5, 1.75, 2.]
        #     self.scale_list = [0.5,0.75,1.0,1.25,1.5,1.75,2.0]

    # def normalise(self, image):
    #     image = image / 255.0
    #     image = image - self.image_norm[0]
    #     image = image / self.image_norm[1]
    #     return image

    def resize(self, image_, label_, size=None):
        h_, w_ = self.image_size if size is None else size
        image_ = F.resize(image_, (h_, w_), transforms.InterpolationMode.BICUBIC)
        label_ = F.resize(label_, (h_, w_), transforms.InterpolationMode.NEAREST)
        return image_, label_

    def random_crop_with_padding(self, image_, label_):
        w_, h_ = image_.size
        if min(h_, w_) < min(self.image_size):
            res_w_ = max(self.image_size[0] - w_, 0)
            res_h_ = max(self.image_size[1] - h_, 0)
            image_ = F.pad(image_, [0, 0, res_w_, res_h_], fill=(numpy.array(self.image_norm[0]) * 255.).tolist())
            # image_ = F.pad(image_, [0, 0, res_w_, res_h_], fill=self.ignore_index) # if error, define the padding value.
            label_ = F.pad(label_, [0, 0, res_w_, res_h_], fill=self.ignore_index)

        pos_ = self.get_crop_pos.get_params(image_, self.image_size)
        image_ = F.crop(image_, *pos_)
        label_ = F.crop(label_, *pos_)

        return image_, label_

    # @staticmethod
    def random_scales(self, image_, label_):
        w_, h_ = image_.size
        chosen_scale = random.choice(self.scale_list)
        w_, h_ = int(w_ * chosen_scale), int(h_ * chosen_scale)
        image_ = F.resize(image_, (h_, w_), transforms.InterpolationMode.BICUBIC)
        label_ = F.resize(label_, (h_, w_), transforms.InterpolationMode.NEAREST)
        return image_, label_

    @staticmethod
    def random_flip_h(image_, label_):
        chosen_flip = random.random() > 0.5
        image_ = F.hflip(image_) if chosen_flip else image_
        label_ = F.hflip(label_) if chosen_flip else label_
        return image_, label_

    def augment_entire_clip(self, x_list, y_list):
        degree_ = float(torch.empty(1).uniform_(float(-25.), float(25.)).item())
        shear_ = [float(torch.empty(1).uniform_(float(-20.), float(20.)).item()),
                 torch.empty(1).uniform_(float(-20.), float(20.)).item()]
        dice =  random.random()
        for index, single_x in enumerate(x_list):
            if dice <= 0.1:
                single_x = F.rgb_to_grayscale(single_x, num_output_channels=3)
            
            single_x = F.affine(single_x, angle=degree_, shear=shear_, translate=[0,0], scale=1.,
                               interpolation=transforms.InterpolationMode.BILINEAR, fill=[0., 0., 0.])
            single_y = F.affine(y_list[index], angle=degree_, shear=shear_, translate=[0,0], scale=1.,
                               interpolation=transforms.InterpolationMode.NEAREST, fill=[0.])
            x_list[index] = single_x
            y_list[index] = single_y

        return x_list, y_list




    def train_aug(self, x_, y_):
        x_, y_ = self.random_flip_h(x_, y_)
        # # x, y = self.random_scales(x, y)
        x_, y_ = self.resize(x_, y_)

        if self.color_jitter is not None and random.random() < 0.5:
            x_ = self.color_jitter(x_)
        if self.gaussian_blurring is not None and random.random() < 0.5:
            x_ = self.gaussian_blurring(x_)

        # x, y = self.random_crop_with_padding(x, y)

        x_ = self.normalise(self.to_tensor(x_)).type(torch.float32)
        # receive pseudo labels.
        y_ = torch.tensor(numpy.array(y_)[numpy.newaxis, ...], dtype=torch.float)
        return x_, y_

    def test_process(self, x_, y_):
        # x = self.to_tensor(x)
        # y = torch.tensor(numpy.asarray(y)).long()

        # following AVSbench setup, we fix image size (224, 224)
        x_, y_ = self.resize(x_, y_)

        x_ = self.normalise(self.to_tensor(x_)).type(torch.float32)
        y_ = torch.tensor(numpy.array(y_)[numpy.newaxis, ...], dtype=torch.float)
        return x_, y_

    def __call__(self, x, y, split):
        return self.train_aug(x, y) if split == "train" \
            else self.test_process(x, y)











