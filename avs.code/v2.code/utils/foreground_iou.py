import numpy
import torch


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def get_entire_dict_for_ddp_calculation(self):
        return self.__data

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v


class ForegroundIoU(AverageMeter):
    def __init__(self):
        super(ForegroundIoU, self).__init__('foreground_iou')

    def calculate_iou(self, pred, target, eps=1e-7, get_entire_list=False):
        r"""
            param (both hard mask):
                pred: size [N x H x W], type: int
                target: size [N x H x W], type: int
            output:
                iou: size [1] (size_average=True) or [N] (size_average=False)
        """
        assert len(pred.shape) == 3 and pred.shape == target.shape, 'shape mismatch.'
        assert pred.dtype is torch.long and target.dtype is torch.long, 'type mismatch.'

        N = pred.size(0)
        num_pixels = pred.size(-1) * pred.size(-2)
        no_obj_flag = (target.sum(2).sum(1) == 0)

        inter = (pred * target).sum(2).sum(1)
        union = torch.max(pred, target).sum(2).sum(1)

        inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
        inter[no_obj_flag] = inter_no_obj[no_obj_flag]
        union[no_obj_flag] = num_pixels

        iou = torch.sum(inter / (union+eps)) / N

        self.add({'foreground_iou': iou})
        return self.get('foreground_iou') if not get_entire_list else self.get_entire_dict_for_ddp_calculation()

    def reset(self,):
        super(ForegroundIoU, self).__init__('foreground_iou')

