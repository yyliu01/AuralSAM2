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


class ForegroundS(AverageMeter):
    def __init__(self):
        super(ForegroundS, self).__init__('foreground_p', 'foreground_n')

    def metric_s_for_null(self, pred, get_entire_list=False):
        NF, bsz, H, W = pred.shape
        pred = pred.view(NF * bsz, H, W)
        assert len(pred.shape) == 3

        N = pred.size(0)
        num_pixels = pred.view(-1).shape[0]

        temp_pred = torch.sigmoid(pred)
        pred = (temp_pred > 0.5).int()

        x = torch.sum(pred.view(-1))
        s = torch.sqrt(x / num_pixels)

        self.add({'foreground_p': x})
        self.add({'foreground_n': num_pixels})
        # self.add({'foreground_s': s})
        return self.get('foreground_p')/self.get('foreground_n') if not get_entire_list else self.get_entire_dict_for_ddp_calculation()

    def reset(self, ):
        super(ForegroundS, self).__init__('foreground_p', 'foreground_n')


