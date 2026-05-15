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


class ForegroundFScore(AverageMeter):
    def __init__(self, rank):
        self.local_rank = rank
        super(ForegroundFScore, self).__init__('foreground_f-score')

    def _eval_pr(self, y_pred, y, num, cuda_flag=True):
        if cuda_flag:
            prec, recall = torch.zeros(num).cuda(self.local_rank), torch.zeros(num).cuda(self.local_rank)
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda(self.local_rank)
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall

    def calculate_f_score(self, pred, gt, pr_num=255,  get_entire_list=False):

        r"""
            param:
                pred: size [N x H x W]
                gt: size [N x H x W]
            output:
                iou: size [1] (size_average=True) or [N] (size_average=False)
        """
        # print('=> eval [FMeasure]..')
        pred = torch.sigmoid(pred) # =======================================[important]
        N = pred.size(0)
        beta2 = 0.3
        avg_f, img_num = 0.0, 0
        score = torch.zeros(pr_num)
        # fLog = open(os.path.join(measure_path, 'FMeasure.txt'), 'w')
        # print("{} videos in this batch".format(N))

        for img_id in range(N):
            # examples with totally black GTs are out of consideration
            if torch.mean(gt[img_id].float()) == 0.0:
                continue
            prec, recall = self._eval_pr(pred[img_id], gt[img_id], pr_num)
            f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
            f_score[f_score != f_score] = 0 # for Nan
            avg_f += f_score
            img_num += 1
            score = avg_f / img_num
            # print('score: ', score)
        # fLog.close()
        self.add({'foreground_f-score': score.max().item()})
        return self.get('foreground_f-score') if not get_entire_list else self.get_entire_dict_for_ddp_calculation()

    def reset(self,):
        super(ForegroundFScore, self).__init__('foreground_f-score')


