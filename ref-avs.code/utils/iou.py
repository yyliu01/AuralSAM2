import torch
import numpy


class BinaryMIoU(object):
    def __init__(self, ignore_index):
        self.num_classes = 2
        self.ignore_index = ignore_index
        self.inter, self.union = 0, 0
        self.correct, self.label = 0, 0
        self.iou = numpy.array([0 for _ in range(self.num_classes)])
        self.acc = 0.0

    def get_metric_results(self, curr_correct_, curr_label_, curr_inter_, curr_union_):
        # calculates the overall miou and acc
        self.correct = self.correct + curr_correct_
        self.label = self.label + curr_label_
        self.inter = self.inter + curr_inter_
        self.union = self.union + curr_union_
        self.acc = 1.0 * self.correct / (numpy.spacing(1) + self.label)
        self.iou = 1.0 * self.inter / (numpy.spacing(1) + self.union)
        return numpy.round(self.iou, 4), numpy.round(self.acc, 4)
        # if class_list is None:
        #     return numpy.round(self.iou.mean().item(), 4), \
        #         numpy.round(self.acc, 4)
        # else:
        #     return numpy.round(self.iou[class_list].mean().item(), 4), \
        #         numpy.round(self.acc, 4)

    @staticmethod
    def get_current_image_results(curr_correct_, curr_label_, curr_inter_, curr_union_):
        curr_acc = 1.0 * curr_correct_ / (numpy.spacing(1) + curr_label_)
        curr_iou = 1.0 * curr_inter_ / (numpy.spacing(1) + curr_union_)
        return curr_iou, curr_acc

    def __call__(self, x, y):
        curr_correct, curr_label, curr_inter, curr_union = self.calculate_current_sample(x, y)
        return (self.get_metric_results(curr_correct, curr_label, curr_inter, curr_union),
                self.get_current_image_results(curr_correct, curr_label, curr_inter, curr_union))

    def calculate_current_sample(self, output, target):
        # output => BxCxHxW (logits)
        # target => Bx1xHxW
        target[target == self.ignore_index] = -1
        correct, labeled = self.batch_pix_accuracy(output.data, target)
        inter, union = self.batch_intersection_union(output.data, target, self.num_classes)
        return [numpy.round(correct, 5), numpy.round(labeled, 5), numpy.round(inter, 5), numpy.round(union, 5)]

    @ staticmethod
    def batch_pix_accuracy(predict, target):
        # _, predict = torch.max(output, 1)

        predict = predict.int() + 1
        target = target.int() + 1

        pixel_labeled = (target > 0).sum()
        pixel_correct = ((predict == target) * (target > 0)).sum()
        assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
        return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

    @ staticmethod
    def batch_intersection_union(predict, target, num_class):
        # _, predict = torch.max(output, 1)
        predict = predict + 1
        target = target + 1

        predict = predict * (target > 0).long()
        intersection = predict * (predict == target).long()

        area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
        area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
        area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
        return area_inter.cpu().numpy(), area_union.cpu().numpy()

