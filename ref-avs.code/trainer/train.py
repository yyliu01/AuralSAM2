"""Training and validation for Ref-AVS (text + audio + SAM2 multimask decoding)."""
import numpy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_DECODE_MODES = frozenset({'', 'iou_select', 'iou_occ_select'})


def _decode_mode_and_wandb_tag(process):
    """Match tmp.code: `process` is decode mode for known strings; else Ref split tag + default decode."""
    if process in _DECODE_MODES:
        return process, process
    return 'iou_select', process


class Trainer:
    """Train / valid / null-valid steps with composite loss, contrastive term, and metrics."""

    def __init__(self, hyp_param, loss, tensorboard, metrics):
        self.param = hyp_param
        self.loss = loss
        self.tensorboard = tensorboard
        self.metrics = metrics
        from loss.training.contrastive_learning import ContrastLoss
        self.cl = ContrastLoss(self.param)

    @torch.no_grad()
    def valid_null(self, epoch, dataloader, model, process='test_n'):
        if not isinstance(dataloader, DataLoader):
            raise TypeError("valid_null() expects a torch.utils.data.DataLoader (do not pass iter(dataloader) first).")
        decode_mode, wandb_tag = _decode_mode_and_wandb_tag(process)
        self.metrics['foreground_s'].reset()
        dataloader_length = len(dataloader)
        tbar = range(dataloader_length)
        tbar = tqdm(tbar, ncols=135) if self.param.local_rank <= 0 else tbar
        p_pool = [None] * self.param.gpus
        n_pool = [None] * self.param.gpus

        data_iter = iter(dataloader)
        for _ in tbar:
            items = next(data_iter)
            frame, spect, prompt_dicts = items['frame'], items['spectrogram'], items['text']
            logits = []
            for frame_, spect_, prompt_dicts_ in zip(frame, spect, prompt_dicts):
                frame_ = frame_.cuda(self.param.local_rank, non_blocking=True)
                spect_ = spect_.cuda(self.param.local_rank, non_blocking=True)
                prompt_dicts_ = [prompt_dicts_]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs, _ = model.module(frame_, spect_, prompt_dicts_, sam_process=False)

                logits_ = torch.cat([torch.cat(i['multistep_pred_multimasks_high_res']) for i in outputs])
                ious_scores = torch.cat([torch.cat(i['multistep_pred_ious']) for i in outputs])
                occ_scores = torch.cat([torch.cat(i['multistep_object_score_logits']) for i in outputs])
                if decode_mode == 'iou_select':
                    ious_scores = torch.argmax(ious_scores, dim=1)
                    logits_ = logits_[torch.arange(0, frame_.shape[0]), ious_scores, ...]
                elif decode_mode == 'iou_occ_select':
                    ious_scores = torch.argmax(ious_scores, dim=1)
                    logits_ = logits_[torch.arange(0, frame_.shape[0]), ious_scores, ...]
                    logits_[occ_scores.squeeze() < 0, ...] = 0.
                else:
                    logits_ = logits_[:, 0, ...]
                logits.append(logits_)

            logits = torch.cat(logits).reshape(frame.shape[0], -1, self.param.image_size, self.param.image_size)
            if len(logits.shape) == 3:
                logits = logits.unsqueeze(1)

            foreground_s = self.metrics['foreground_s'].metric_s_for_null(logits, get_entire_list=True)
            torch.distributed.all_gather_object(p_pool, foreground_s['foreground_p'])
            torch.distributed.all_gather_object(n_pool, foreground_s['foreground_n'])
            foreground_s = sum([i[0].cpu() for i in p_pool]) / sum([i[0] for i in n_pool])

            if self.param.local_rank <= 0:
                tbar.set_description(
                    'epoch {} | valid.null_s {}'.format(epoch, numpy.round(foreground_s, 5)),
                )
            torch.cuda.empty_cache()

        final_s = foreground_s
        if self.param.local_rank <= 0 and self.tensorboard is not None:
            self.tensorboard.upload_wandb_info({"valid.f_s/{}".format(wandb_tag): final_s})

        return numpy.round(final_s, 5)

    @torch.no_grad()
    def valid(self, epoch, dataloader, model, process='iou_select'):
        """Evaluate IoU / F-score; `process` is decode mode (tmp) or split tag (test_s / test_u). Wandb keys like tmp."""
        if not isinstance(dataloader, DataLoader):
            raise TypeError("valid() expects a torch.utils.data.DataLoader (do not pass iter(dataloader) first).")
        decode_mode, wandb_tag = _decode_mode_and_wandb_tag(process)
        self.metrics['foreground_iou'].reset()
        self.metrics['foreground_f-score'].reset()
        dataloader_length = len(dataloader)
        tbar = range(dataloader_length)
        tbar = tqdm(tbar, ncols=135) if self.param.local_rank <= 0 else tbar
        iou_pool = [None] * self.param.gpus
        fscore_pool = [None] * self.param.gpus

        data_iter = iter(dataloader)
        for _ in tbar:
            items = next(data_iter)
            frame, spect, label, prompt_dicts = (
                items['frame'], items['spectrogram'], items['label'], items['text']
            )
            logits = []
            labels = []
            for frame_, spect_, label_, prompt_dicts_ in zip(frame, spect, label, prompt_dicts):
                frame_ = frame_.cuda(self.param.local_rank, non_blocking=True)
                spect_ = spect_.cuda(self.param.local_rank, non_blocking=True)
                label_ = label_.cuda(self.param.local_rank, non_blocking=True)
                prompt_dicts_ = [prompt_dicts_]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs, _ = model.module(frame_, spect_, prompt_dicts_, sam_process=False)

                logits_ = torch.cat([torch.cat(i['multistep_pred_multimasks_high_res']) for i in outputs])
                ious_scores = torch.cat([torch.cat(i['multistep_pred_ious']) for i in outputs])
                occ_scores = torch.cat([torch.cat(i['multistep_object_score_logits']) for i in outputs])
                if decode_mode == 'iou_select':
                    ious_scores = torch.argmax(ious_scores, dim=1)
                    logits_ = logits_[torch.arange(0, frame_.shape[0]), ious_scores, ...]
                elif decode_mode == 'iou_occ_select':
                    ious_scores = torch.argmax(ious_scores, dim=1)
                    logits_ = logits_[torch.arange(0, frame_.shape[0]), ious_scores, ...]
                    logits_[occ_scores.squeeze() < 0, ...] = 0.
                else:
                    logits_ = logits_[:, 0, ...]
                logits.append(logits_)
                labels.append(label_)

            logits = torch.cat(logits)
            labels = torch.cat(labels)
            foreground_iou_rank = self.metrics['foreground_iou'].calculate_iou(
                (logits > 0.).squeeze().long(), labels.squeeze().long(), get_entire_list=True,
            )
            foreground_f_score_rank = self.metrics['foreground_f-score'].calculate_f_score(
                logits.squeeze(), labels.squeeze().long(), get_entire_list=True,
            )
            torch.distributed.all_gather_object(iou_pool, foreground_iou_rank)
            torch.distributed.all_gather_object(fscore_pool, foreground_f_score_rank)
            foreground_iou = sum([i['foreground_iou'][0].cpu() for i in iou_pool]) / sum(
                [i['foreground_iou'][1] for i in iou_pool])
            foreground_f_score = sum([i['foreground_f-score'][0] for i in fscore_pool]) / sum(
                [i['foreground_f-score'][1] for i in fscore_pool])

            if self.param.local_rank <= 0:
                tbar.set_description(
                    'epoch {} | valid.f_iou {}, valid.f_f-score {}'.format(
                        epoch,
                        numpy.round(foreground_iou.cpu().numpy(), 5),
                        numpy.round(foreground_f_score, 5),
                    ),
                )
            torch.cuda.empty_cache()

        final_iou = foreground_iou
        final_fscore = foreground_f_score
        if self.param.local_rank <= 0 and self.tensorboard is not None:
            self.tensorboard.upload_wandb_info({
                "valid.f_iou/{}".format(wandb_tag): final_iou,
                "valid.f_f-score/{}".format(wandb_tag): final_fscore,
            })

        def _to_float(x):
            if isinstance(x, torch.Tensor):
                return float(x.detach().cpu().item())
            return float(x)

        return numpy.round(_to_float(final_iou), 5), numpy.round(_to_float(final_fscore), 5)

    def train(self, epoch, dataloader, model, optimiser):
        if not isinstance(dataloader, DataLoader):
            raise TypeError("train() expects a torch.utils.data.DataLoader (do not pass iter(dataloader) first).")
        dataloader_length = len(dataloader)
        tbar = range(dataloader_length)
        tbar = tqdm(tbar, ncols=135) if self.param.local_rank <= 0 else tbar

        data_iter = iter(dataloader)
        for batch_index in tbar:
            current_index = dataloader_length * epoch + batch_index
            items = next(data_iter)
            frame, spect, label, prompt_dicts = (
                items['frame'], items['spectrogram'], items['label'], items['text'],
            )
            frame = torch.flatten(frame, start_dim=0, end_dim=1).cuda(self.param.local_rank, non_blocking=True)
            spect = torch.flatten(spect, start_dim=0, end_dim=1).cuda(self.param.local_rank, non_blocking=True)
            label = torch.flatten(label, start_dim=0, end_dim=1).cuda(self.param.local_rank, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs, proj_feats = model(frame, spect, prompt_dicts, sam_process=False)
            loss_dict = self.loss(outputs, label.unsqueeze(1))
            cl_loss = self.cl(proj_feats, outputs, label)

            optimiser.zero_grad()
            (loss_dict['core_loss'] + cl_loss).backward()
            optimiser.step()

            current_lr = self.param.lr * (1 - current_index / (dataloader_length * self.param.epochs)) ** 0.9
            for params_lr in optimiser.param_groups:
                names = params_lr.get("name", [])
                if names and any("vgg" in n for n in names):
                    params_lr['lr'] = current_lr * 0.1
                else:
                    params_lr['lr'] = current_lr

            if self.param.local_rank <= 0 and self.tensorboard is not None:
                logits = torch.cat([i['multistep_pred_multimasks_high_res'][0] for i in outputs])
                foreground_iou = self.metrics['foreground_iou'].calculate_iou(
                    (logits > 0)[:, 0, ...].long(), label.long(),
                )
                self.tensorboard.upload_wandb_info({
                    "loss": loss_dict['core_loss'].item(), "f_iou": foreground_iou.item(),
                    "lr": optimiser.param_groups[0]['lr'],
                    "loss_dice": loss_dict['loss_dice'],
                    "loss_focal": loss_dict['loss_mask'],
                    "loss_contras": cl_loss.item(),
                })
                tbar.set_description(
                    'epoch {} | loss {}, f_iou {}'.format(
                        epoch, loss_dict['core_loss'].item(), foreground_iou.item(),
                    ),
                )
        return
