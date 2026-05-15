import os

import PIL
import matplotlib.pyplot as plt
import numpy
import torch
import torchvision
try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

# from utils.visualize import show_img


color_map = {"background": (0, 0, 0), "longitudinal": (128, 0, 0), "pothole": (0, 128, 0),
             "alligator": (128, 128, 0),  "transverse": (128, 0, 128), "ignore": (255, 255, 255)}


class _DummyWandb:
    def log(self, *args, **kwargs):
        return None


class Tensorboard:
    def __init__(self, config):
        self._log_images = bool(config.get('wandb_online', False))
        if not self._log_images or wandb is None or not hasattr(wandb, "init"):
            self.tensor_board = _DummyWandb()
            self._log_images = False
        elif config.get('wandb_online', False):
            key = config.get('wandb_key') or os.environ.get('WANDB_API_KEY', '')
            if key:
                os.environ['WANDB_API_KEY'] = key
                wandb.login(key=key, relogin=False)
            self.tensor_board = wandb.init(project=config['proj_name'], name=config['experiment_name'],
                                           config=config, settings=wandb.Settings(code_dir="."))

        self.restore_transform = torchvision.transforms.Compose([
            DeNormalize(config['image_mean'], config['image_std']),
            torchvision.transforms.ToPILImage()])

    def upload_wandb_info(self, info_dict):
        for i, info in enumerate(info_dict):
            self.tensor_board.log({info: info_dict[info]})
        return


    def upload_wandb_image(self, frames, pseudo_label_from_pred, pseudo_label_from_sam, img_number=4):
        if not self._log_images:
            return

        def _batched_rgb(t):
            """[N,C,H,W] or [C,H,W] float tensor on CPU."""
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t)
            t = t.detach().cpu().float()
            if t.dim() == 3:
                return t.unsqueeze(0)
            if t.dim() == 4:
                return t
            raise ValueError("frames must be [C,H,W] or [N,C,H,W], got shape {}".format(tuple(t.shape)))

        def _batched_mask(t):
            """[N,H,W] or [N,1,H,W] or [H,W]."""
            if not isinstance(t, torch.Tensor):
                t = torch.as_tensor(t)
            t = t.detach().cpu().float()
            while t.dim() > 3:
                t = t.squeeze(1)
            if t.dim() == 2:
                t = t.unsqueeze(0)
            if t.dim() != 3:
                raise ValueError("masks must be [H,W], [N,H,W] or [N,1,H,W], got shape {}".format(tuple(t.shape)))
            return t

        frames = _batched_rgb(frames)
        pseudo_label_from_pred = _batched_mask(pseudo_label_from_pred)
        pseudo_label_from_sam = _batched_mask(pseudo_label_from_sam)

        n = min(frames.shape[0], pseudo_label_from_pred.shape[0], pseudo_label_from_sam.shape[0], img_number)
        frames = frames[:n]
        pseudo_label_from_pred = pseudo_label_from_pred[:n]
        pseudo_label_from_sam = pseudo_label_from_sam[:n]

        pseudo_label_from_sam = pseudo_label_from_sam.clone()
        pseudo_label_from_pred = pseudo_label_from_pred.clone()
        pseudo_label_from_sam[pseudo_label_from_sam == 255.] = 0.5
        pseudo_label_from_pred[pseudo_label_from_pred == 255.] = 0.5

        denorm = self.restore_transform.transforms[0]
        image_list = []
        label_list = []
        logits_list = []
        for i in range(n):
            fi = frames[i].clone()
            if fi.shape[0] == 3:
                denorm(fi)
                fi.clamp_(0.0, 1.0)
            image_list.append(wandb.Image(fi, caption="id {}".format(str(i))))
            # wandb.Image expects torch tensors as [C, H, W] (it permutes CHW→HWC)
            ms = pseudo_label_from_sam[i].squeeze()
            mp = pseudo_label_from_pred[i].squeeze()
            if ms.dim() == 2:
                ms = ms.unsqueeze(0)
            if mp.dim() == 2:
                mp = mp.unsqueeze(0)
            label_list.append(wandb.Image(ms, caption="id {}".format(str(i))))
            logits_list.append(wandb.Image(mp, caption="id {}".format(str(i))))

        self.tensor_board.log({"image": image_list, "label": label_list, "logits": logits_list})

    def de_normalize(self, image):
        return [self.restore_transform(i.detach().cpu()) if (isinstance(i, torch.Tensor) and len(i.shape) == 3)
                else colorize_mask(i.detach().cpu().numpy(), self.palette)
                for i in image]

    def finish(self):
        self.tensor_board.finish()


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    # palette[-6:-3] = [183, 65, 14]
    new_mask = PIL.Image.fromarray(mask.astype(numpy.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask
