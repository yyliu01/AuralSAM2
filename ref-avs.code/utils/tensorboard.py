"""Optional Weights & Biases logging for Ref-AVS training."""
import os

import torchvision
import wandb


class Tensorboard:
    def __init__(self, config):
        key = config.get('wandb_key') or os.environ.get('WANDB_API_KEY', '')
        if key:
            os.environ['WANDB_API_KEY'] = key
        mode = 'online' if config.get('wandb_online', False) else 'disabled'
        self.tensor_board = wandb.init(
            project=config['proj_name'],
            name=config['experiment_name'],
            config=config,
            mode=mode,
            settings=wandb.Settings(code_dir='.'),
        )
        self.restore_transform = torchvision.transforms.Compose([
            DeNormalize(config['image_mean'], config['image_std']),
            torchvision.transforms.ToPILImage(),
        ])

    def upload_wandb_info(self, info_dict):
        for key, value in info_dict.items():
            self.tensor_board.log({key: value})

    def upload_wandb_image(self, frames, pseudo_label_from_pred, pseudo_label_from_sam, img_number=4):
        n = min(pseudo_label_from_pred.shape[0], img_number)
        frames = frames[:n]
        pseudo_label_from_sam = pseudo_label_from_sam[:n].float()
        pseudo_label_from_pred = pseudo_label_from_pred[:n].float()
        pseudo_label_from_sam[pseudo_label_from_sam == 255.] = 0.5
        pseudo_label_from_pred[pseudo_label_from_pred == 255.] = 0.5
        self.tensor_board.log({
            'image': [wandb.Image(j, caption=f'id {i}') for i, j in enumerate(frames)],
            'label': [wandb.Image(j.squeeze(), caption=f'id {i}') for i, j in enumerate(pseudo_label_from_sam)],
            'logits': [wandb.Image(j.squeeze(), caption=f'id {i}') for i, j in enumerate(pseudo_label_from_pred)],
        })

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
