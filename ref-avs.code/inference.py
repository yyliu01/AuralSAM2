"""Distributed inference on Ref-AVS (test_s / test_u / test_n); uses Trainer.valid / valid_null like main.py."""
import os
import pathlib
import argparse
import random

import numpy
import torch
from easydict import EasyDict


_real_mkdir = pathlib.Path.mkdir


def _safe_mkdir(self, mode=0o777, parents=False, exist_ok=False):
    try:
        return _real_mkdir(self, mode, parents, exist_ok=exist_ok)
    except PermissionError:
        pass


pathlib.Path.mkdir = _safe_mkdir


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


class _DummyTensorboard:
    """Minimal Tensorboard stub so Trainer.valid / valid_null run without wandb logging."""

    def upload_wandb_info(self, info_dict):
        pass

    def upload_wandb_image(self, *args, **kwargs):
        pass


def main(local_rank, ngpus_per_node, hyp_param):
    hyp_param.local_rank = local_rank
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=hyp_param.local_rank,
        world_size=hyp_param.gpus,
    )
    seed_it(local_rank + hyp_param.seed)
    torch.cuda.set_device(hyp_param.local_rank)

    import model.visual.sam2  # noqa: F401 — registers Hydra config store
    from hydra import compose
    from omegaconf import OmegaConf

    arch_h = compose(config_name='configs/auralfuser/architecture.yaml')
    OmegaConf.resolve(arch_h)
    hyp_param.aural_fuser = OmegaConf.to_container(arch_h.aural_fuser, resolve=True)

    train_cfg = compose(config_name='configs/training/sam2_training_config.yaml')
    OmegaConf.resolve(train_cfg)
    hyp_param.contrastive_learning = OmegaConf.to_container(train_cfg.contrastive_learning, resolve=True)

    hyp_param.image_size = 1024
    hyp_param.image_embedding_size = int(hyp_param.image_size / 16)

    from model.mymodel import AVmodel
    av_model = AVmodel(hyp_param).cuda(hyp_param.local_rank)
    if not hyp_param.inference_ckpt:
        raise ValueError("--inference_ckpt is required for inference.")

    ckpt_sd = torch.load(hyp_param.inference_ckpt, map_location="cpu")
    if not isinstance(ckpt_sd, dict):
        raise TypeError("Checkpoint must be a state_dict dictionary.")
    if any(k.startswith("v_model.") or k.startswith("aural_fuser.") for k in ckpt_sd):
        av_model.load_state_dict(ckpt_sd, strict=True)
    else:
        av_model.aural_fuser.load_state_dict(ckpt_sd, strict=True)

    av_model = torch.nn.parallel.DistributedDataParallel(
        av_model, device_ids=[hyp_param.local_rank], find_unused_parameters=False,
    )
    av_model.eval()

    from dataloader.dataset import AV
    from dataloader.visual.visual_augmentation import Augmentation as VisualAugmentation
    from dataloader.audio.audio_augmentation import Augmentation as AudioAugmentation
    from torch.utils.data import DataLoader, Subset
    from torch.utils.data.distributed import DistributedSampler

    visual_aug = VisualAugmentation(
        hyp_param.image_mean, hyp_param.image_std,
        hyp_param.image_size, hyp_param.image_size,
        hyp_param.scale_list, ignore_index=hyp_param.ignore_index,
    )
    audio_aug = AudioAugmentation(mono=True)

    max_batches = getattr(hyp_param, "inference_max_batches", 0) or 0
    val_batch_size = getattr(hyp_param, "inference_val_batch_size", 4)

    def _test_loader(split):
        ds = AV(
            split=split,
            augmentation={"visual": visual_aug, "audio": audio_aug},
            param=hyp_param,
            root_path=hyp_param.data_root_path,
        )
        if max_batches > 0:
            n_samples = min(max_batches * val_batch_size, len(ds))
            ds = Subset(ds, range(n_samples))
        sampler = DistributedSampler(ds, shuffle=False)
        return DataLoader(
            ds,
            batch_size=val_batch_size,
            sampler=sampler,
            num_workers=hyp_param.num_workers,
        )

    test_s_loader = _test_loader('test_s')
    test_u_loader = _test_loader('test_u')
    test_n_loader = _test_loader('test_n')

    from trainer.train import Trainer
    from utils.foreground_iou import ForegroundIoU
    from utils.foreground_fscore import ForegroundFScore
    from utils.foreground_s import ForegroundS

    metrics = {
        "foreground_iou": ForegroundIoU(),
        "foreground_f-score": ForegroundFScore(hyp_param.local_rank),
        "foreground_s": ForegroundS(),
    }
    trainer = Trainer(hyp_param, loss=None, tensorboard=_DummyTensorboard(), metrics=metrics)

    test_s_iou, test_s_f = trainer.valid(
        epoch=0, dataloader=test_s_loader, model=av_model, process='test_s',
    )
    torch.cuda.empty_cache()

    test_u_iou, test_u_f = trainer.valid(
        epoch=0, dataloader=test_u_loader, model=av_model, process='test_u',
    )
    torch.cuda.empty_cache()

    test_n_s = trainer.valid_null(
        epoch=0, dataloader=test_n_loader, model=av_model, process='test_n',
    )
    torch.cuda.empty_cache()

    if hyp_param.local_rank <= 0:
        print("\n========== Ref-AVS inference (same splits / metrics as training valid) ==========")
        print("  test_s   f_iou={}  f_f-score={}".format(test_s_iou, test_s_f))
        print("  test_u   f_iou={}  f_f-score={}".format(test_u_iou, test_u_f))
        print("  test_n   f_s={}".format(test_n_s))
        print("=======================================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ref-AVS inference: test_s / test_u / test_n')

    parser.add_argument('--local_rank', type=int, default=-1,
                        help='multi-process training for DDP')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='unused at inference (validation uses inference_val_batch_size)')
    parser.add_argument('--epochs', default=80, type=int, help='unused')
    parser.add_argument('--lr', default=1e-5, type=float, help='unused')
    parser.add_argument('--online', action='store_true', help='unused')
    parser.add_argument(
        '--inference_ckpt', type=str, required=True,
        help='Trained AuralFuser checkpoint (.pth). SAM2 from backbone_weight in configs.',
    )
    parser.add_argument('--inference_max_batches', type=int, default=0,
                        help='0 = full split; >0 = first N batches per split (debug)')
    parser.add_argument('--inference_val_batch_size', type=int, default=4,
                        help='Validation batch size (default 4, same as main.py _test_loader)')

    args = parser.parse_args()

    from configs.config import C
    args = EasyDict({**C, **vars(args)})

    _repo = pathlib.Path(__file__).resolve().parent
    _workspace = _repo.parent
    args.data_root_path = str(_workspace / 'REFAVS')
    args.backbone_weight = str(_workspace / 'ckpts' / 'sam_ckpts' / 'sam2_hiera_large.pt')
    args.audio.PRETRAINED_VGGISH_MODEL_PATH = str(_workspace / 'ckpts' / 'vggish-10086976.pth')
    args.saved_dir = '/tmp/ref_avs_infer_ckpt'
    pathlib.Path(args.saved_dir).mkdir(parents=True, exist_ok=True)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9902'

    torch.multiprocessing.spawn(main, nprocs=args.gpus, args=(args.gpus, args))
