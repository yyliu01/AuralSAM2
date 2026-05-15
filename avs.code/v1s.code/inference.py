"""Distributed inference on the test set; runs the same three `process` modes as training validation."""
import os
import pathlib
import torch
import numpy
import random
import argparse
from easydict import EasyDict

# Avoid import failure when configs.config creates saved_dir without write permission.
_real_mkdir = pathlib.Path.mkdir


def _safe_mkdir(self, mode=0o777, parents=False, exist_ok=False):
    try:
        return _real_mkdir(self, mode, parents=parents, exist_ok=exist_ok)
    except PermissionError:
        pass


pathlib.Path.mkdir = _safe_mkdir


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    numpy.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


class _DummyTensorboard:
    """Minimal Tensorboard stub so Trainer.valid runs without wandb logging."""

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
        world_size=hyp_param.gpus * 1
    )
    seed_it(local_rank + hyp_param.seed)

    import model.visual.sam2  # noqa: F401 — registers Hydra `configs`
    from hydra import compose
    from omegaconf import OmegaConf

    arch_h = compose(config_name='auralfuser/architecture.yaml')
    OmegaConf.resolve(arch_h)
    hyp_param.aural_fuser = OmegaConf.to_container(arch_h.aural_fuser, resolve=True)

    train_cfg = compose(config_name='training/sam2_training_config.yaml')
    OmegaConf.resolve(train_cfg)
    hyp_param.contrastive_learning = OmegaConf.to_container(train_cfg.contrastive_learning, resolve=True)

    from model.mymodel import AVmodel
    av_model = AVmodel(hyp_param).cuda()
    torch.cuda.set_device(hyp_param.local_rank)
    ckpt_sd = torch.load(hyp_param.inference_ckpt, map_location="cpu")
    if not isinstance(ckpt_sd, dict):
        raise TypeError("Checkpoint must be a state_dict dictionary.")
    # Support both formats:
    # 1) full-model checkpoint (keys like `v_model.*`, `aural_fuser.*`)
    # 2) train-only checkpoint for aural_fuser (keys without `aural_fuser.` prefix)
    if any(k.startswith("v_model.") or k.startswith("aural_fuser.") for k in ckpt_sd.keys()):
        av_model.load_state_dict(ckpt_sd, strict=True)
    else:
        av_model.aural_fuser.load_state_dict(ckpt_sd, strict=True)

    av_model = torch.nn.parallel.distributed.DistributedDataParallel(av_model, device_ids=[hyp_param.local_rank],
                                                                     find_unused_parameters=False)
    av_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(av_model)
    av_model.eval()

    from dataloader.dataset import AV
    from dataloader.visual.visual_augmentation import Augmentation as VisualAugmentation
    from dataloader.audio.audio_augmentation import Augmentation as AudioAugmentation
    from torch.utils.data import DataLoader, Subset
    from torch.utils.data.distributed import DistributedSampler

    visual_augmentation = VisualAugmentation(hyp_param.image_mean, hyp_param.image_std,
                                             hyp_param.image_size, hyp_param.image_size,
                                             hyp_param.scale_list, ignore_index=hyp_param.ignore_index)
    audio_augmentation = AudioAugmentation(mono=True)

    dataset = AV(split='test', augmentation={"visual": visual_augmentation, "audio": audio_augmentation},
                 param=hyp_param, root_path=hyp_param.data_root_path, data_name=hyp_param.inference_data_name)

    max_batches = getattr(hyp_param, "inference_max_batches", 0) or 0
    if max_batches > 0:
        n_samples = min(max_batches * hyp_param.batch_size, len(dataset))
        dataset = Subset(dataset, range(n_samples))

    sampler = DistributedSampler(dataset, shuffle=False)
    test_dataloader = DataLoader(dataset, batch_size=hyp_param.batch_size, sampler=sampler,
                                 num_workers=hyp_param.num_workers)

    from trainer.train import Trainer
    from utils.foreground_iou import ForegroundIoU
    from utils.foreground_fscore import ForegroundFScore

    metrics = {
        "foreground_iou": ForegroundIoU(),
        "foreground_f-score": ForegroundFScore(hyp_param.local_rank),
    }
    trainer = Trainer(hyp_param, loss=None, tensorboard=_DummyTensorboard(), metrics=metrics)

    # Same three modes as main.py validation: default first mask / iou_select / iou_occ_select
    runs = [
        ("", "default (logits[:,0])"),
        ("iou_select", "iou_select"),
        ("iou_occ_select", "iou_occ_select"),
    ]
    results = []
    for process, label in runs:
        fiou, ffscore = trainer.valid(epoch=0, dataloader=test_dataloader, model=av_model, process=process)
        results.append((label, fiou, ffscore))
        torch.cuda.empty_cache()

    if hyp_param.local_rank <= 0:
        print("\n========== inference (same three process flags as training valid) ==========")
        for label, fiou, ffscore in results:
            print("  {:32s}  f_iou={}  f_f-score={}".format(label, fiou, ffscore))
        print("=======================================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference: full test set + three process modes')

    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')

    parser.add_argument("--local_rank", type=int, default=-1,
                        help='multi-process training for DDP')

    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')

    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size (match training if needed)')

    parser.add_argument('--epochs', default=80, type=int,
                        help="unused")

    parser.add_argument('--lr', default=1e-5, type=float,
                        help="unused")

    parser.add_argument('--online', action="store_true",
                        help='unused')

    parser.add_argument(
        '--inference_ckpt', type=str, default=None,
        help='Trained AuralSAM2 checkpoint (.pth state_dict: full model or aural_fuser-only). '
             'SAM2 backbone is loaded from backbone_weight in configs (same path as training: repo_root/ckpts/sam_ckpts/). '
             'Default if unset: avs.code/training_details/.../hiera_l.pth',
    )
    parser.add_argument('--inference_data_name', type=str, default='v1s',
                        help='AVSBench subset folder label (v1s|v1m|v2); must match training test split')
    parser.add_argument('--inference_max_batches', type=int, default=0,
                        help='0 = full test; >0 = first N batches only (debug)')

    args = parser.parse_args()

    from configs.config import C

    args = EasyDict({**C, **vars(args)})

    _repo = pathlib.Path(__file__).resolve().parent
    # Repo root: .../AuralSAM2 (parent of avs.code)
    _workspace = _repo.parent.parent
    args.data_root_path = str(_workspace / 'AVSBench')
    args.backbone_weight = str(_workspace / 'ckpts' / 'sam_ckpts' / 'sam2_hiera_large.pt')
    args.audio.PRETRAINED_VGGISH_MODEL_PATH = str(_workspace / 'ckpts' / 'vggish-10086976.pth')
    args.saved_dir = '/tmp/v1s_infer_ckpt'
    pathlib.Path(args.saved_dir).mkdir(parents=True, exist_ok=True)
    if args.inference_ckpt is None:
        args.inference_ckpt = str(
            _repo.parent / 'training_details' / 'v1s' / 'hiera_l' / 'hiera_l.pth'
        )

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9901'

    torch.multiprocessing.spawn(main, nprocs=args.gpus, args=(args.gpus, args))
