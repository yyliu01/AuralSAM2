"""DDP training: frozen SAM2 + text, trainable AuralFuser (Ref-AVS)."""
import os
import argparse
import random

import numpy
import torch
from easydict import EasyDict


def seed_it(seed):
    os.environ["PYTHONSEED"] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(local_rank, ngpus_per_node, hyp_param):
    hyp_param.local_rank = local_rank
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://',
        rank=local_rank, world_size=hyp_param.gpus,
    )
    seed_it(local_rank + hyp_param.seed)
    torch.cuda.set_device(local_rank)

    import model.visual.sam2  # noqa: F401 — registers Hydra config store

    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = compose(config_name='configs/training/sam2_training_config.yaml')
    OmegaConf.resolve(cfg)
    hyp_param.contrastive_learning = OmegaConf.to_container(cfg.contrastive_learning, resolve=True)

    arch_h = compose(config_name='configs/auralfuser/architecture.yaml')
    OmegaConf.resolve(arch_h)
    hyp_param.aural_fuser = OmegaConf.to_container(arch_h.aural_fuser, resolve=True)

    hyp_param.image_size = 1024
    hyp_param.image_embedding_size = int(hyp_param.image_size / 16)

    from model.mymodel import AVmodel
    av_model = AVmodel(hyp_param).cuda(local_rank)
    av_model = torch.nn.parallel.DistributedDataParallel(
        av_model, device_ids=[local_rank], find_unused_parameters=True,
    )

    from utils.utils import manipulate_params
    optimiser = torch.optim.AdamW(manipulate_params(hyp_param, av_model.module.aural_fuser), betas=(0.9, 0.999))

    from dataloader.dataset import AV
    from dataloader.visual.visual_augmentation import Augmentation as VisualAugmentation
    from dataloader.audio.audio_augmentation import Augmentation as AudioAugmentation
    from torch.utils.data.distributed import DistributedSampler

    compose_api = instantiate(cfg.train_transforms, _recursive_=True)[0]
    audio_aug = AudioAugmentation(mono=True)
    train_dataset = AV(
        split='train',
        augmentation={"visual": compose_api, "audio": audio_aug},
        param=hyp_param,
        root_path=hyp_param.data_root_path,
    )

    visual_aug = VisualAugmentation(
        hyp_param.image_mean, hyp_param.image_std,
        hyp_param.image_size, hyp_param.image_size,
        hyp_param.scale_list, ignore_index=hyp_param.ignore_index,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyp_param.batch_size,
        sampler=DistributedSampler(train_dataset, shuffle=True),
        num_workers=hyp_param.num_workers,
        drop_last=True,
    )

    def _test_loader(split):
        ds = AV(split=split, augmentation={"visual": visual_aug, "audio": audio_aug},
                param=hyp_param, root_path=hyp_param.data_root_path)
        return torch.utils.data.DataLoader(
            ds, batch_size=4,
            sampler=DistributedSampler(ds, shuffle=False),
            num_workers=hyp_param.num_workers,
        )

    test_s_loader = _test_loader('test_s')
    test_u_loader = _test_loader('test_u')
    test_n_loader = _test_loader('test_n')

    criterion = instantiate(cfg.loss, _recursive_=True)['all']

    from utils.tensorboard import Tensorboard
    tensorboard = Tensorboard(config=hyp_param) if local_rank <= 0 else None

    from trainer.train import Trainer
    from utils.foreground_iou import ForegroundIoU
    from utils.foreground_fscore import ForegroundFScore
    from utils.foreground_s import ForegroundS
    metrics = {
        "foreground_iou": ForegroundIoU(),
        "foreground_f-score": ForegroundFScore(0 if local_rank <= 0 else local_rank),
        "foreground_s": ForegroundS(),
    }
    trainer = Trainer(hyp_param, loss=criterion, tensorboard=tensorboard, metrics=metrics)

    test_s_best, test_u_best = 0.2, 0.2
    for epoch in range(hyp_param.epochs + 1):
        av_model.train()
        av_model.module.freeze_sam_parameters()
        train_loader.sampler.set_epoch(epoch)
        trainer.train(epoch=epoch, dataloader=train_loader, model=av_model, optimiser=optimiser)

        torch.distributed.barrier()
        torch.cuda.empty_cache()

        av_model.eval()
        test_s, _ = trainer.valid(epoch=epoch, dataloader=test_s_loader, model=av_model, process='test_s')
        test_u, _ = trainer.valid(epoch=epoch, dataloader=test_u_loader, model=av_model, process='test_u')
        trainer.valid_null(epoch=epoch, dataloader=test_n_loader, model=av_model, process='test_n')

        if local_rank <= 0 and (test_s > test_s_best or test_u > test_u_best):
            test_s_best = max(test_s, test_s_best)
            test_u_best = max(test_u, test_u_best)
            torch.save(
                av_model.module.aural_fuser.state_dict(),
                os.path.join(
                    hyp_param.saved_dir,
                    f's({float(test_s)})_u({float(test_u)}).pth',
                ),
            )

        torch.distributed.barrier()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ref-AVS training')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('-g', '--gpus', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    args = parser.parse_args()

    from configs.config import C
    args = EasyDict({**C, **vars(args)})
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9901'
    torch.multiprocessing.spawn(main, nprocs=args.gpus, args=(args.gpus, args))
