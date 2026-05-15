"""DDP training entry: AV model with SAM2 frozen, AuralFuser trainable, Hydra transforms and loss."""
import os
import torch
import numpy
import random
import argparse
from easydict import EasyDict


def seed_it(seed):
    """Fix RNGs and cuDNN for reproducible runs (rank offsets seed in DDP)."""
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
    # NCCL process group; world size = GPUs on this node
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=hyp_param.local_rank,
        world_size=hyp_param.gpus * 1
    )
    seed_it(local_rank + hyp_param.seed)

    torch.cuda.set_device(hyp_param.local_rank)

    import model.visual.sam2  # noqa: F401 — registers Hydra `configs` (initialize_config_module)

    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    # Hydra configs under v1m.code/configs (same pattern as training/sam2_training_config.yaml)
    transform_config_path = 'training/sam2_training_config.yaml'

    if 'hiera_t' in hyp_param.sam_config_path:
        hyp_param.image_size = 224
        hyp_param.image_embedding_size = int(hyp_param.image_size / 16)
        print('\n upload image size to be {}x{} \n'.format(224, 224), flush=True)

    cfg = compose(config_name=transform_config_path)
    OmegaConf.resolve(cfg)
    hyp_param.contrastive_learning = OmegaConf.to_container(cfg.contrastive_learning, resolve=True)

    arch_h = compose(config_name='auralfuser/architecture.yaml')
    OmegaConf.resolve(arch_h)
    hyp_param.aural_fuser = OmegaConf.to_container(arch_h.aural_fuser, resolve=True)

    from model.mymodel import AVmodel
    av_model = AVmodel(hyp_param).cuda(hyp_param.local_rank)

    av_model = torch.nn.parallel.distributed.DistributedDataParallel(av_model, device_ids=[hyp_param.local_rank],
                                                                     find_unused_parameters=True)

    # Optimizer: parameter groups from AuralFuser only (train_* vs VGG backbone)
    from utils.utils import manipulate_params
    parameter_list = manipulate_params(hyp_param, av_model.module.aural_fuser)
    optimiser = torch.optim.AdamW(parameter_list, betas=(0.9, 0.999))

    from dataloader.dataset import AV
    from dataloader.visual.visual_augmentation import Augmentation as VisualAugmentation
    from dataloader.audio.audio_augmentation import Augmentation as AudioAugmentation
    from torch.utils.data.distributed import DistributedSampler

    compose_api = instantiate(cfg.train_transforms, _recursive_=True)[0]

    audio_augmentation = AudioAugmentation(mono=True)
    train_dataset = AV(split='train', augmentation={"visual": compose_api, "audio": audio_augmentation},
                       param=hyp_param, root_path=hyp_param.data_root_path, data_name='v1s')


    visual_augmentation = VisualAugmentation(hyp_param.image_mean, hyp_param.image_std,
                                             hyp_param.image_size, hyp_param.image_size,
                                             hyp_param.scale_list, ignore_index=hyp_param.ignore_index)

    audio_augmentation = AudioAugmentation(mono=True)

    random_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=hyp_param.batch_size,
                                                   sampler=random_sampler,
                                                   num_workers=hyp_param.num_workers, drop_last=True)

    test_dataset = AV(split='test', augmentation={"visual": visual_augmentation, "audio": audio_augmentation},
                      param=hyp_param, root_path=hyp_param.data_root_path, data_name='v1s')

    order_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, sampler=order_sampler,
                                                  num_workers=hyp_param.num_workers)


    criterion = instantiate(cfg.loss, _recursive_=True)['all']
    from utils.tensorboard import Tensorboard
    tensorboard = Tensorboard(config=hyp_param) if hyp_param.local_rank <= 0 else None

    from trainer.train import Trainer
    from utils.foreground_iou import ForegroundIoU
    from utils.foreground_fscore import ForegroundFScore
    metrics = {"foreground_iou": ForegroundIoU(), "foreground_f-score": ForegroundFScore(0 if hyp_param.local_rank <= 0 else hyp_param.local_rank)}

    trainer = Trainer(hyp_param, loss=criterion, tensorboard=tensorboard, metrics=metrics)
    

    curr_best = 0.  # checkpoint when IoU (iou_select mode) improves

    for epoch in range(hyp_param.epochs):
        av_model.train()
        av_model.module.freeze_sam_parameters()
        random_sampler.set_epoch(epoch)
        trainer.train(epoch=epoch, dataloader=train_dataloader, model=av_model, optimiser=optimiser)

        torch.distributed.barrier()
        torch.cuda.empty_cache()

        av_model.eval()
        # Three validation modes: default first mask / IoU-selected mask / IoU + objectness gate
        curr_results1, _ = trainer.valid(epoch=epoch, dataloader=test_dataloader, model=av_model, process='first_index')
        curr_results, _ = trainer.valid(epoch=epoch, dataloader=test_dataloader, model=av_model, process='iou_select')
        curr_results3, _ = trainer.valid(epoch=epoch, dataloader=test_dataloader, model=av_model, process='iou_occ_select')
        if hyp_param.local_rank <= 0 and curr_results > curr_best:
            curr_best = curr_results
            torch.save(av_model.module.aural_fuser.state_dict(), os.path.join(hyp_param.saved_dir, str(curr_results) + ".pth"))
        torch.distributed.barrier()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')

    parser.add_argument("--local_rank", type=int, default=-1,
                        help='multi-process training for DDP')

    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')

    parser.add_argument('--batch_size', default=1, type=int)

    parser.add_argument('--epochs', default=80, type=int,
                        help="total epochs that used for the training")

    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Default HEAD Learning rate is same as others, '
                             '*Note: in ddp training, lr will automatically times by n_gpu')

    parser.add_argument('--online', action="store_true",
                        help='switch on for visualization; switch off for debug')

    args = parser.parse_args()

    from configs.config import C

    args = EasyDict({**C, **vars(args)})
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9902'

    torch.multiprocessing.spawn(main, nprocs=args.gpus, args=(args.gpus, args))
