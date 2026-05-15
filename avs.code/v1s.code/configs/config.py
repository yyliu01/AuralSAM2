import os
import numpy
from easydict import EasyDict

# v1m.code package root (parent of this `configs/` directory)
_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# workspace root (parent of avs.code)
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(_CODE_ROOT))

C = EasyDict()
config = C
cfg = C

C.seed = 666

C.audio = EasyDict()
C.audio.FREEZE_AUDIO_EXTRACTOR = True
C.audio.PRETRAINED_VGGISH_MODEL_PATH = os.path.join(_WORKSPACE_ROOT, 'ckpts', 'vggish-10086976.pth')
C.audio.PREPROCESS_AUDIO_TO_LOG_MEL = False
C.audio.POSTPROCESS_LOG_MEL_WITH_PCA = False
C.train_vggish = False

"""Root Directory Config"""
C.repo_name = 'AV'
C.root_dir = _CODE_ROOT

"""Data Dir and Weight Dir"""
C.data_root_path = os.path.join(_WORKSPACE_ROOT, 'AVSBench')
C.backbone_weight = os.path.join(_WORKSPACE_ROOT, 'ckpts', 'sam_ckpts', 'sam2_hiera_large.pt')
C.sam_config_path = os.path.join('sam2', 'sam2_hiera_l.yaml')

"""Network Config"""
C.fix_bias = True
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Image Config"""
C.num_classes = 2

C.image_mean = numpy.array([0.485, 0.456, 0.406])
C.image_std = numpy.array([0.229, 0.224, 0.225])


C.image_size = 1024
C.image_embedding_size = int(C.image_size / 16)
C.avsbench_size = (224, 224)

C.scale_list = [.5, .75, 1., 1.25, 1.5]
C.ignore_index = 255

"""Train Config"""
C.lr = 7.5e-5
C.batch_size = 8
C.energy_weight = .05

C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.05

C.num_workers = 4

"""Display Config"""
C.record_info_iter = 20
C.display_iter = 50

"""Wandb Config"""
# Paste your W&B API key here, or set the WANDB_API_KEY environment variable instead.
C.wandb_key = ""

# Your project [work_space] name
C.proj_name = "AVS-final-report"

C.experiment_name = "v1s-hiera-l"


# False = no wandb logging (see utils/tensorboard.py)
C.wandb_online = False

"""Save Config"""
C.saved_dir = os.path.join(_WORKSPACE_ROOT, 'ckpts', C.experiment_name)

import pathlib

pathlib.Path(C.saved_dir).mkdir(parents=True, exist_ok=True)
