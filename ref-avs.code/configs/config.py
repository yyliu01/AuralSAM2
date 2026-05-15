"""Ref-AVS training / inference defaults (paths relative to repo root)."""
import os
import pathlib
import numpy
from easydict import EasyDict

_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
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

C.root_dir = _CODE_ROOT

# REFAVS layout: REFAVS/metadata.csv, REFAVS/media/<vid>/...
C.data_root_path = os.path.join(_WORKSPACE_ROOT, 'REFAVS')
C.backbone_weight = os.path.join(_WORKSPACE_ROOT, 'ckpts', 'sam_ckpts', 'sam2_hiera_large.pt')
C.sam_config_path = os.path.join('sam2', 'sam2_hiera_l.yaml')

C.num_classes = 2
C.image_mean = numpy.array([0.485, 0.456, 0.406])
C.image_std = numpy.array([0.229, 0.224, 0.225])
C.image_size = 1024
C.image_embedding_size = int(C.image_size / 16)
C.scale_list = [.5, .75, 1., 1.25, 1.5]
C.ignore_index = 255

C.lr = 7.5e-5
C.batch_size = 8
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.05
C.num_workers = 4

# Paste W&B API key here or set WANDB_API_KEY in the environment.
C.wandb_key = ""
C.proj_name = "AVS-final-report"
C.experiment_name = "ref-hiera-l"
C.wandb_online = False

C.saved_dir = os.path.join(_WORKSPACE_ROOT, 'ckpts', 'exp', C.experiment_name)
pathlib.Path(C.saved_dir).mkdir(parents=True, exist_ok=True)
