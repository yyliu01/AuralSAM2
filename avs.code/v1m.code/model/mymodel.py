import logging

from typing import List, Optional, Tuple, Union

import numpy
import numpy as np
import torch
from PIL.Image import Image

from model.visual.sam2.modeling.sam2_base import SAM2Base

from model.visual.sam2.modeling.backbones.hieradet import Hiera
from model.visual.sam2.modeling.backbones.image_encoder import FpnNeck
from model.visual.sam2.modeling.backbones.image_encoder import ImageEncoder
from model.visual.sam2.modeling.position_encoding import PositionEmbeddingSine

from model.visual.sam2.modeling.memory_attention import MemoryAttention
from model.visual.sam2.modeling.memory_attention import MemoryAttentionLayer
from model.visual.sam2.modeling.sam.transformer import RoPEAttention
from model.visual.sam2.modeling.memory_encoder import MemoryEncoder
from model.visual.sam2.modeling.memory_encoder import MaskDownSampler
from model.visual.sam2.modeling.memory_encoder import Fuser
from model.visual.sam2.modeling.memory_encoder import CXBlock

from model.visual.sam2.utils.transforms import SAM2Transforms
from model.visual.sam2.modeling.backbones.hieradet import do_pool
from model.visual.sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)


class AVmodel(torch.nn.Module):
    """End-to-end AV segmentation: SAM2 visual backbone + AuralFuser audio-visual fusion + tracking head."""

    def __init__(self, param, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0, ):
        super().__init__()
        self.param = param
        self.mask_threshold = mask_threshold
        self._bb_feat_sizes = [(int(self.param.image_size / 4), int(self.param.image_size / 4)),
                               (int(self.param.image_size / 8), int(self.param.image_size / 8)),
                               (int(self.param.image_size / 16), int(self.param.image_size / 16))]

        from model.visual.sam2.build_sam import build_sam2_visual_predictor
        self.v_model = build_sam2_visual_predictor(self.param.sam_config_path, self.param.backbone_weight,
                                                   apply_postprocessing=True, mode='train')
        self._transforms = SAM2Transforms(
            resolution=self.v_model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
        from model.aural_fuser import AuralFuser
        self.aural_fuser = AuralFuser(hyp_param=self.param)



    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.v_model.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.v_model.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.v_model.num_feature_levels:]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def forward_frame(self, frame_):
        frame = torch.nn.functional.interpolate(frame_, (self.param.image_size, self.param.image_size),
                                                antialias=True, align_corners=False, mode='bilinear')
        return self.v_model.image_encoder(frame)

    def forward(self, frames, spect, prompts, sam_process=False):
        """Fuse audio into FPN features, then run SAM2 tracking. `sam_process` is reserved for prompt path."""
        backbone_feats = self.v_model.forward_image(frames, pre_compute=False)
        audio_residual_feats = self.aural_fuser(backbone_feats, spect)
        visual_resfeats, audio_resfeats, proj_feats = audio_residual_feats

        map_res = visual_resfeats[::-1]
        vec_res = audio_resfeats[::-1]

        av_feats = (map_res, vec_res)
        backbone_feats = self.v_model.precompute_high_res_features(backbone_feats)
        backbone_feats = self.v_model.dont_prepare_prompt_inputs(backbone_feats, num_frames=frames.shape[0], 
                cond_frame=int(frames.shape[0]/2) if self.training else 0)
        outputs = self.v_model.forward_tracking_wo_prompt(backbone_feats, audio_res=av_feats)
        return outputs, proj_feats

    @property
    def device(self) -> torch.device:
        return self.v_model.device

    def freeze_sam_parameters(self):
        self.v_model.eval()
        for name, parameter in self.v_model.named_parameters():
            parameter.requires_grad = False
