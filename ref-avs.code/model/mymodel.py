"""End-to-end Ref-AVS: SAM2 visual backbone + AuralFuser fusion + tracking head.

Orchestration follows ``avs.code/v1m.code/model/mymodel.py``.
"""
import torch

from model.visual.sam2.build_sam import build_sam2_visual_predictor
from model.visual.sam2.utils.transforms import SAM2Transforms
from model.aural_fuser import AuralFuser
from transformers import AutoTokenizer, AutoModel


class AVmodel(torch.nn.Module):
    """SAM2 + audio/text fusion (``aural_fuser``) + SAM2 tracking decoder."""

    def __init__(self, param, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0):
        super().__init__()
        self.param = param
        self.mask_threshold = mask_threshold
        self._bb_feat_sizes = [
            (int(self.param.image_size / 4), int(self.param.image_size / 4)),
            (int(self.param.image_size / 8), int(self.param.image_size / 8)),
            (int(self.param.image_size / 16), int(self.param.image_size / 16)),
        ]

        self.v_model = build_sam2_visual_predictor(
            self.param.sam_config_path,
            self.param.backbone_weight,
            apply_postprocessing=True,
            mode='train',
            hydra_overrides_extra=["++model.image_size={}".format(self.param.image_size)],
        )
        self._transforms = SAM2Transforms(
            resolution=self.v_model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
        self.aural_fuser = AuralFuser(hyp_param=self.param)
        self.text_tokenizer = AutoTokenizer.from_pretrained('distilbert/distilroberta-base')
        self.t_model = AutoModel.from_pretrained('distilbert/distilroberta-base')

    def _encode_text(self, prompts):
        """RoBERTa embeddings for referring expressions (frozen at train time)."""
        enc = self.text_tokenizer(
            *prompts,
            max_length=25,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc['input_ids'] = enc['input_ids'].cuda(self.param.local_rank, non_blocking=True)
        enc['attention_mask'] = enc['attention_mask'].cuda(self.param.local_rank, non_blocking=True)
        with torch.no_grad():
            return self.t_model(**enc).last_hidden_state

    def forward_frame(self, frame_):
        """Single-frame SAM2 image encoder pass (same helper pattern as v1m)."""
        frame = torch.nn.functional.interpolate(
            frame_, (self.param.image_size, self.param.image_size),
            antialias=True, align_corners=False, mode='bilinear',
        )
        return self.v_model.image_encoder(frame)

    def forward(self, frames, spect, prompts, sam_process=False):
        """Fuse audio+text into FPN, then run SAM2 tracking without box/mask prompts."""
        text_feats = self._encode_text(prompts)
        backbone_feats = self.v_model.forward_image(frames, pre_compute=False)
        audio_residual_feats = self.aural_fuser(backbone_feats, spect, text_feats)
        visual_resfeats, audio_resfeats, proj_feats = audio_residual_feats

        map_res = visual_resfeats[::-1]
        vec_res = audio_resfeats[::-1]
        av_feats = (map_res, vec_res)

        backbone_feats = self.v_model.precompute_high_res_features(backbone_feats)
        backbone_feats = self.v_model.dont_prepare_prompt_inputs(
            backbone_feats,
            num_frames=frames.shape[0],
            condition_frame=int(frames.shape[0] / 2),
        )
        outputs = self.v_model.forward_tracking_wo_prompt(backbone_feats, audio_res=av_feats)
        return outputs, proj_feats

    @property
    def device(self) -> torch.device:
        return self.v_model.device

    def freeze_sam_parameters(self):
        """Freeze SAM2 and text backbone; only ``aural_fuser`` is trained."""
        self.v_model.eval()
        self.t_model.eval()
        for _, parameter in self.v_model.named_parameters():
            parameter.requires_grad = False
