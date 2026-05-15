from abc import ABC

import torch
import torch.nn as nn


class ContrastLoss(nn.Module, ABC):
    def __init__(self, hyp_param):
        super().__init__()
        self.param = hyp_param
        _defaults = {
            "temperature": 0.10,
            "ignore_idx": 255,
            "ood_idx": 254,
            "max_views": 512,
            "proj_dim": 512,
            "sample_limits": 128,
            "total_limits": 15240,
        }
        _raw = getattr(hyp_param, "contrastive_learning", None) or {}
        _cfg = {**_defaults, **_raw}
        self.temperature = _cfg["temperature"]
        self.ignore_idx = _cfg["ignore_idx"]
        self.ood_idx = _cfg["ood_idx"]
        self.max_views = _cfg["max_views"]
        self.proj_dim = _cfg["proj_dim"]
        self.sample_limits = _cfg["sample_limits"]
        self.total_limits = _cfg["total_limits"]

    def select_class_wise_samples(self, embeddings, audio_embeddings, predictions, masks, batch_idx):
        embedding_sample_list = []
        label_list = []
        embedding_sample_list_a = []
        label_list_a = []
        class_index_list = torch.unique(masks)

        if len(class_index_list) > 1:
            for class_index in class_index_list[1:]:
                embedding_sample_list_a.append(audio_embeddings.unsqueeze(0))
                label_list_a.append(class_index.unsqueeze(0) + batch_idx * 1e3)
        else:
            embedding_sample_list_a.append(audio_embeddings.unsqueeze(0))
            label_list_a.append(torch.zeros([1], device=embeddings.device) + batch_idx * 1e3)

        sample_limits = self.sample_limits
        embeddings = embeddings.permute(1, 0)
        for class_index in class_index_list:
            hard_indices = embeddings[((masks != predictions) & (masks == class_index)).nonzero()]
            easy_indices = embeddings[((masks == predictions) & (masks == class_index)).nonzero()]

            hard_indices_num, easy_indices_num = hard_indices.shape[0], easy_indices.shape[0]
            selective_num_hard = min(sample_limits, hard_indices_num)
            selective_num_easy = min(sample_limits, easy_indices_num)

            if (selective_num_hard + selective_num_easy) < sample_limits * 2:
                if selective_num_hard > selective_num_easy:
                    selective_num_hard += sample_limits * 2 - selective_num_easy
                else:
                    selective_num_easy += sample_limits * 2 - selective_num_hard

            hard_chosen_indices = torch.randperm(hard_indices_num)[:selective_num_hard]
            embedding_sample_list.append(hard_indices[hard_chosen_indices])
            label_list.append(masks[hard_chosen_indices] + batch_idx * 1e3)

            easy_chosen_indices = torch.randperm(easy_indices_num)[:selective_num_easy]
            embedding_sample_list.append(easy_indices[easy_chosen_indices])
            label_list.append(masks[easy_chosen_indices] + batch_idx * 1e3)
        return embedding_sample_list, label_list, embedding_sample_list_a, label_list_a

    def forward_audio_visual(self, visual_embeddings, audio_embeddings, masks, predictions):
        masks = masks.flatten(start_dim=1)
        predictions = predictions.flatten(start_dim=1)
        visual_embeddings = visual_embeddings.flatten(start_dim=-2)

        visual_embedding_sample_list = []
        visual_label_list = []
        audio_embedding_sample_list = []
        audio_label_list = []

        for frame_idx in range(masks.shape[0]):
            current_vision_feats = visual_embeddings[frame_idx]
            current_masks = masks[frame_idx]
            current_predictions = predictions[frame_idx]
            current_audio_feats = audio_embeddings[frame_idx]
            for layer_idx in range(3):
                (
                    selected_vision_embeddings,
                    selected_vision_labels,
                    selected_audio_embeddings,
                    selected_audio_labels,
                ) = self.select_class_wise_samples(
                    current_vision_feats[layer_idx],
                    current_audio_feats[layer_idx],
                    current_predictions,
                    current_masks,
                    0,
                )
                visual_embedding_sample_list += selected_vision_embeddings
                visual_label_list += selected_vision_labels
                audio_embedding_sample_list += selected_audio_embeddings
                audio_label_list += selected_audio_labels

        if len(visual_embedding_sample_list) == 0:
            return 0.0

        visual_embedding_sample_list = torch.cat(visual_embedding_sample_list, dim=0).squeeze()
        visual_label_list = torch.cat(visual_label_list, dim=0).unsqueeze(-1)
        audio_embedding_sample_list = torch.cat(audio_embedding_sample_list, dim=0).squeeze()
        audio_label_list = torch.cat(audio_label_list).unsqueeze(1)

        total_limits = self.total_limits
        if visual_embedding_sample_list.shape[0] > total_limits:
            rand_index = torch.randperm(visual_embedding_sample_list.shape[0])[total_limits]
            visual_embedding_sample_list = visual_embedding_sample_list[:rand_index]
            visual_label_list = visual_label_list[:rand_index]
        loss = self.info_nce(
            visual_embedding_sample_list,
            visual_label_list,
            audio_embedding_sample_list,
            audio_label_list,
        )
        return loss

    def forward(self, embeddings, output_dicts, masks):
        predictions = torch.cat([i["multistep_pred_masks"] for i in output_dicts])
        predictions = torch.nn.functional.interpolate(
            predictions,
            size=(int(self.param.image_size / 16), int(self.param.image_size / 16)),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        masks = torch.nn.functional.interpolate(
            masks.unsqueeze(1),
            size=(int(self.param.image_size / 16), int(self.param.image_size / 16)),
            mode="nearest",
        ).squeeze()
        visual_embeddings, audio_embeddings = embeddings
        visual_embeddings = torch.cat(
            [
                torch.cat(
                    [
                        visual_embeddings[0][i].unsqueeze(0),
                        visual_embeddings[1][i].unsqueeze(0),
                        visual_embeddings[2][i].unsqueeze(0),
                    ]
                ).unsqueeze(0)
                for i in range(masks.shape[0])
            ]
        )
        audio_embeddings = torch.cat(
            [
                torch.cat(
                    [
                        audio_embeddings[0][i].unsqueeze(0),
                        audio_embeddings[1][i].unsqueeze(0),
                        audio_embeddings[2][i].unsqueeze(0),
                    ]
                ).unsqueeze(0)
                for i in range(masks.shape[0])
            ]
        )
        return self.forward_audio_visual(
            visual_embeddings, audio_embeddings.squeeze(), masks, predictions
        )

    @staticmethod
    def manipulate_cover_mask(a_label, current_mask):
        a_label = a_label + 1
        visual_mask = torch.matmul(a_label, torch.transpose(a_label, 0, 1))
        current_mask[: visual_mask.shape[1], : visual_mask.shape[0]][visual_mask == 1.0] = 0
        current_mask[: visual_mask.shape[1], : visual_mask.shape[0]][visual_mask == 4.0] = 0
        return current_mask

    def info_nce(self, anchors_, a_labels_, contras_, c_labels_):
        c_labels_ = torch.cat([a_labels_, c_labels_])
        contras_ = torch.cat([anchors_, contras_])
        mask = torch.eq(a_labels_, torch.transpose(c_labels_, 0, 1)).float()

        anchor_dot_contrast = torch.div(
            torch.matmul(anchors_, torch.transpose(contras_, 0, 1)),
            self.temperature,
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        neg_mask = 1 - mask

        mask = self.manipulate_cover_mask(a_label=a_labels_, current_mask=mask)
        mask = mask.fill_diagonal_(0.0)

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits + neg_logits)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        assert not torch.isnan(mean_log_prob_pos).any(), print(torch.isnan(log_prob).any())
        return -mean_log_prob_pos.mean()

