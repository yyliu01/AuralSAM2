"""Contrastive loss used during SAM2 + fusion training (config from Hydra `contrastive_learning`, tmp.code style)."""
import torch
from abc import ABC
import torch.nn as nn


class ContrastLoss(nn.Module, ABC):
    def __init__(self, hyp_param):
        super(ContrastLoss, self).__init__()
        self.param = hyp_param
        _defaults = {
            "temperature": 0.10,
            "ignore_idx": 255,
            "ood_idx": 254,
            "max_views": 512,
            "proj_dim": 512,
            "sample_limits": 64,
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
        # means not silence
        if len(class_index_list) > 1:
            for class_index in class_index_list[1:]:
                embedding_sample_list_a.append(audio_embeddings.unsqueeze(0))
                label_list_a.append(class_index.unsqueeze(0) + batch_idx * 1e3)
        else:
            embedding_sample_list_a.append(audio_embeddings.unsqueeze(0))
            label_list_a.append(torch.zeros([1], device=embeddings.device) + batch_idx * 1e3)

        # contras_list = []
        # contras_label_list = []
        sample_limits = self.sample_limits
        # we only have 0, 1
        embeddings = embeddings.permute(1, 0)
        for class_index in class_index_list:
            hard_indices = embeddings[((masks != predictions) & (masks == class_index)).nonzero()]
            easy_indices = embeddings[((masks == predictions) & (masks == class_index)).nonzero()]

            hard_indices_num, easy_indices_num = hard_indices.shape[0], easy_indices.shape[0]

            # the number that is selected to the contrastive learning.
            selective_num_hard = min(sample_limits, hard_indices_num)
            selective_num_easy = min(sample_limits, easy_indices_num)

            if (selective_num_hard + selective_num_easy) < sample_limits * 2:
                if selective_num_hard > selective_num_easy:
                    selective_num_hard += sample_limits * 2 - selective_num_easy
                else:
                    selective_num_easy += sample_limits * 2 - selective_num_hard

            # skip if contains too limited samples.
            # if selective_num_hard < 10 and selective_num_easy < 10:
            #     continue
            hard_chosen_indices = torch.randperm(hard_indices_num)[:selective_num_hard]
            embedding_sample_list.append(hard_indices[hard_chosen_indices])
            label_list.append(masks[hard_chosen_indices] + batch_idx * 1e3)

            # add negative features to list.
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
                (selected_vision_embeddings, selected_vision_labels,
                 selected_audio_embeddings, selected_audio_labels) = self.select_class_wise_samples(current_vision_feats[layer_idx],
                                                                                                  current_audio_feats[layer_idx],
                                                                                                   current_predictions,
                                                                                                   current_masks,
                                                                                                   0)

                visual_embedding_sample_list += selected_vision_embeddings
                visual_label_list += selected_vision_labels

                audio_embedding_sample_list += selected_audio_embeddings
                audio_label_list += selected_audio_labels

        if len(visual_embedding_sample_list) == 0: return 0.
        visual_embedding_sample_list = torch.cat(visual_embedding_sample_list, dim=0).squeeze()
        visual_label_list = torch.cat(visual_label_list, dim=0).unsqueeze(-1)
        audio_embedding_sample_list = torch.cat(audio_embedding_sample_list, dim=0).squeeze()
        audio_label_list = torch.cat(audio_label_list).unsqueeze(1)

        # print(visual_embedding_sample_list.shape, visual_label_list.shape)
        # print(audio_embedding_sample_list.shape, audio_label_list.shape)
        # exit(1)
        total_limits = self.total_limits
        if visual_embedding_sample_list.shape[0] > total_limits:
            rand_index = torch.randperm(visual_embedding_sample_list.shape[0])[total_limits]
            visual_embedding_sample_list = visual_embedding_sample_list[:rand_index]
            visual_label_list = visual_label_list[:rand_index]
        loss = self.info_nce(visual_embedding_sample_list, visual_label_list, audio_embedding_sample_list,
                             audio_label_list)
        return loss


    # proof the q-project CAN BE the projector head of the contrastive learning.
    # At the moment, I do believe the ATTENTION is the another format of the contrastive learning.
    # First experiment: ignore the sound, only work on the projected vision mask.
    def forward(self, embeddings, output_dicts, masks):
        predictions = torch.cat([i['multistep_pred_masks'] for i in output_dicts])
        predictions = torch.nn.functional.interpolate(predictions, size=(int(self.param.image_size/16), int(self.param.image_size/16)),
                                                      mode='bilinear', align_corners=False).squeeze(1)
        masks = torch.nn.functional.interpolate(masks.unsqueeze(1), size=(int(self.param.image_size/16), int(self.param.image_size/16)),
                                                      mode='nearest').squeeze(1)
        visual_embeddings, audio_embeddings = embeddings
        # if len(predictions.shape) < 3 and len(masks.shape) < 3:
        #     predictions = predictions.unsqueeze(0)
        #     masks = masks.unsqueeze(0)

        visual_embeddings = torch.cat([torch.cat([visual_embeddings[0][i].unsqueeze(0),
                                                  visual_embeddings[1][i].unsqueeze(0),
                                                  visual_embeddings[2][i].unsqueeze(0)]).unsqueeze(0)
                             for i in range(masks.shape[0])])
        audio_embeddings = torch.cat([torch.cat([audio_embeddings[0][i].unsqueeze(0),
                                                  audio_embeddings[1][i].unsqueeze(0),
                                                  audio_embeddings[2][i].unsqueeze(0)]).unsqueeze(0)
                             for i in range(masks.shape[0])])

        # dict_keys(['point_inputs', 'mask_inputs', 'multistep_pred_masks', 'multistep_pred_masks_high_res',
        # 'multistep_pred_multimasks', 'multistep_pred_multimasks_high_res', 'multistep_pred_ious',
        # 'multistep_point_inputs', 'multistep_object_score_logits', 'pred_masks', 'pred_masks_high_res',
        # 'maskmem_features', 'maskmem_pos_enc'])
        return self.forward_audio_visual(visual_embeddings, audio_embeddings.squeeze(-1), masks, predictions)

    # def forward_visual_only(self, visual_embeddings, masks, predictions):
    #     masks = masks.flatten(start_dim=1)
    #     predictions = predictions.flatten(start_dim=1)
    #     visual_embeddings = visual_embeddings.flatten(start_dim=-2)
    #
    #     visual_embedding_sample_list = []
    #     visual_label_list = []
    #     audio_embedding_sample_list = []
    #     audio_label_list = []
    #
    #     for frame_idx in range(masks.shape[0]):
    #         current_vision_feats = visual_embeddings[frame_idx]
    #         current_masks = masks[frame_idx]
    #         current_predictions = predictions[frame_idx]
    #         for layer_idx in range(3):
    #             current_select_embeddings, current_select_labels = self.select_class_wise_samples(current_vision_feats[layer_idx],
    #                                                                                               None,
    #                                                                                                current_predictions,
    #                                                                                                current_masks,
    #                                                                                                frame_idx)
    #             visual_embedding_sample_list += current_select_embeddings
    #             visual_label_list += current_select_labels
    #
    #
    #
    #     if len(embedding_sample_list) == 0: return 0.
    #     embedding_sample_list = torch.cat(embedding_sample_list, dim=0).squeeze()
    #     label_list = torch.cat(label_list, dim=0).unsqueeze(-1)
    #     total_limits = 15240
    #     if embedding_sample_list.shape[0] > total_limits:
    #         rand_index = torch.randperm(embedding_sample_list.shape[0])[total_limits]
    #         embedding_sample_list = embedding_sample_list[:rand_index]
    #         label_list = label_list[:rand_index]
    #     loss = self.info_nce(embedding_sample_list, label_list, embedding_sample_list,
    #                          label_list)
    #     return loss


    """
        # embeddings_size = (int(self.param.image_size/16), int(self.param.image_size/16))
        # masks = torch.nn.functional.interpolate(masks.float(), embeddings_size, mode='nearest')
        # masks = masks.flatten(start_dim=1)
        # predictions = torch.nn.functional.interpolate(predictions.float(), embeddings_size, mode='nearest')
        # predictions = predictions.flatten(start_dim=1)
        # 
        # embedding_sample_list = []
        # label_list = []
        # contras_sample_list = []
        # contras_label_list = []

        # temp3.
        # embedding_visual, embedding_audio = embeddings
        # embedding_visual = torch.nn.functional.normalize(embedding_visual, p=2, dim=1)
        # embedding_audio = torch.nn.functional.normalize(embedding_audio, p=2, dim=1)
        # embedding_visual = embedding_visual.reshape(self.param.batch_size, int(embedding_visual.shape[0]/self.param.batch_size),
        #                                 *embedding_visual.shape[-2:])
        # 
        # embedding_audio = embedding_audio.reshape(self.param.batch_size, int(embedding_audio.shape[0]/self.param.batch_size),
        #                                             *embedding_audio.shape[-2:])
        # masks = masks.reshape(self.param.batch_size, int(masks.shape[0]/self.param.batch_size),
        #                       masks.shape[-1])
        # predictions = predictions.reshape(self.param.batch_size, int(predictions.shape[0]/self.param.batch_size),
        #                                   predictions.shape[-1])
        # 
        # for batch_idx in range(masks.shape[0]):
        #     current_video_clip_embed = embedding_visual[batch_idx]
        #     current_video_clip_masks = masks[batch_idx]
        #     current_video_clip_preds = predictions[batch_idx]
        #     current_audio_clip_embed = embedding_audio[batch_idx]
        #     # print(current_video_clip_embed.shape, current_audio_clip_embed.shape, current_video_clip_masks.shape, current_video_clip_preds.shape)
        #     # exit(1)
        #     for sample_idx in range(masks.shape[1]):
        #         current_vision_feats = current_video_clip_embed[batch_idx]
        #         current_audio_feats = current_audio_clip_embed[batch_idx]
        #         current_masks = current_video_clip_masks[batch_idx]
        #         current_predictions = current_video_clip_preds[batch_idx]
        #         current_select_embeddings, current_select_labels = self.select_class_wise_samples(current_vision_feats,
        #                                                                                           current_audio_feats,
        #                                                                                           current_predictions,
        #                                                                                           current_masks,
        #                                                                                           batch_idx)
        
        # temp2.
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        embeddings = embeddings.reshape(self.param.batch_size, int(embeddings.shape[0]/self.param.batch_size),
                                        *embeddings.shape[-2:])
        masks = masks.reshape(self.param.batch_size, int(masks.shape[0]/self.param.batch_size),
                              masks.shape[-1])
        predictions = predictions.reshape(self.param.batch_size, int(predictions.shape[0]/self.param.batch_size),
                                          predictions.shape[-1])

        for batch_idx in range(masks.shape[0]):
            current_video_clip_embed = embeddings[batch_idx]
            current_video_clip_masks = masks[batch_idx]
            current_video_clip_preds = predictions[batch_idx]
            # current_audio_clip_feats =
            for sample_idx in range(masks.shape[1]):
                current_vision_feats = current_video_clip_embed[batch_idx]
                current_masks = current_video_clip_masks[batch_idx]
                current_predictions = current_video_clip_preds[batch_idx]
                current_select_embeddings, current_select_labels = self.select_class_wise_samples(current_vision_feats,
                                                                                                  current_predictions,
                                                                                                  current_masks,
                                                                                                  batch_idx)
                embedding_sample_list += current_select_embeddings
                label_list += current_select_labels
                # hard_indices = current_vision_feats[(current_masks != current_predictions).nonzero()]
                # easy_indices = current_vision_feats[(current_masks == current_predictions).nonzero()]
                #
                # hard_indices_num, easy_indices_num = hard_indices.shape[0], easy_indices.shape[0]
                #
                # # the number that is selected to the contrastive learning.
                # selective_num_hard = min(sample_limits, hard_indices_num)
                # selective_num_easy = min(sample_limits, easy_indices_num)
                # # skip if contains too limited samples.
                # if selective_num_hard < 10 or selective_num_easy < 10:
                #     continue
                #
                # hard_chosen_indices = torch.randperm(hard_indices_num)[:selective_num_hard]
                # embedding_sample_list.append(hard_indices[hard_chosen_indices])
                # label_list.append(current_masks[hard_chosen_indices] + batch_idx * 1e3)
                #
                # # add negative features to list.
                # easy_chosen_indices = torch.randperm(easy_indices_num)[:selective_num_easy]
                # embedding_sample_list.append(easy_indices[easy_chosen_indices])
                # label_list.append(current_masks[easy_chosen_indices] + batch_idx * 1e3)

        if len(embedding_sample_list) == 0: return 0.
        embedding_sample_list = torch.cat(embedding_sample_list, dim=0).squeeze()
        label_list = torch.cat(label_list, dim=0).unsqueeze(-1)
        total_limits = self.total_limits
        if embedding_sample_list.shape[0] > total_limits:
            rand_index = torch.randperm(embedding_sample_list.shape[0])[total_limits]
            embedding_sample_list = embedding_sample_list[:rand_index]
            label_list = label_list[:rand_index]
        loss = self.info_nce(embedding_sample_list, label_list, embedding_sample_list,
                             label_list)

        # temp.
        # sample_limits = 500
        # for batch_idx in range(masks.shape[0]):
        #     # go through 3 layers embeddings.
        #     for j in range(len(embeddings)):
        #         current_vision_feats_list = embeddings[j]
        #         current_vision_feats = torch.nn.functional.normalize(current_vision_feats_list[batch_idx], p=2, dim=1)
        #         current_masks = masks[batch_idx]
        #         positive_indices = current_vision_feats[current_masks > 0, ...]
        #         negative_indices = current_vision_feats[current_masks == 0, ...]
        #         positive_indices_num, negative_indices_num = positive_indices.shape[0], negative_indices.shape[0]
        #
        #         # the number that is selected to the contrastive learning.
        #         selective_num = min(sample_limits, positive_indices_num, negative_indices_num)
        #         if selective_num < 50: continue  # skip if contains too limited samples.
        #
        #         embedding_sample_list.append(positive_indices[torch.randperm(positive_indices_num)[:selective_num]])
        #         label_list.append(torch.tensor([batch_idx + (self.param.local_rank * 100)] * selective_num,
        #                                        device=positive_indices.device))
        #
        #         # add negative features to list.
        #         negative_sample_list.append(negative_indices[torch.randperm(negative_indices_num)[:selective_num]])
        #         negative_label_list.append(torch.tensor([-1] * selective_num, device=negative_indices.device))
        #
        # if len(embedding_sample_list) == 0: return 0.
        # embedding_sample_list = torch.cat(embedding_sample_list, dim=0)
        # negative_sample_list = torch.cat(negative_sample_list, dim=0)
        # label_list = torch.cat(label_list)
        # negative_label_list = torch.cat(negative_label_list)
        #
        # loss = self.info_nce(embedding_sample_list, label_list.unsqueeze(-1),
        #                      torch.cat([embedding_sample_list, negative_sample_list], dim=0),
        #                      torch.cat([label_list, negative_label_list]).unsqueeze(-1))

        # output_list_embeddings = [torch.zeros_like(embedding_sample_list) for _ in range(torch.distributed.get_world_size())]
        # output_list_labels = [torch.zeros_like(label_list) for _ in range(torch.distributed.get_world_size())]
        #
        # torch.distributed.all_gather(output_list_embeddings, embedding_sample_list)
        # torch.distributed.all_gather(output_list_labels, label_list)
        #
        # output_list_embeddings = torch.cat(output_list_embeddings)
        # output_list_labels = torch.cat(output_list_labels, dim=1)
        # loss = self.info_nce(output_list_embeddings, output_list_labels, output_list_embeddings, output_list_labels)
        return loss
    """
    # q_max.
    # def forward(self, embeddings, masks):
    #     # for single-sounding obj. only, with first idx mask.
    #     masks = torch.nn.functional.interpolate(masks.float(), (64, 64), mode='bilinear', align_corners=False)
    #     masks = masks.flatten(start_dim=1)
    #     # embedding_sample_list = torch.zeros([masks.shape[0], 128]).to(self.param.local_rank)
    #     embedding_sample_list = []
    #     label_list = []
    #
    #     negative_sample_list = []
    #     negative_label_list = []
    #     sample_limits = 20
    #     for batch_idx in range(masks.shape[0]):
    #         # go through 3 layers embeddings.
    #         for j in range(len(embeddings)):
    #             current_vision_feats_list, current_audio_feats_list = embeddings[j]
    #             current_audio_feats = torch.nn.functional.normalize(current_audio_feats_list[batch_idx], p=2, dim=1)
    #             current_vision_feats = torch.nn.functional.normalize(current_vision_feats_list[batch_idx], p=2, dim=1)
    #             current_masks = masks[batch_idx]
    #
    #             # add following features to list.
    #             embedding_sample_list.append(current_vision_feats[current_masks > 0, ...].max(dim=0)[0].unsqueeze(0))
    #             label_list.append(batch_idx + (self.param.local_rank * 100))
    #
    #             embedding_sample_list.append(current_audio_feats)
    #             label_list.append(batch_idx + (self.param.local_rank * 100))
    #
    #             # add negative features to list.
    #             negative_num = min(current_vision_feats[current_masks == 0, ...].shape[0], sample_limits)
    #             if negative_num < 5: continue  # skip if contains too limited samples.
    #             rand_idx = torch.randperm(current_vision_feats[current_masks == 0, ...].shape[0])[:negative_num]
    #             negative_sample_list.append(current_vision_feats[current_masks == 0, ...][rand_idx])
    #             negative_label_list.append(torch.tensor([-1] * negative_num, device=current_vision_feats.device))
    #
    #     embedding_sample_list = torch.cat(embedding_sample_list)
    #     label_list = torch.tensor(label_list, device=masks.device)
    #     negative_sample_list = torch.cat(negative_sample_list, dim=0)
    #     negative_label_list = torch.cat(negative_label_list)
    #
    #     loss = self.info_nce(embedding_sample_list, label_list.unsqueeze(-1),
    #                          torch.cat([embedding_sample_list, negative_sample_list], dim=0),
    #                          torch.cat([label_list, negative_label_list]).unsqueeze(-1))
    #
    #     # output_list_embeddings = [torch.zeros_like(embedding_sample_list) for _ in range(torch.distributed.get_world_size())]
    #     # output_list_labels = [torch.zeros_like(label_list) for _ in range(torch.distributed.get_world_size())]
    #     #
    #     # torch.distributed.all_gather(output_list_embeddings, embedding_sample_list)
    #     # torch.distributed.all_gather(output_list_labels, label_list)
    #     #
    #     # output_list_embeddings = torch.cat(output_list_embeddings)
    #     # output_list_labels = torch.cat(output_list_labels, dim=1)
    #     # loss = self.info_nce(output_list_embeddings, output_list_labels, output_list_embeddings, output_list_labels)
    #     return loss

    # attention mean.
    # def forward(self, embeddings):
    #     embedding_sample_list = []
    #     label_list = []
    #     for layer_embeddings in embeddings:
    #         embedding_sample_list.append(torch.nn.functional.normalize(layer_embeddings, p=2, dim=1))
    #         # currently we only utilise single frame.
    #         label_list.append(torch.tensor(list(range(0, 1 + 1)) * self.param.batch_size) + (self.param.local_rank * 100))
    #     embedding_sample_list = torch.cat(embedding_sample_list).cuda(self.param.local_rank)
    #     label_list = torch.cat(label_list).cuda(self.param.local_rank).unsqueeze(0)
    #
    #     """
    #     all gather implementation.
    #     """
    #     """
    #     output_list_embeddings = [torch.zeros_like(embedding_sample_list) for _ in range(torch.distributed.get_world_size())]
    #     output_list_labels = [torch.zeros_like(label_list) for _ in range(torch.distributed.get_world_size())]
    #
    #     torch.distributed.all_gather(output_list_embeddings, embedding_sample_list)
    #     torch.distributed.all_gather(output_list_labels, label_list)
    #
    #     output_list_embeddings = torch.cat(output_list_embeddings)
    #     output_list_labels = torch.cat(output_list_labels, dim=1)
    #     loss = self.info_nce(output_list_embeddings, output_list_labels, output_list_embeddings, output_list_labels)
    #     """
    #     loss = self.info_nce(embedding_sample_list, label_list, embedding_sample_list, label_list)
    #     # frame_token_semantic_attn = torch.nn.functional.normalize(frame_token_semantic_attn.squeeze(), p=2, dim=1)
    #     # audio_token_attn = torch.nn.functional.normalize(audio_token_attn, p=2, dim=1)
    #     # city_gt = torch.nn.functional.interpolate(city_gt.unsqueeze(1).float(), size=city_proj.shape[2:],
    #     #                                           mode='nearest').squeeze().long()
    #     #
    #     # ood_gt = torch.nn.functional.interpolate(ood_gt.unsqueeze(1).float(), size=ood_proj.shape[2:],
    #     #                                          mode='nearest').squeeze().long()
    #     #
    #     # # normalise the embed results
    #     # city_proj = torch.nn.functional.normalize(city_proj, p=2, dim=1)
    #     # ood_proj = torch.nn.functional.normalize(ood_proj, p=2, dim=1)
    #
    #     # randomly extract embed samples within a batch
    #     # anchor_embeds, anchor_labels, contrs_embeds, contrs_labels = self.extraction_samples(city_proj, city_gt,
    #     #                                                                                      ood_proj, ood_gt)
    #     #
    #     # # calculate the CoroCL
    #     # loss = self.info_nce(anchors_=anchor_embeds, a_labels_=anchor_labels.unsqueeze(1), contras_=contrs_embeds,
    #     #                      c_labels_=contrs_labels.unsqueeze(1)) if anchor_embeds.nelement() > 0 else \
    #     #     torch.tensor([.0], device=city_proj.device)
    #
    #     return loss
    @staticmethod
    def manipulate_cover_mask(a_label, current_mask):
        # shifting current visual index value
        # background:=1, foreground:=2.
        a_label = a_label + 1
        visual_mask = torch.matmul(a_label, torch.transpose(a_label, 0, 1))
        # kicked out the positive value in same visual class.
        current_mask[:visual_mask.shape[1], :visual_mask.shape[0]][visual_mask == 1.] = 0
        current_mask[:visual_mask.shape[1], :visual_mask.shape[0]][visual_mask == 4.] = 0

        return current_mask

    # The implementation of cross-image contrastive learning is based on:
    # https://github.com/tfzhou/ContrastiveSeg/blob/287e5d3069ce6d7a1517ddf98e004c00f23f8f99/lib/loss/loss_contrast.py
    def info_nce(self, anchors_, a_labels_, contras_, c_labels_):
        c_labels_ = torch.cat([a_labels_, c_labels_])
        contras_ = torch.cat([anchors_, contras_])
        # calculates the binary mask: same category => 1, different categories => 0
        mask = torch.eq(a_labels_, torch.transpose(c_labels_, 0, 1)).float()

        # calculates the dot product
        anchor_dot_contrast = torch.div(torch.matmul(anchors_, torch.transpose(contras_, 0, 1)),
                                        self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # calculates the negative mask
        neg_mask = 1 - mask

        # avoid the self duplicate issue
        mask = self.manipulate_cover_mask(a_label=a_labels_, current_mask=mask)
        mask = mask.fill_diagonal_(0.)

        # sum the negative odot results
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        # log_prob -> log(exp(x))-log(exp(x) + exp(y))
        # log_prob -> log{exp(x)/[exp(x)+exp(y)]}
        log_prob = logits - torch.log(exp_logits + neg_logits)
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # calculate the info-nce based on the positive samples (under same categories)
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        assert not torch.isnan(mean_log_prob_pos).any(), print(torch.isnan(log_prob).any())
        return - mean_log_prob_pos.mean()

    # def extraction_samples(self, city_embd, city_label, ood_embd, ood_label):
    #     # reformat the matrix
    #     city_embd = city_embd.flatten(start_dim=2).permute(0, 2, 1)
    #     city_label = city_label.flatten(start_dim=1)
    #     ood_embd = ood_embd.flatten(start_dim=2).permute(0, 2, 1)
    #     ood_label = ood_label.flatten(start_dim=1)
    #
    #     # define different types of embeds
    #     city_positive = city_embd[city_label == self.ood_idx]
    #     city_negative = city_embd[(city_label != self.ood_idx) & (city_label != self.ignore_idx)]
    #     ood_positive = ood_embd[ood_label == self.ood_idx]
    #     ood_negative = ood_embd[(ood_label != self.ood_idx) & (ood_label != self.ignore_idx)]
    #
    #     # define the number of choice
    #     sample_num = int(min(self.max_views, city_positive.shape[0], ood_positive.shape[0],
    #                          city_negative.shape[0], ood_negative.shape[0]))
    #
    #     # randomly extract the anchor set with {city_ood, city_inlier}
    #     city_positive_anchor = city_positive[torch.randperm(city_positive.shape[0])][:sample_num]
    #     city_negative_anchor = city_negative[torch.randperm(city_negative.shape[0])][:sample_num]
    #
    #     anchor_embed = torch.cat([city_positive_anchor, city_negative_anchor], dim=0)
    #
    #     anchor_label = torch.cat([torch.empty(city_positive_anchor.shape[0],
    #                                           device=city_positive_anchor.device).fill_(1.),
    #                               torch.empty(city_negative_anchor.shape[0],
    #                                           device=city_negative_anchor.device).fill_(0.)])
    #
    #     # randomly extract the contras set with {city_ood, city_inlier, coco_ood, coco_inlier}
    #     city_positive_contras = city_positive_anchor.clone()
    #     city_negative_contras = city_negative_anchor.clone()
    #     ood_positive_contras = ood_positive[torch.randperm(ood_positive.shape[0])][:sample_num]
    #     ood_negative_contras = ood_negative[torch.randperm(ood_negative.shape[0])][:sample_num]
    #
    #     contrs_embed = torch.cat([city_positive_contras, city_negative_contras,
    #                               ood_positive_contras, ood_negative_contras], dim=0)
    #
    #     contrs_label = torch.cat([torch.empty(city_positive_contras.shape[0],
    #                                           device=city_positive_contras.device).fill_(1.),
    #                               torch.empty(city_negative_contras.shape[0],
    #                                           device=city_negative_contras.device).fill_(0.),
    #                               torch.empty(ood_positive_contras.shape[0],
    #                                           device=ood_positive_contras.device).fill_(1.),
    #                               torch.empty(ood_negative_contras.shape[0],
    #                                           device=ood_negative_contras.device).fill_(0.)])
    #
    #     return anchor_embed, anchor_label, contrs_embed, contrs_label



