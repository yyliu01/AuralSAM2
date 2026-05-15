import torch
import copy
from typing import List, Dict, Set, Any
import itertools

def manipulate_params(cfg, model):
    weight_decay_norm = 0
    weight_decay_embed = 0
    defaults = {}
    defaults["lr"] = cfg.lr
    defaults["weight_decay"] = cfg.weight_decay

    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )

    params_training: List[Dict[str, Any]] = []
    params_finetuning: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()

    train_prefixes = (
        "patch_embeds",
        "f_blocks",
        "a_blocks",
        "fusion_modules",
        "smooth_convs",
        "train_proj_v1",
        "train_proj_a1",
        "text_proj",
    )

    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            if value in memo:
                continue
            memo.add(value)
            hyperparams = copy.copy(defaults)
            if 'vgg' in module_name or 'vgg' in module_param_name:
                hyperparams['lr'] *= 0.1
                params_finetuning.append({"params": [value], "name": [module_name], **hyperparams})
            elif (
                'train' in module_name
                or 'train' in module_param_name
                or module_name.startswith(train_prefixes)
            ):
                if (
                        "relative_position_bias_table" in module_param_name
                        or "pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = 0.0
                params_training.append({"params": [value], "name": [module_name], **hyperparams})
            else:
                print('undefined layer type.')
                raise NotImplementedError
    final_list = params_training + params_finetuning
    assert len([p for p in model.parameters() if p.requires_grad]) == len(final_list), 'checksum confirmed not pass.'
    return final_list