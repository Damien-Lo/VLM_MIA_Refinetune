import numpy as np
from collections import defaultdict
import torch

"""
Our proposed metric computation functions are listed here

Metric functions design structure

1. Input:

A value of meta_metrics dictionary has the structure as follows.

{
    "orig": [
        [
            (metric of a sample 0 in the batch),
            (metric of a sample 1 in the batch), ...
        ]
    ],
    "aug_1: [
        [
            (metric of a sample 0 in the batch, processed with a setting of aug_1),
            (metric of a sample 1 in the batch, processed with another setting of aug_1), ...
        ],
        [
            (metric of a sample 0 in the batch, processed with a setting of aug_1),
            (metric of a sample 1 in the batch, processed with another setting of aug_1), ...
        ]
    ]
}

2. Output:

Computed img_metrics that has following structure

If the metric has different sub_settings (such as different ratios)

{
    "Metric_1" : [ # ex: Min_30%_entropy 
        finalized score from the sample 0 in the batch,
        finalized score from the sample 1 in the batch, ...
    ],
    "Metric_2" : [
        finalized score from the sample 0 in the batch,
        finalized score from the sample 1 in the batch, ...
    ],
    ...
}

If the metric does not have such settings:

[
    finalized score from the sample 0 in the batch,
    finalized score from the sample 1 in the batch, ...
]

This will be automatically aggregated afterwards for AUC.

"""

def cross_entropy_mink(per_token_ce, cfg):
    ratio = cfg.ratio

    aug_names = list(per_token_ce.keys())
    result = dict()
    
    for _sample_idx, per_token_loss in enumerate(per_token_ce["orig"][0]):
        _loss_diff = dict()
        for aug_name in aug_names:
            if aug_name == "orig":
                continue
            current_augs = per_token_ce[aug_name]
            for _aug in current_augs:
                for _ratio in ratio:
                    k_length = int(len(_aug[_sample_idx])*_ratio)
                    if k_length == 0:
                        k_length = 1
                    if _ratio not in _loss_diff:
                        _loss_diff[_ratio] = list()

                    # Convert to numpy array if needed and get indices
                    aug_values = np.array(_aug[_sample_idx])
                    aug_max_idx = np.argsort(aug_values)[-k_length:]
                    aug_max_avg = aug_values[aug_max_idx].mean()
                    
                    # Convert per_token_loss to numpy array if needed
                    orig_values = np.array(per_token_loss)
                    orig_max_avg = orig_values[aug_max_idx].mean()
                    
                    _loss_diff[_ratio].append(orig_max_avg - aug_max_avg)

        for _key in _loss_diff:
            result_key = f"Min_{_key*100}% Cross_Entro_Augs"
            if result_key not in result: 
                result[result_key] = list()
            result[result_key].append(-1*np.mean(_loss_diff[_key]))

    return result


def cross_entropy_diff_mink(per_token_ce, cfg):
    ratio = cfg.ratio

    aug_names = list(per_token_ce.keys())
    result = dict()
    
    for _sample_idx, per_token_loss in enumerate(per_token_ce["orig"][0]):
        _loss_diff = dict()
        for aug_name in aug_names:
            if aug_name == "orig":
                continue
            current_augs = per_token_ce[aug_name]
            for _aug in current_augs:
                # Convert to numpy arrays
                orig_values = np.array(per_token_loss)
                aug_values = np.array(_aug[_sample_idx])
                _diff = orig_values - aug_values
                
                for _ratio in ratio:
                    k_length = int(len(_diff)*_ratio)
                    if k_length == 0:
                        k_length = 1
                    if _ratio not in _loss_diff:
                        _loss_diff[_ratio] = list()
                    
                    _diff_max = np.sort(_diff[k_length:]) # Bigger difference is bigger in negative
                    _loss_diff[_ratio].append(np.mean(_diff_max))

        for _key in _loss_diff:
            result_key = f"Min_{_key*100}% Cross_Entro_Augs"
            if result_key not in result: 
                result[result_key] = list()
            result[result_key].append(np.mean(_loss_diff[_key])) # members has smaller score (later will be flipped.)

    return result


def renyi_kl_div_mink(renyi_probs, cfg):
    ratio = cfg.ratio
    result = dict()
    original_probs = renyi_probs['orig'][0]

    setting_version_accumilator = cfg.augmentation_accumilator
    aug_version_accumilator = cfg.augmentation_setting_version_accumilator

    per_sample_kld = list()

    for aug, settings in renyi_probs.items():
        if aug == "orig":
            continue

        per_sample_tokenwise_kl = defaultdict(list)

        for setting in settings:
            for sample_idx, aug_probs in enumerate(setting):
                kl = kl_div_per_token(torch.stack(original_probs[sample_idx]).cpu().numpy(), np.log(torch.stack(original_probs[sample_idx]).cpu().numpy()), np.log(torch.stack(aug_probs).cpu().numpy()))
                per_sample_tokenwise_kl[sample_idx].append(kl)

        setting_aggregated_per_sample_tokenwise_kl = defaultdict(list)

        if setting_version_accumilator == 'max':
            for idx, sample in per_sample_tokenwise_kl.items():
                setting_aggregated_per_sample_tokenwise_kl[idx].append(
                    np.max(np.array(sample), axis=0)
                )

        if setting_version_accumilator == 'avg':
            for idx, sample in per_sample_tokenwise_kl.items():
                setting_aggregated_per_sample_tokenwise_kl[idx].append(
                    np.mean(np.array(sample), axis=0)
                )

        if aug_version_accumilator == 'max':
            for _, sample in setting_aggregated_per_sample_tokenwise_kl.items():
                per_sample_kld.append(np.max(np.array(sample), axis=0))

        if aug_version_accumilator == 'avg':
            for _, sample in setting_aggregated_per_sample_tokenwise_kl.items():
                per_sample_kld.append(np.mean(np.array(sample), axis=0))

    # Min-K
    for _ratio in ratio:
        key = f"Min_{_ratio}_{cfg.suffix}_kld_{aug_version_accumilator}ed_aug_{setting_version_accumilator}ed_versions"
        min_k_kl = list()
        for sample in per_sample_kld:
            k_length = int(_ratio * len(sample))
            if k_length == 0:
                k_length = 1
            min_k_kl.append(np.mean(np.sort(sample)[:k_length]))
            if key not in result: 
                result[key] = list()
            result[key] = min_k_kl

    return result


def kl_div_per_token(org_probs, org_log_probs, aug_log_probs):
    return org_probs * (org_log_probs - aug_log_probs)
    