import torch
import statistics
import numpy as np
from src.metrics.utils import kl_divergence

def get_img_metric_by_parts(meta_metrics, cfg):
    """
    meta_metrics: batched or full meta_metrics dictionary
    meta_metrics : {
        "ppl" : {
            "aug_1" : [
                [sample1, sample2, sample3],
                [sample1, sample2, sample3]
            ],
            "aug_2" : [
            
            ]
        },
        "entropies": {
            "aug_1" : [
                [    
                    [ entropy1, entropy2, ...]
                ]      
            ] 
        }
        
    }

    part is not necessary because it was already processed from the get_meta_metrics_by_part function.
    """

    pred = dict()

    if "aug_kl" in cfg.img_metrics.metrics_to_use:
        _aug_kl = aug_kl(meta_metrics["probabilities"], meta_metrics["log_probabilities"], cfg.img_metrics.metrics.aug_kld)        
        pred["aug_kl"] = _aug_kl

    if "mink" in cfg.img_metrics.metrics_to_use:
        _min_k = min_k(meta_metrics["all_prob"], cfg.metrics.mink)
        pred["min_k"] = _min_k

    if "mod_entro" in cfg.img_metrics.metrics_to_use:
        _mod_entropy = mod_entropy(meta_metrics["modified_entropies"])
        pred["mod_renyi_1_entro"] = _mod_entropy

    if "mod_renyi_05" in cfg.img_metrics.metrics_to_use:
        _mod_renyi_05 = mod_renyi(meta_metrics["modified_entropies_alpha_05"])
        pred["mod_renyi_05_entro"] = _mod_renyi_05

    if "mod_renyi_2" in cfg.img_metrics.metrics_to_use:
        _mod_renyi_2 = mod_renyi(meta_metrics["modified_entropies_alpha_2"])
        pred["mod_renyi_2_entro"] = _mod_renyi_2

    if "max_prob_gap" in cfg.img_metrics.metrics_to_use:
        _max_prob_gap = max_prob_gap(meta_metrics["gap_probs"])
        pred["max_prob_gap"] = _max_prob_gap

    if "max_k_renyi_1_entro" in cfg.img_metrics.metrics_to_use:
        _max_k_renyi_1_entro = max_entropy(meta_metrics["entropies"], cfg.img_metrics.metrics.max_k_renyi_1_entro)
        pred["max_k_renyi_1_entro"] = _max_k_renyi_1_entro

    if "max_k_renyi_2_entro" in cfg.img_metrics.metrics_to_use:
        _max_k_renyi_2_entro = max_entropy(meta_metrics["renyi_2_entro"], cfg.img_metrics.metrics.max_k_renyi_2_entro)
        pred["max_k_renyi_2_entro"] = _max_k_renyi_2_entro

    if "max_k_renyi_05_entro" in cfg.img_metrics.metrics_to_use:
        _max_k_renyi_05_entro = max_entropy(meta_metrics["renyi_05_entro"], cfg.img_metrics.metrics.max_k_renyi_05_entro)
        pred["max_k_renyi_05_entro"] = _max_k_renyi_05_entro

    if "max_k_renyi_inf" in cfg.img_metrics.metrics_to_use:
        _max_k_renyi_inf = max_entropy(meta_metrics["renyi_inf_entro"], cfg.img_metrics.metrics.max_k_renyi_inf_entro)
        pred["max_k_renyi_inf"] = _max_k_renyi_inf

    if "min_k_renyi_1_entro" in cfg.img_metrics.metrics_to_use:
        _min_k_renyi_1_entro = min_entropy(meta_metrics["entropies"], cfg.img_metrics.metrics.min_k_renyi_1_entro)
        pred["min_k_renyi_1_entro"] = _min_k_renyi_1_entro

    if "min_k_renyi_2_entro" in cfg.img_metrics.metrics_to_use:
        _min_k_renyi_2_entro = min_entropy(meta_metrics["renyi_2_entro"], cfg.img_metrics.metrics.min_k_renyi_2_entro)
        pred["min_k_renyi_2_entro"] = _min_k_renyi_2_entro

    if "min_k_renyi_05_entro" in cfg.img_metrics.metrics_to_use:
        _min_k_renyi_05_entro = min_entropy(meta_metrics["renyi_05_entro"], cfg.img_metrics.metrics.min_k_renyi_05_entro)
        pred["min_k_renyi_05_entro"] = _min_k_renyi_05_entro

    return pred

def aug_k(probabilities, log_probabilities, cfg):

    result = list()
    aug_names = list(probabilities.keys())
    if len(aug_names) == 1:
        return None
    
    for _sample_idx, (_orig_prob, _orig_log_prob) in enumerate(zip(probabilities["orig"][0], log_probabilities["orig"][0])):
        # iterating over samples
        _kld = list()
        for aug_name in aug_names:
            if aug_name == "orig":
                continue
            _kld.append(kl_divergence(_orig_prob.cpu().numpy(),
                                      _orig_log_prob.cpu().numpy(),
                                      probabilities[aug_name][0][_sample_idx]))
                                     # change [0] to something else or add a new logic if we use multiple params for each augmentation.
        result.append(-statistics.mean(_kld))
    return result


def min_k(all_prob, cfg):
    
    ratio = cfg.ratio
    # result = dict()
    # for aug_name, aug_list in all_prob.items():
    #     result[aug_name] = list()
    #     for _aug in aug_list:
    #         _result = dict()
    #         for _ratio in ratio:
    #             k_length = int(len(_prob)*_ratio)
    #             if k_length == 0:
    #                 k_length = 1
    #             _result[f"Min_{_ratio*100}% prob"] = list()
    #             for _prob in _aug:
    #                 token_prob = np.sort(_prob)[:k_length]
    #                 _result.append(-1 * np.mean(_prob).item())
    #         result[aug_name].append(_result)
     

    result = dict()
    for _ratio in ratio: 
        _key = f"Min_{_ratio*100}% prob"
        result[_key] = list()
        for _prob in all_prob["orig"][0]:
            k_length = int(len(_prob)*_ratio)
            if k_length == 0:
                k_length = 1
            topk_prob = np.sort(_prob)[:k_length]
            result[_key].append(-1 * np.mean(topk_prob).item())

    return result

def mod_entropy(mod_entropy):
    
    result = list()
    for _sample in mod_entropy["orig"][0]:
        result.append(np.nanmean(_sample).item())

    return result

def mod_renyi(renyi):

    result = list()
    for _sample in renyi["orig"][0]:
        result.append(np.nanmean(_sample).item())
    
    return result

def max_prob_gap(gap_p):

    result = list()
    for _sample in gap_p["orig"][0]:
        result.append(-np.mean(_sample).item())
    
    return result

def max_entropy(entropy, cfg):
    
    ratio = cfg.ratio
    result = dict()

    for _ratio in ratio: 
        _key = f"Max_{_ratio*100}% "+cfg.suffix
        result[_key] = list()
        for _entro in entropy["orig"][0]:
            k_length = int(len(_entro)*_ratio)
            if k_length == 0:
                k_length = 1
            topk_prob = np.sort(_entro)[-k_length:]
            result[_key].append(np.mean(topk_prob).item())
    
    return result

def min_entropy(renyi, cfg):
    
    ratio  = cfg.ratio
    result = dict()

    for _ratio in ratio:
        _key = f"Min_{_ratio*100}% "+cfg.suffix
        for _renyi in renyi["orig"][0]:
            k_length = int(len(_renyi)*_ratio)
            if k_length == 0:
                k_length = 1
            topk_prob = np.sort(_renyi)[:k_length]
            result[_key].append(np.mean(topk_prob).item())
    
    return result