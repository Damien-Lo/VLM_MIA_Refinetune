import torch

from src.metrics.utils import kl_divergence

def get_img_metric(cfg, goals, meta_metrics):
    """
    meta_metrics: batched or full meta_metrics dictionary
    meta_metrics : {
        "img" : {
            "ppl" :  [ 1D list - num of samples (batch_size) & num of input_ids],
            "entropies" : [ 2D list - num of samples (batch_size) * num of input_ids]
        },
        "desc": {
        ...
        }
    }
    """

    predictions = dict()
    
    for _part, _part_meta_metrics in meta_metics.items():
        predictions[_part] = dict()

        if "aug_kl" in cfg.img_metrics.metrics_to_use:
            kl_1 = kl_divergence()

