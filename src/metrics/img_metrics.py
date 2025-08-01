import torch

from src.metrics.utils import kl_divergence

def get_img_metric_by_parts(cfg, meta_metrics):
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
    """

    