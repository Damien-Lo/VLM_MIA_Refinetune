import os
import json
from tqdm import tqdm
from src.model import mod_infer_batch
from src.model.infer import BatchProcessor
from src.metrics import get_meta_metrics_by_part, get_img_metric_by_parts

def inference(model, tokenizer, dataset, cfg):
    """
    For each batch
        1. Conduct mod-infer
        2. Obtain meta-metrics
        3. Obtain img_metrics
        4. Combine img_metrics
    """

    batch_processor = BatchProcessor(dataset=dataset,
                                     batch_size=cfg.inference.batch_size,
                                     eos_token_id=tokenizer.eos_token_id,
                                     use_augmentation=cfg.inference.use_augmentation)
    parts = cfg.img_metrics.parts
    global_pred = dict()
    for _part in parts:
        global_pred[_part] = dict()
    
    for b_idx, batch in enumerate(tqdm(batch_processor,
                                       total=len(batch_processor),
                                       desc="Running inference",
                                       unit="batch")):
        
        # For Test Run, just run 1 batch
        if cfg.test_run.test_run and b_idx >= 1:
            break
            
        
        target_parts = mod_infer_batch(model, batch, tokenizer,
                                       parts=parts,
                                       use_augmentation=cfg.inference.use_augmentation)

        # Process separately for each part
        for _part in parts:
            _meta_metrics = get_meta_metrics_by_part(target_parts,
                                                     part=_part,
                                                    cfg=cfg)            
            _pred = get_img_metric_by_parts(_meta_metrics, cfg)
            
            for metric_name, metric_values in _pred.items():
                if metric_name not in global_pred[_part]:
                    # Initialize the dictionary
                    if isinstance(metric_values, list):
                        global_pred[_part][metric_name] = list()
                    elif isinstance(metric_values, dict):
                        global_pred[_part][metric_name] = dict()
                        for metric_key in metric_values.keys():
                            global_pred[_part][metric_name][metric_key] = list()
                if isinstance(metric_values, list):
                    global_pred[_part][metric_name].extend(metric_values)
                elif isinstance(metric_values, dict):
                    for _key, _value in metric_values.items():
                        global_pred[_part][metric_name][_key].extend(_value)

    return global_pred

