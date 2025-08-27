import os
import json
from tqdm import tqdm
from src.model import mod_infer_batch
from src.model.infer import BatchProcessor
from src.metrics import get_meta_metrics_by_part, get_img_metric_by_parts
import numpy as np
from collections import defaultdict

def inference(model, tokenizer, dataset, sampled_indices, cfg):
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
    sampled_proc_meta = dict()
    global_token_labels = list()
    
    


    for _part in parts:
        global_pred[_part] = dict()
        sampled_proc_meta[_part] = dict()
        
        
    sample_map = defaultdict(list)
    for idx in sampled_indices:
        sample_map[int(idx/cfg.inference.batch_size)].append(idx % cfg.inference.batch_size)
    print(f"Sample Map: {sample_map}")
        
    
    for b_idx, batch in enumerate(tqdm(batch_processor,
                                       total=len(batch_processor),
                                       desc="Running inference",
                                       unit="batch")):
        
        
        print(f"\n\nRunning Batch: {b_idx}")
        
        batch_sampled_indices = sample_map[b_idx]
            
        
        # For Test Run, just run 1 batch
        if cfg.test_run.test_run and b_idx >= cfg.inference.test_number_of_batches:
            break
            
        
        target_parts, total_token_labels = mod_infer_batch(model, batch, tokenizer,
                                       parts=parts,
                                       use_augmentation=cfg.inference.use_augmentation)
        
        global_token_labels.extend(total_token_labels)


        # Process separately for each part
        for _part in parts:
            _meta_metrics = get_meta_metrics_by_part(target_parts,
                                                     part=_part,
                                                    cfg=cfg)            
            _pred, _proc_meta = get_img_metric_by_parts(_meta_metrics, batch_sampled_indices, cfg)
            
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
                        
                    
                        
            # Getting raw token values 
            # Final Structure of Meta Values:
            '''
            {
                part: {
                    metric_name: {
                        aug_title:
                            [setting][sample][kl_values]  # 3D array
                    }
                }
            }
            '''
            if b_idx in sample_map and cfg.img_metrics.get_meta_values >= 0:
                for metric_name, meta_dict in _proc_meta.items():
                    if metric_name not in sampled_proc_meta[_part]:
                            sampled_proc_meta[_part][metric_name] = defaultdict(list)
                    
                    for aug, meta in meta_dict.items():
                        for sample_idx, kld_for_all_settings in enumerate(meta):
                            if sample_idx in sample_map[b_idx]:
                                sampled_proc_meta[_part][metric_name][aug].append(kld_for_all_settings)
                    
                    
                            
                    
     
    # If We want want the meta values, only return the values at the desired sampled_indices, else return empty array         
    if cfg.img_metrics.get_token_labels > 0 or cfg.img_metrics.get_meta_values > 0:
        global_token_labels = [global_token_labels[i] for i in sampled_indices]
                
    if cfg.img_metrics.get_token_labels <= 0 or cfg.img_metrics.get_meta_values <= 0:
        global_token_labels = []
        sampled_proc_meta = {}
        
        
    for part, metric_dict in sampled_proc_meta.items():
        for metric_name, metric_values in metric_dict.items():
            for aug, value_array in metric_values.items():
                
                
                if isinstance(value_array, np.ndarray):
                    sampled_proc_meta[part][metric_name][aug] = value_array.tolist()
        
    return global_pred, sampled_proc_meta, global_token_labels 
