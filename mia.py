import numpy as np
from collections import defaultdict

"""
Main MIA entry point
"""
import os
import json
import hydra
from src.eval import evaluate
from src.inference import inference
from src.data import get_mod_infer_data
from src.data import get_generation_data
from src.model import generate, mod_infer_batch
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from src.misc import save_to_json, load_conversation_template

@hydra.main(version_base=None, config_path="./config", config_name="run_img")
def main(cfg):

    # Print cfgs
    if cfg.test_run.test_run:
        print("This is a test run")
        print(f"Only Running on first {cfg.inference.test_number_of_batches} batches")
        
    print(cfg)
    print("Augmentations Used:")
    print(cfg.data.augmentations)
    
    if cfg.img_metrics.get_meta_values > 0:
        print(f"Requested meta values of first {cfg.img_metrics.get_meta_values} of each class")
        
    if cfg.img_metrics.get_token_labels > 0 :
        print(f"Requested token labels of first {cfg.img_metrics.get_token_labels} of each class")

    # Load the target model
    model_name = get_model_name_from_path(cfg.target_model.model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        cfg.target_model.model_path, 
        cfg.target_model.model_base, 
        model_name, 
        gpu_id=cfg.target_model.gpu_id,
        cache_dir=cfg.path.cache_dir
    )
    conv_mode = load_conversation_template(model_name)


    # Generation data
    text = cfg.prompt.text

    # gen_data = get_generation_data(cfg, tokenizer, image_processor, text, model.config, conv_mode)
    # indices, descriptions = generate(model, tokenizer, gen_data, cfg)
    
    from gen_text import sentences as descriptions

    mod_infer_data = get_mod_infer_data(cfg, descriptions, tokenizer, image_processor, text, model.config, conv_mode)
    
    # If we want to get meta values and labels for some samples (first x members and nonmembers) find the indecies these samples live
    sampled_indices = []
    class_labels = mod_infer_data["label"]
    
    if cfg.test_run.test_run:
        class_labels = class_labels[: (cfg.inference.batch_size * cfg.inference.test_number_of_batches)]
    
    if cfg.img_metrics.get_meta_values > 0:
        sampled_indices = np.sort(np.concatenate([
        np.where(np.array(class_labels) == 1)[0][:cfg.img_metrics.get_meta_values],
        np.where(np.array(class_labels) == 0)[0][:cfg.img_metrics.get_meta_values]
        ]))
    
    print(f"Sampled Indecies: {sampled_indices}")
            
        
        
        

    preds, proc_meta, global_token_labels = inference(model, tokenizer, mod_infer_data, sampled_indices, cfg)
    
    
    auc, acc, auc_low = evaluate(preds, mod_infer_data["label"], "img", cfg)

    # Save
    save_to_json(preds, "preds", cfg)
    save_to_json(auc, "auc", cfg)
    save_to_json(acc, "acc", cfg)
    save_to_json(auc_low, "auc_low", cfg)
    
    
    if cfg.img_metrics.get_meta_values > 0:
        save_to_json(sampled_indices.tolist(), "sampled_examples",cfg)
        save_to_json(class_labels, "class_labels", cfg)
        save_to_json(proc_meta, "processed_meta_values", cfg)
        
        
    if cfg.img_metrics.get_token_labels > 0:
        save_to_json(sampled_indices.tolist(), "sampled_examples",cfg)
        save_to_json(class_labels, "class_labels", cfg)
        save_to_json(global_token_labels, "token_labels", cfg)

if __name__ == "__main__":
    main()