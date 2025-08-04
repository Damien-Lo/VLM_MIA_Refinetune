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

    print(cfg)

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

    preds = inference(model, tokenizer, mod_infer_data, cfg)
    
    auc, acc, auc_low = evaluate(preds, mod_infer_data["label"], "img")

    # Save
    save_to_json(preds, "preds", cfg)
    save_to_json(auc, "auc", cfg)
    save_to_json(acc, "acc", cfg)
    save_to_json(auc_low, "auc_low", cfg)

if __name__ == "__main__":
    main()