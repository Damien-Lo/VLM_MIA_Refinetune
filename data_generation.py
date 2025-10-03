import os
import json
import hydra
import torch
from src.data import get_generation_data
from src.model import generate
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from src.misc import load_conversation_template

"""
A debug code for data generation

"""
@hydra.main(version_base=None, config_path="./config", config_name="run_img")
def main(cfg):

    print("Full Config Paramters:")
    print(cfg)
    print("\n")
    print("AUGMENTATIONS USED:")
    print(cfg.data.augmentations)
    print("\nREQUESTED DATA")        
        
    print('''
          \n \n
          ==================================================
                            LOADING MODEL
          ==================================================
          '''
          )

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
    gen_data = get_generation_data(cfg, tokenizer, image_processor, text, model.config, conv_mode)
    idxs, descriptions = generate(model, tokenizer, gen_data, cfg)
    
    # Save the generated text
    save_path = os.path.join(cfg.path.output_dir, "generation", str(cfg.data.subset))
    os.makedirs(save_path, exist_ok=True)

    sentences = {
        "idxs": idxs,
        "sentences": descriptions
    }

    import json
    with open(os.path.join(save_path, "sentences.json"), 'w') as f:
        json.dump(sentences, f, indent=2)


    
if __name__ == "__main__":
    main()

