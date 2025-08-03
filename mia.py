"""
Main MIA entry point
"""

import hydra
from src.data import get_generation_data
from src.model import generate, mod_infer_batch
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

def load_conversation_template(model_name):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    return conv_mode


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
    gen_data = get_generation_data(cfg, tokenizer, image_processor, text, model.config, conv_mode)

    indices, descriptions = generate(model, tokenizer, gen_data, cfg)


    for d in descriptions:
        print(d)

if __name__ == "__main__":
    main()