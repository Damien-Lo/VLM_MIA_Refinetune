import os
import json

def save_to_json(dict_obj, filename, cfg):
    output_dir = cfg.path.output_dir
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{filename}.json")
    with open(save_path, "w") as f:
        json.dump(dict_obj, f, indent=4)
    print(f"Saved {filename} to {save_path}")


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
