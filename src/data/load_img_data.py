from dataets import Dataset
from datasets import load_dataset

from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


from src.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

def load_image(image_file):
    """
    Opens an image in a image_file,
    returns PIL.Image instance
    """
    if isinstance(image_file, Image.Image):  
        return image_file.convert("RGB")  
    
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def load_image_data(cfg, tokenizer, text, model_config, conv):
    """
    cfg :  dataset config
    tokenizer: tokenizer instance
    text: Input instruction text (Describe this image)
    model_config: model.config
    conv: conv from cfg.target_model
    """
    _dataset = load_dataset(path=cfg.dataset_name,
                          name=cfg.subset,
                          split=cfg.split,
                          cache_dir=cache_dir)
    _dataset = _dataset.add_column("indices", list(range(len(_dataset))))
    _dataset = _dataset.map(convert_to_input_ids,
                           batched=True,
                           load_from_cache_file=False,
                           fn_kwargs={
                               "tokenizer": tokenizer,
                               "instruction": text,
                               "model_config": model_config,
                               "conv": conv,
                               "cfg": cfg
                          })
    return _dataset

def get_mod_infer_data(cfg, descriptions, tokenizer, text, model_config, conv):
    """
    cfg :  dataset config
    descriptions: generated responses
    tokenizer: tokenizer instance
    text: Input instruction text (Describe this image)
    model_config: model.config
    conv: conv from cfg.target_model
    """
    _dataset = load_dataset(path=cfg.dataset_name,
                          name=cfg.subset,
                          split=cfg.split,
                          cache_dir=cache_dir)
    _dataset = _dataset.add_column("indices", list(range(len(_dataset))))
    _dataset = _dataset.add_column("desc", descriptions)
    _dataset = _dataset.map(convert_to_mod_infer,
                           batched=True,
                           load_from_cache_file=False,
                           fn_kwargs={
                               "tokenizer": tokenizer,
                               "instruction": text,
                               "model_config": model_config,
                               "conv": conv,
                               "cfg": cfg
                          })
    return _dataset

def convert_to_input_ids(examples, tokenizer, instruction, model_config, conv, cfg):
    """
    Preprocess the generation dataset
    """
    image_paths = examples["image"]
    inage_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    qs = instruction
    if IMAGE_PLACEHOLDER in qs:
        if model_config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model_cfg.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    
    # Load The conversation with the query
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    all_input_ids = list()
    all_image_tensors = list()
    all_image_sizes = list()
    for _image_path in image_paths:
        images = load_images([_image_path])
        image_sizes = [x.size for x in images]
        image_tensor = process_images(
            images,
            image_processor,
            model_config
        )

        # Tokenize Image Based On Prompt
        input_ids, prompt_chunks = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        all_input_ids.append(input_ids)
        all_image_tensors.append(image_tensor.squeeze(0))
        all_image_sizes.append(image_sizes[0])

    return {
        "indices": torch.tensor(idx),
        "input_ids": torch.tensor(all_input_ids),
        "image_tensors": torch.tensor(all_image_tensors),
        "image_sizes": torch.tensor(all_image_sizes)
    }

def convert_to_mod_infer(examples, tokenizer, instruction, model_config, conv, cfg):
    """
    Preprocess the mod_infer dataset
    example: batched rows of dataset (has image paths and the descriptions paired with each image)
    instruction: instruction
    descriptions: generated descriptions
    """
    image_paths = examples["image"]
    descriptions = examples["desc"]
    qs = instruction
    
    inage_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model_config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model_cfg.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    
    all_input_ids = list()
    all_image_tensors = list()
    all_attention_mask = list()
    all_image_sizes = list()
    all_img_slices = list()
    all_img_loss_slices = list()
    all_inst_desp_slices = list()
    all_inst_slices = list()
    all_desp_slices = list()

    for _image_path, _description in zip(image_paths, descriptions):
        
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], _description)
        prompt = conv.get_prompt()[:-4]

        images = load_images([_image_path])
        image_sizes = [x.size for x in images]
        image_tensor = process_images(
            images,
            image_processor,
            model_config
        )

        # Tokenize Image Based On Prompt
        input_ids, prompt_chunks = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        
        # Manual padding
        padding_size = cfg.mod_infer_pad_size - input_ids.size(0)
        padding = tokenizer.eos_token_id * torch.ones(padding_size)
        padded_input_ids = torch.cat([input_ids, padding])
        attention_mask = torch.cat([torch.ones_like(input_ids), torch.zeros_like(padding)], dtype=torch.long)

        all_input_ids.append(padded_input_ids)
        all_image_tensors.append(image_tensor)
        all_attention_mask.append(attention_mask)
        all_image_sizes.append(image_sizes[0])

        desc_encoding = tokenizer(_description, return_tensors="pt", add_special_tokens = False).to(device).input_ids

        img_slice = slice(len(prompt_chunks[0]),-len(prompt_chunks[-1])+1)
        inst_desc = slice(-len(prompt_chunks[-1])+1, input_ids.size(0))
        inst = slice(-len(prompt_chunks[-1])+1,-desc_encoding.shape[1])
        desc = slice(-desc_encoding.shape[1], input_ids.size(0))

        img_slice_mask = torch.zeros_like(padded_input_ids, dtype=torch.bool)
        img_slice_mask[img_slice] = 1
        img_loss_mask = torch.zeros_like(padded_input_ids, dtype=torch.bool)
        img_loss_mask[img_slice.start-1:img_slice.stop-1] = 1
        inst_desc_mask = torch.zeros_like(padded_input_ids, dtype=torch.bool)
        inst_desc_mask[inst_desc] = 1
        inst_mask = torch.zeros_like(padded_input_ids, dtype=torch.bool)
        inst_mask[inst] = 1
        desc_mask = torch.zeros_like(padded_input_ids, dtype=torch.bool)
        desc_mask[desc] = 1

        all_img_slices.append(img_slice_mask)
        all_img_loss_slices.append(img_loss_mask)
        all_inst_desp_slices.append(inst_desc_mask)
        all_inst_slices.append(inst_mask)
        all_desp_slices.append(desc_mask)

    return {
        "input_ids": torch.tensor(all_input_ids),
        "image_tensors": torch.tensor(all_image_tensors),
        "attention_masks" : torch.tensor(all_attention_mask),
        "image_sizes" : torch.tensor(all_image_sizes),
        "img": torch.tensor(all_img_slices),
        "img_loss": torch.tensor(all_img_loss_slices),
        "inst_desp": torch.tensor(all_inst_desp_slices),
        "inst": torch.tensor(all_inst_slices),
        "desp": torch.tensor(all_desp_slices)
    }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

        all_input_ids.append(input_ids)
        all_image_tensors.append(image_tensor.squeeze(0))

    return {
        "indices": torch.tensor(idx),
        "input_ids": torch.tensor(all_input_ids),
        "image_tensors": torch.tensor(all_image_tensors)
    }


def convert_to_mod_infer_with_transformation():
