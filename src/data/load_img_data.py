from dataets import Dataset
from datasets import load_dataset
from src.data.augmentations import get_augmentations

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
    _dataset = _dataset.map(convert_to_generation_input_ids,
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


def convert_to_generation_input_ids(examples, tokenizer, instruction, model_config, conv, cfg):
    """
    Preprocess the generation dataset
    """
    image_paths = examples["image"]
    indices = examples["indices"]
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

    all_indices = list()
    all_input_ids = list()
    all_image_tensors = list()
    all_image_sizes = list()
    for _image_path, _indices in zip(image_paths, indices):
        images = load_images([_image_path])
        image_sizes = [x.size for x in images]
        image_tensor = process_images(
            images,
            image_processor,
            model_config
        )

        # Tokenize Image Based On Prompt
        input_ids, prompt_chunks = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        all_indices.append(_indices)
        all_input_ids.append(input_ids)
        all_image_tensors.append(image_tensor.squeeze(0))
        all_image_sizes.append(image_sizes[0])

    return {
        "indices": all_indices,
        "input_ids": all_input_ids,
        "image_tensors": all_image_tensors,
        "image_sizes": all_image_sizes
    }

def convert_to_aug_generation_input_ids(examples, tokenizer, instruction, model_config, conv, cfg):
    """
    Preprocess the generation dataset
    """
    indices = examples["indices"]
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

    all_indices = list()
    all_input_ids = list()
    all_orig_image_tensors = list()
    all_image_sizes = list()
    all_aug_images = list()
    aug_dict = get_augmentations(cfg)
    for _image_path, _indices in zip(image_paths, indices):
        image = load_image(_image_path)  # Loading just one image
        orig_image_tensor = process_images(
            [images],
            image_processor,
            model_config
        )
        # Get augmented views
        image_sizes = image.size()
        aug_imgs = dict()
        for k, aug_f_list in aug_imgs:
            _aug_tensor_list = list()
            for _aug_f in aug_f_list:
                _aug_image  = _aug_f(images[0]))
                _aug_image_tensor = process_images(
                    [_aug_image],
                    image_processor,
                    model_config
                )
                _aug_tensor_list.append(_aug_image_tensor)
            aug_imgs[k] = _aug_tensor_list
        
        # Tokenize Image Based On Prompt
        input_ids, prompt_chunks = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        all_indices.append(_indices)
        all_aug_images.append(aug_imgs)
        all_input_ids.append(input_ids)
        all_orig_image_tensors.append(orig_image_tensor.squeeze(0))
        all_image_sizes.append(image_sizes[0])

    return {
        "indices": all_indices,
        "input_ids": all_input_ids,
        "orig_image_tensors": all_orig_image_tensors,
        "image_sizes": all_img_sizes,
        "aug_image_tensors": all_aug_images
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
    all_prompt_0 = list()
    all_prompt_1 = list()
    all_desc_shape = list()

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

        all_prompt_0.append(prompt_chunks[0])
        all_prompt_1.append(prompt_chunks[-1])
        all_desc_shape.append(desc_encoding.shape[1])

    return {
        "input_ids": all_input_ids,
        "image_tensors": all_image_tensors,
        "attention_masks" : all_attention_mask,
        "image_sizes" : all_image_sizes,
        "prompt_0" : all_prompt_0,
        "prompt_1" :  all_prompt_1,
        "desc_shape": all_desc_shape
    }


def convert_to_augmention_mod_infer(examples, tokenizer, instruction, model_config, conv, cfg):
    """
    Preprocess the mod_infer dataset
    example: batched rows of dataset (has image paths and the descriptions paired with each image)
    instruction: instruction
    descriptions: generated descriptions
    """
    indices = examples["indices"]
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

    all_indices = list()
    all_input_ids = list()
    all_orig_image_tensors = list()
    all_aug_images = list()
    all_image_sizes = list()
    all_prompt_0 = list()
    all_prompt_1 = list()
    all_desc_shape = list()
    
    aug_dict = get_augmentations(cfg)

    for _indices, _image_path, _description in zip(indices, image_paths, descriptions):
        
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], _description)
        prompt = conv.get_prompt()[:-4]

        images = load_images([_image_path])
        image_sizes = [x.size for x in images]
        orig_image_tensor = process_images(
            images,
            image_processor,
            model_config
        )

        aug_imgs = dict()
        for k, aug_f_list in aug_imgs:
            _aug_tensor_list = list()
            for _aug_f in aug_f_list:
                _aug_image = _aug_f(images[0])
                _aug_image_tensor = process_images(
                    [_aug_images],
                    image_processor,
                    model_config
                )
                _aug_tensor_list.append(_aug_image_tensor)
            aug_imgs[k] = _aug_tensor_list
        

        # Tokenize Image Based On Prompt
        input_ids, prompt_chunks = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        
        all_indices.append(_indices)
        all_input_ids.append(padded_input_ids)
        all_orig_image_tensors.append(image_tensor)
        all_image_sizes.append(image_sizes[0])
        all_aug_images.append(aug_imgs)

        desc_encoding = tokenizer(_description, return_tensors="pt", add_special_tokens = False).to(device).input_ids

        all_prompt_0.append(prompt_chunks[0])
        all_prompt_1.append(prompt_chunks[-1])
        all_desc_shape.append(desc_encoding.shape[1])

    return {
        "input_ids": all_input_ids,
        "orig_image_tensors": all_image_tensors,
        "aug_image_tensors": all_aug_images,
        "image_sizes" : all_image_sizes,
        "prompt_0": all_prompt_0,
        "prompt_1": all_prompt_1,
        "desc_shape": all_desc_shape
    }