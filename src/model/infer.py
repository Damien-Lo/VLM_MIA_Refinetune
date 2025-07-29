from src.metrics import get_meta_metrics

def mod_infer(cfg, model, batch, tokenizer, goal):
    """
    mod_infer function.
    With the instruction, image and the descriptions,
    Do an inference using them.
    Return the batch meta metics (for all parts)

    cfg: configs
    model: target model
    batch: a batch of the dataloader from get_mod_infer_data
    tokenizer: tokenizer,
    """

    meta_metrics = list()

    input_ids = batch["input_ids"].cuda()
    image_tensors = batch["image_tensors"].cuda()
    attention_masks = batch["attention_masks"].cuda()

    # These are masked indices
    img_mask = batch["img"].cuda()
    img_loss_mask = batch["img_loss"].cuda()
    inst_desp_mask = batch["inst_desp"].cuda()
    inst_mask = batch["inst"].cuda()
    desp_mask = batch["desp"].cuda()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            image_tensors=image_tensors,
            image_sizes=image_sizes
        )
    logits = outputs.logits

    img_loss_slices = logits[:, img_loss_mask, :]
    img_target = torch.nn.functional.softmax(img_loss_slices, axis=-1)
    max_indices = torch.argmax(img_target, axis=-1)

    mix_input_ids_mask = img_mask
    _last_true_list = list()
    for _idx in range(img_mask.size(0)):
        last_true = torch.where(img_mask[_idx] == True)[0][-1]
        mix_input_ids_mask[_idx, last_true+1] = True
        _last_true_list.append(last_true+1)

    mix_input_ids = input_ids * (~mix_input_ids_mask) * attention_mask
    for _idx in range(img_mask.size(0)):
        _zeros = torch.where(mix_input_ids_mask[_idx] == 1)[0]
        mix_input_ids[_idx, _zeros] = max_indices[_idx]

    # Remove the first element after the end of the imag token
    img_last_mask = torch.ones_like(img_mask, dtype=torch.bool)
    for _idx, _last in enumerate(_last_true_list):
        img_last[_idx, _last] = 0
    
    mix_input_ids = mix_input_ids[img_last_mask]
    mix_attention_mask = attention_mask[img_last_mask]

    target_goals = dict()
    for g in goal:
        if g == "img":
            target_goals[g] = dict()
            target_goals[g]["logits"] = logits[:, img_mask, :]
        elif g == "inst_desp":
            target_goals[g] = dict()
            target_goals[g]["logits"] = logits[:, inst_desp_mask, :]
        elif g == "inst":
            target_goals[g] = dict()
            taget_goals[g]["logits"] = logits[:, inst_mask, :]
        elif g == "desp":
            target_goals[g] = dict()
            target_goals[g]["logits"] = logits[:, desp_mask, :]
        else:
            raise ValueError(f"Not supported goal {g}")

        target_goals[g]["probabilities"] = torch.nn.functional.softmax(target_goals[g]["logits"], dim=-1)
        target_goals[g]["log_probabilities"] = torch.nn.functional.log_softmax(target_goals[g]["logits"], dim=-1)
    
    return mix_input_ids, mix_attention_mask, target_goals


def inference(cfg, model, dataloader):
    goals = cfg.image_metrics.goals

    for _ in range( )