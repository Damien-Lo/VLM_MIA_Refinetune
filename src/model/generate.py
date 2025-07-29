

def generate_all_response(model, tokenizer, dataloader, num_gen_tokens):
    outputs = list()
    for _idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].cuda()
        image_tensors = batch["image_tensor"].cuda()
        image_sizes = batch["image_size"].cuda()

        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=num_gen_tokens,
            use_cahce=True
        )

        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for _out_text in output_text:
            outputs.append(_out_text.strip())

    return outputs

def generate_a_batch(model, tokenizer, batch, num_gen_tokens):
    outputs = list()
    input_ids = batch["input_ids"].cuda()
    image_tensors = batch["image_tensor"].cuda()
    image_sizes = batch["image_size"].cuda()

    output_ids = model.generate(
        input_ids,
        images=images_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        max_new_tokens=num_gen_tokens,
        use_cahce=True
    )

    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    for _out_text in output_text:
        outputs.append(_out_text.strip())
    
    return outputs