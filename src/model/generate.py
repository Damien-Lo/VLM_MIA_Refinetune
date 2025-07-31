import torch

def generate_all_response(model, tokenizer, dataloader, num_gen_tokens):
    outputs = list()
    for _idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].cuda()
        image_tensors = batch["image_tensor"].cuda()
        image_sizes = batch["image_size"].cuda()

        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=num_gen_tokens,
            use_cache=True
        )

        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for _out_text in output_text:
            outputs.append(_out_text.strip())

    return outputs

def generate_a_batch(model, tokenizer, batch, num_gen_tokens, use_augmentation):
    if use_augmentation:
        outputs = dict()
        input_ids = batch["input_ids"]
        image_sizes = batch["image_sizes"].cuda()

        # 1.Generate using the original images
        orig_image_tensors = batch["orig_image_tensors"].cuda()

        output_ids = model.generate(
            input_ids,
            images=orig_image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=num_gen_tokens,
            use_cache=True
        )
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        outputs["orig"] = list()
        for _out_text in output_text: 
            outputs["orig"].append(_out_text)

        # 2. Generate using each augmented view
        for k, aug_imgs in batch["aug_image_tensors"].items():
            outputs[k] = list()
            for _aug_img in aug_imgs:
                output_ids = model.generate(
                    input_ids,
                    images=_aug_img.cuda(),
                    image_sizes=image_sizes,
                    do_sample=False,
                    max_new_tokens=num_gen_tokens,
                    use_cache=True
                )
                output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                for _out_text in output_text: 
                    outputs[k].append(_out_text)        

    else:
        outputs = list()
        input_ids = batch["input_ids"].cuda()
        image_tensors = batch["image_tensors"].cuda()
        image_sizes = batch["image_sizes"].cuda()

        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            max_new_tokens=num_gen_tokens,
            use_cache=True
        )

        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        for _out_text in output_text:
            outputs.append(_out_text.strip())
        
    return outputs

class BatchProcessor:
    def __init__(self, dataset, batch_size, use_augmentation):
        self.dataset = dataset
        self.use_augmentation = use_augmentation
        self.batch_size = batch_size
        self.last_batch = 0
        self.current_batch = 0
        self.num_batch = int(len(dataset)/self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.use_augmentation:
            return self._get_augmented_batch()
        else:
            return self._get_normal_batch()
        
    def _get_normal_batch(self):
        """
        indices: 1D vector has index of samples in the batch
        input_ids : 2D vector [sample_idx, input_ids]
        image_tensors: 4D vector [sample_idx, channel_dim, width, height]
        image_sizes: 3D vector [sample_idx, height, width]
        """
        if self.last_batch == self.num_batch:
            raise StopIteration
 
        indices = list()
        input_ids = list()
        image_tensors = list()
        image_sizes = list()
        
        batch_begin = self.last_batch * self.batch_size
        if self.last_batch == self.num_batch-1:
            batch_end = len(self.dataset)
        else:
            batch_end = batch_begin + self.batch_size        

        for _idx in range(batch_begin, batch_end):
            indices.append(self.dataset[_idx]["indices"])
            input_ids.append(self.dataset[_idx]["input_ids"])
            image_tensors.append(self.dataset[_idx]["image_tensors"])
            image_sizes.append(self.dataset[_idx]["image_sizes"])

        self.last_batch+=1

        return {
            "indices" : torch.tensor(indices),
            "input_ids" : torch.tensor(input_ids),
            "image_tensors" : torch.tensor(image_tensors),
            "image_sizes" : torch.tensor(image_sizes)
        }

    def _get_augmented_batch(self):
        indices = list()
        input_ids = list()
        orig_image_tensors = list()
        image_sizes = list()
        aug_image_tensors = dict()


        batch_begin = self.last_batch * self.batch_size
        if self.last_batch == self.num_batch-1:
            batch_end = len(self.dataset)
        else:
            batch_end = batch_begin + self.batch_size        

        for _idx in range(batch_begin, batch_end):
            indices.append(self.dataset[_idx]["indices"])
            input_ids.append(self.dataset[_idx]["input_ids"])
            orig_image_tensors.append(self.dataset[_idx]["image_tensors"])
            image_sizes.append(self.dataset[_idx]["image_sizes"])

            for k, aug_imgs in self.dataset[_idx]["aug_img_tensors"].items():
                if k not in aug_image_tensors :
                    aug_image_tensors[k] = [[] for _ in range(len(aug_imgs))]
                for _aug_idx, _aug_img in enumerate(aug_imgs):
                    aug_image_tensors[k][_aug_idx].append(_aug_img)

        for k, aug_imgs in aug_image_tensors.items():
            for _aug_idx in range(len(aug_imgs)):
                aug_image_tensors[k][_aug_idx] = torch.tensor(aug_image_tensors[k][_aug_idx])

        self.last_batch+=1
        return {
            "indices" : torch.tensor(indices),
            "input_ids": torch.tensor(input_ids),
            "orig_image_tensors": torch.tensor(orig_image_tensors),
            "aug_image_tensors": aug_image_tensors
        }

def generate(model, tokenizer, dataset, cfg):
    num_gen_tokens = cfg.generation.num_gen_tokens
    batch_generator = BatchProcessor(dataset=dataset,
                                     batch_size=cfg.generation.batch_size,
                                     use_augmentation=cfg.generation.use_augmentation)
    outputs = list()
    for b_idx, batch in enumerate(batch_generator):
        _outputs = generate_a_batch(model, tokenizer, batch, cfg.generation.num_gen_tokens, cfg.generation.use_augmentation)
        outputs.extend(_outputs)

    return outputs