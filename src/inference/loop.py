from tqdm import tqdm
from src.model import mod_infer_batch
from src.model.infer import BatchProcessor
from src.metrics import get_meta_metrics_by_part, get_img_metric_by_parts

def inference(model, tokenizer, dataset, cfg):
    """
    For each batch
        1. Conduct mod-infer
        2. Obtain meta-metrics
        3. Obtain img_metrics
        4. Combine img_metrics
    """

    batch_processor = BatchProcessor(dataset=dataset,
                                     batch_size=cfg.inference.batch_size,
                                     eos_token_id=tokenizer.eos_token_id,
                                     use_augmentation=cfg.inference.use_augmentation)
    parts = cfg.img_metrics.parts

    for b_idx, batch in enumerate(tqdm(batch_processor,
                                       total=len(batch_processor),
                                       desc="Running inference",
                                       unit="batch")):
        target_parts = mod_infer_batch(model, batch, tokenizer,
                                       parts=parts,
                                       use_augmentation=cfg.inference.use_augmentation)

        # Process separately for each part
        pred_all_parts = dict()
        for _part in parts:
            _meta_metrics = get_meta_metrics_by_part(target_parts,
                                                     part=_part,
                                                    cfg=cfg)
            _pred = get_img_metric_by_parts(_meta_metrics, cfg)
            pred_all_parts[_part] = _pred
    
        print("Initial batch done")
        exit()

def combine_batch_preds(pred):
    pass