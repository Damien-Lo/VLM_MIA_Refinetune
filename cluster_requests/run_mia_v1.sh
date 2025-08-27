#!/bin/bash
#SBATCH --job-name=run_mia_v1
#SBATCH --output=out_run_mia_v1.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G


# Load environment
source ~/.bashrc
conda activate vlm_large_mia_llava_venv


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

python /home/clo37/priv/VLM-MIA-Study/mia.py \
    img_metrics.metrics_to_use=["aug_kl"] \
    img_metrics.parts=["img"] \
    path.output_dir=/home/clo37/priv/VLM-MIA-Study/LatestResults/2025_08_27_17-04 \
    img_metrics.get_meta_values=0 \
    img_metrics.get_token_labels=0 \
    data.augmentations.RandomResize.use=true \
    data.augmentations.RandomRotation.use=true \
    data.augmentations.ColorJitter.use=true \
    data.augmentations.RandomAffine.use=true \
    data.augmentations.GaussianNoise.use=false \