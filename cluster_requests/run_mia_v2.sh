#!/bin/bash
#SBATCH --job-name=run_mia_v2
#SBATCH --output=out_run_mia_v2.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G


# Load environment
source ~/.bashrc
conda activate vlm_large_mia_llava_venv


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

python /home/clo37/priv/VLM-MIA-Study/mia.py \
    img_metrics.metrics_to_use=["min_k_renyi_05_kl_div","min_k_renyi_1_kl_div","min_k_renyi_2_kl_div","min_k_renyi_inf_kl_div","min_k_renyi_divergence_025","min_k_renyi_divergence_05","min_k_renyi_divergence_2","min_k_renyi_divergence_4"] \
    img_metrics.parts=["img"] \
    path.output_dir=/home/clo37/priv/VLM-MIA-Study/LatestResults/2025_08_26_13-14 \
    img_metrics.get_meta_values=1000 \
    img_metrics.get_token_labels=1000 \
    data.augmentations.RandomResize.use=false \
    data.augmentations.RandomResize.size='[[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256]]' \
    data.augmentations.RandomResize.scale='[[0.2,0.2],[0.4,0.4],[0.6,0.6],[0.8,0.8],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0]]' \
    data.augmentations.RandomResize.ratio='[[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[0.5,0.5],[0.75,0.75],[1.0,1.0],[1.25,1.25],[1.5,1.5]]' \
    data.augmentations.RandomRotation.use=true \
    data.augmentations.RandomRotation.degrees='[[0.1, 0.1],[0.2, 0.2],[0.3, 0.3],[0.4, 0.4],[0.5, 0.5],[0.6, 0.6],[0.7, 0.7],[0.8, 0.8],[0.9, 0.9],[1.0, 1.0]]' \
    data.augmentations.GaussianNoise.use=false \
    data.augmentations.GaussianNoise.mean='[0.0,0.0,0.0,0.0,0.0]' \
    data.augmentations.GaussianNoise.std='[1.0,2.0,3.0,4.0,5.0]' \
    data.augmentations.RandomAffine.use=false \