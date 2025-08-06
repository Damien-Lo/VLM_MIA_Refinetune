#!/bin/bash
#SBATCH --job-name=run_mia_v1
#SBATCH --output=run_mia_v1.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G


# Load environment
source ~/.bashrc
conda activate vlm_large_mia_llava_venv


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

python /home/clo37/priv/VLM-MIA-Study/mia.py \
    img_metrics.metrics_to_use=["aug_kld","mink","max_k_renyi_1_entro","max_k_renyi_2_entro","max_k_renyi_05_entro","max_prob_gap","min_k_renyi_1_entro","min_k_renyi_2_entro","min_k_renyi_05_entro","mod_renyi_1_entro","mod_renyi_2_entro","mod_renyi_05_entro","cross_entropy_mink"] \
    img_metrics.parts=["img"] \