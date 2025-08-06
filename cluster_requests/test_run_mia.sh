#!/bin/bash
#SBATCH --job-name=test_run
#SBATCH --output=out_test_run.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G


# Load environment
source ~/.bashrc
conda activate vlm_large_mia_llava_venv


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/

python /home/clo37/priv/VLM-MIA-Study/mia.py \
    img_metrics.metrics_to_use=["aug_kld","cross_entropy_mink"] \
    img_metrics.parts=["img"] \
    test_run.test_run=true \