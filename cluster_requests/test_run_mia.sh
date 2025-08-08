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
    img_metrics.metrics_to_use=["min_k_renyi_05_kl_div","min_k_renyi_inf_kl_div","min_k_renyi_divergence_25","min_k_renyi_divergence_4"] \
    img_metrics.parts=["img"] \
    test_run.test_run=true \
    path.output_dir=/home/clo37/priv/VLM-MIA-Study/results/TEST_MIA \



