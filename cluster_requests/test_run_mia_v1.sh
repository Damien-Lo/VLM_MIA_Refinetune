#!/bin/bash
#SBATCH --job-name=test_run_v1
#SBATCH --output=out_test_run_v1.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G


# Load environment
source ~/.bashrc
conda activate vlm_large_mia_llava_venv


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/
python /home/clo37/priv/VLM-MIA-Study/mia.py \
    img_metrics.metrics_to_use=["min_k_renyi_divergence_025"] \
    img_metrics.parts=["img"] \
    path.output_dir=/home/clo37/priv/VLM-MIA-Study/results/TEST_MIA1 \
    img_metrics.get_meta_values=15 \
    img_metrics.get_token_labels=15 \
    test_run.test_run=true \






# python /home/clo37/priv/VLM-MIA-Study/mia.py \
#     img_metrics.metrics_to_use=["min_k_renyi_05_kl_div","min_k_renyi_1_kl_div","min_k_renyi_2_kl_div","min_k_renyi_inf_kl_div","min_k_renyi_divergence_25","min_k_renyi_divergence_05","min_k_renyi_divergence_2","min_k_renyi_divergence_4"] \
#     img_metrics.parts=["img"] \
#     test_run.test_run=true \
#     path.output_dir=/home/clo37/priv/VLM-MIA-Study/results/TEST_MIA \
#     img_metrics.get_meta_values=20 \
#     img_metrics.get_token_labels=20 \
#     data.augmentations.RandomResize.use=true \
#     data.augmentations.RandomRotation.use=false \
#     data.augmentations.RandomAffine.use=false \
#     data.augmentations.GaussianNoise.use=true