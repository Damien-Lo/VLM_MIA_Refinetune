#!/bin/bash
#SBATCH --job-name=test_run_v2
#SBATCH --output=out_test_run_v2.log
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G


# Load environment
source ~/.bashrc
conda activate vlm_large_mia_llava_venv


export PYTHONPATH=$PYTHONPATH:/local/scratch/clo37/vlm_large_mia/
python /home/clo37/priv/VLM-MIA-Study/mia.py \
    test_run.test_run=true \
    img_metrics.metrics_to_use=["min_k_renyi_05_kl_div","min_k_renyi_divergence_025"] \
    img_metrics.parts=["img"] \
    path.output_dir=/home/clo37/priv/VLM-MIA-Study/results/TEST_MIA2 \
    img_metrics.get_meta_values=15 \
    img_metrics.get_token_labels=15 \
    data.augmentations.RandomResize.size='[[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256]]' \
    data.augmentations.RandomResize.scale='[[0.2,0.2],[0.4,0.4],[0.6,0.6],[0.8,0.8],[1.0,1.0],[0.75,1.33333333],[0.75,1.33333333],[0.75,1.33333333],[0.75,1.33333333],[0.75,1.33333333]]' \
    data.augmentations.RandomResize.ratio='[[0.08,1.0],[0.08,1.0],[0.08,1.0],[0.08,1.0],[0.08,1.0],[0.5,0.5],[0.75,0.75],[1.0,1.0],[1.25,1.25],[1.5,1.5]]' \
    data.augmentations.RandomRotation.degrees='[5,30,45,60,90]' \
    data.augmentations.GaussianNoise.mean='[0.0,0.0,0.0,0.0,0.0]' \
    data.augmentations.GaussianNoise.std='[1.0,2.0,3.0,4.0,5.0]' \
    




# python /home/clo37/priv/VLM-MIA-Study/mia.py \
#     img_metrics.metrics_to_use=["min_k_renyi_05_kl_div","min_k_renyi_divergence_025"] \
#     img_metrics.parts=["img"] \
#     test_run.test_run=true \
#     path.output_dir=/home/clo37/priv/VLM-MIA-Study/results/TEST_MIA \
#     img_metrics.get_meta_values=20 \
#     img_metrics.get_token_labels=20 \
#     data.augmentations.RandomResize.use=true \
#     data.augmentations.RandomRotation.use=false \
#     data.augmentations.RandomAffine.use=false \
#     data.augmentations.GaussianNoise.use=true


