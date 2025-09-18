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
    test_run.test_run=true \
    img_metrics.metrics_to_use=["min_k_renyi_05_kl_div","min_k_renyi_divergence_025"] \
    img_metrics.parts=["img"] \
    path.output_dir=/home/clo37/priv/VLM-MIA-Study/results/TEST_MIA1 \
     img_metrics.get_proc_meta_values=1000 \
    img_metrics.get_token_labels=1000 \
    img_metrics.get_raw_images=5 \
    img_metrics.get_raw_meta_values=5 \
    data.augmentations.RandomResize.use=false \
    data.augmentations.RandomResize.size='[[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256],[256,256]]' \
    data.augmentations.RandomResize.scale='[[0.2,0.2],[0.4,0.4],[0.6,0.6],[0.8,0.8],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0]]' \
    data.augmentations.RandomResize.ratio='[[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[0.5,0.5],[0.75,0.75],[1.0,1.0],[1.25,1.25],[1.5,1.5]]' \
    data.augmentations.RandomRotation.use=false \
    data.augmentations.RandomRotation.degrees='[0.1,0.2,0.3,0.4,0.5,5,30,45,60,90]' \
    data.augmentations.GaussianNoise.use=true \
    data.augmentations.GaussianNoise.mean='[0.0,0.0,0.0,0.0,0.0,0.0]' \
    data.augmentations.GaussianNoise.std='[0.025,0.5,2.5,50.0,250.0,5000.0]' \
    data.augmentations.RandomAffine.use=false \
    data.augmentations.ColorJitter.use=false\

    






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