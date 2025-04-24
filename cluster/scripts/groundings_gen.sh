#!/bin/bash
#SBATCH --job-name=bu-groundings-gen
#SBATCH --mem=64G # Requested Memory
#SBATCH --partition=gpu # Partition
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00  # Job time limit
#SBATCH --constraint=vram48
#SBATCH -o ./cluster/logs/bu_groundings_gen/job-%j.out
#SBATCH -e ./cluster/logs/bu_groundings_gen/job-%j.err

module load conda/latest
conda activate /work/pi_wenlongzhao_umass_edu/16/vmuralikrish_umass_edu/.conda/envs/vllm_env

python -m src.query_set_generation.groundings_generator --model_index 6 --topic_index 1 --filename 10-K_NVDA_20240128 --no_of_qstns 5 
