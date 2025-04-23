#!/bin/bash
#SBATCH --job-name=bu-query-gen
#SBATCH --mem=64G # Requested Memory
#SBATCH --partition=gpu # Partition
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00  # Job time limit
#SBATCH --constraint=vram48
#SBATCH -o ../logs/bu_query_gen/job-%j.out
#SBATCH -e ../logs/bu_query_gen/job-%j.err

module load conda/latest
conda activate /work/pi_wenlongzhao_umass_edu/16/vmuralikrish_umass_edu/.conda/envs/vllm_env

python -m src.query_generator --model_index 6 --topic_index 3 --filename 10-K_NVDA_20240128 --no_of_trials 10
