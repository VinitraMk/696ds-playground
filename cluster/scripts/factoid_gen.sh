#!/bin/bash
#SBATCH --job-name=bu-factoid-gen
#SBATCH --mem=64G # Requested Memory
#SBATCH --partition=gpu # Partition
#SBATCH --gres=gpu:1
#SBATCH -t 8:00:00  # Job time limit
#SBATCH --constraint=vram48
#SBATCH -o ../logs/bu_factoid_gen/job-%j.out
#SBATCH -e ../logs/bu_factoid_gen/job-%j.err

module load conda/latest
conda activate /work/pi_wenlongzhao_umass_edu/16/vmuralikrish_umass_edu/.conda/envs/vllm_env

python -m src.factoid_generator --model_index 6 --topic_index 0 --filename 10-K_INTC_20231230
