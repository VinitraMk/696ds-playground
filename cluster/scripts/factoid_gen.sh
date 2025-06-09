#!/bin/bash
#SBATCH --job-name=bu-factoid-gen
#SBATCH --mem=64G # Requested Memory
#SBATCH --partition=gpu # Partition
#SBATCH --gres=gpu:1
#SBATCH -t 4:00:00  # Job time limit
#SBATCH --constraint=vram48
#SBATCH -o ./cluster/logs/bu_factoid_gen/job-%j.out
#SBATCH -e ./cluster/logs/bu_factoid_gen/job-%j.err

module load conda/latest
conda activate /work/pi_wenlongzhao_umass_edu/16/vmuralikrish_umass_edu/.conda/envs/vllm_env

python -m src.factoid_generator --model_index 6 --topic_index -1 --filename 10-K_TSLA_20231231
