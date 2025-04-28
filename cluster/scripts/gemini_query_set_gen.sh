#!/bin/bash
#SBATCH --job-name=bu-gemini-query-gen
#SBATCH --mem=32G # Requested Memory
#SBATCH --partition=cpu # Partition
#SBATCH -t 2:00:00  # Job time limit
#SBATCH -o ./cluster/logs/bu_gemini_query_gen/job-%j.out
#SBATCH -e ./cluster/logs/bu_gemini_query_gen/job-%j.err

module load conda/latest
conda activate /work/pi_wenlongzhao_umass_edu/16/vmuralikrish_umass_edu/.conda/envs/vllm_env

python -m src.gemini_query_generator --topic_index 3 --filename 10-K_GM_20231231

