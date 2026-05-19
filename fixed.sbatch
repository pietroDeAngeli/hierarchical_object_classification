#!/bin/bash
#SBATCH -p edu-thesis
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:0
#SBATCH -J hol_fixed
#SBATCH -o faiss_rand/fixed_FAISS%j.log

source ~/.bashrc
conda activate hol

PYTHONPATH=. python -m recsiam.init_hierarchy \
  --descriptor descriptor.json \
  --obj-mem-args input_static/obj_mem_args_faiss.json \
  --fit-train \
  --eval-test \
  --test-size 0.15 \
  --test-output outputs/test_samples.json \
  --output outputs/init_summary.json \
  --evm-batch-size 999999 #\ 
  #--flat-hierarchy