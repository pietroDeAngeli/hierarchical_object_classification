#!/bin/bash
#SBATCH -p edu-thesis
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:0
#SBATCH -J no-hierarchy
#SBATCH -o no_hierarchy_%j.log

set -e

source ~/.bashrc
conda activate hol

DESCRIPTOR=${DESCRIPTOR:-descriptor.json}
OBJ_MEM_ARGS=${OBJ_MEM_ARGS:-inputs/a95_b_1.json}
TEST_SIZE=${TEST_SIZE:-0.15}
SEED=${SEED:-0}
BATCH_SIZE=${BATCH_SIZE:-999999}

SUMMARY_OUT=${SUMMARY_OUT:-outputs/no_hierarchy_summary.json}
MODEL_OUT=${MODEL_OUT:-outputs/no_hierarchy_model.lz4}

mkdir -p outputs

PYTHONPATH=. python -m recsiam.no_hierarchy \
  --descriptor "$DESCRIPTOR" \
  --obj-mem-args "$OBJ_MEM_ARGS" \
  --test-size "$TEST_SIZE" \
  --seed "$SEED" \
  --batch-size "$BATCH_SIZE" \
  --eval-test \
  --output "$SUMMARY_OUT" \
  --model-output "$MODEL_OUT"
