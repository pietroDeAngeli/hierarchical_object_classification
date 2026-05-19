#!/bin/bash
#SBATCH -p edu-thesis
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -J logistic-head
#SBATCH -o logistic_head_%j.log

set -e

source ~/.bashrc
conda activate hol

DESCRIPTOR=${DESCRIPTOR:-descriptor.json}
TEST_SIZE=${TEST_SIZE:-0.10}
SEED=${SEED:-0}
C=${C:-1.0}
MAX_ITER=${MAX_ITER:-10000}
SOLVER=${SOLVER:-lbfgs}
N_JOBS=${N_JOBS:-4}

SUMMARY_OUT=${SUMMARY_OUT:-outputs/logistic_head_summary.json}
MODEL_OUT=${MODEL_OUT:-outputs/logistic_head_model.lz4}

mkdir -p outputs

PYTHONPATH=. python -m recsiam.logistic_head \
  --descriptor "$DESCRIPTOR" \
  --test-size "$TEST_SIZE" \
  --seed "$SEED" \
  --C "$C" \
  --max-iter "$MAX_ITER" \
  --solver "$SOLVER" \
  --n-jobs "$N_JOBS" \
  --eval-test \
  --output "$SUMMARY_OUT" \
  --model-output "$MODEL_OUT"
