#!/bin/bash
#SBATCH -p edu-thesis
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -J data_preprocessing
#SBATCH -o data_prep%j.log

source ~/.bashrc
conda activate hol

bash dataset_preprocessing.sh