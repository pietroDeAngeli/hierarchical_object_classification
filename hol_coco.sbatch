#!/bin/bash
#SBATCH -p edu-thesis
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:0
#SBATCH -J hol-coco
#SBATCH -o hol_coco_%j.log

source ~/.bashrc
conda activate hol

bash runexp.sh