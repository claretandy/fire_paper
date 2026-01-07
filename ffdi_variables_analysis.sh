#!/bin/bash -l
#SBATCH --qos=high  # normal # low # high
#SBATCH --mem=50000
#SBATCH --ntasks=8
#SBATCH --partition=cpu-long
#SBATCH --output=/data/scratch/andrew.hartley/fire/ffdi_fire_paper_%j_%N.out
#SBATCH --time=3600  # 4320  #4320 #360

conda activate impacts_toolbox

python ffdi_variables_analysis.py
