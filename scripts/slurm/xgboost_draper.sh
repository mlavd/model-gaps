#!/bin/bash
#
#SBATCH --job-name=xg-dr
#SBATCH --output=../../logs/xgboost_draper.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=30-0:0:0

export SINGULARITYENV_BASE=/home/w174dxg/main
srun singularity exec \
    /home/w174dxg/main/singularity/xgboost.sif \
    bash /home/w174dxg/main/scripts/train/xgboost.sh draper
srun echo "Slurm Training Done"
