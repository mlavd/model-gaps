#!/bin/bash
#
#SBATCH --job-name=xg-t6
#SBATCH --output=../../logs/xgboost_task6.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=30-0:0:0

export SINGULARITYENV_BASE=/home/w174dxg/main
srun singularity exec \
    /home/w174dxg/main/singularity/xgboost.sif \
    bash /home/w174dxg/main/scripts/train/xgboost.sh task6
srun echo "Slurm Training Done"
