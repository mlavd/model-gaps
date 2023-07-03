#!/bin/bash
#
#SBATCH --job-name=xg-t4
#SBATCH --output=../../logs/xgboost_task4.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=30-0:0:0

export SINGULARITYENV_BASE=/home/w174dxg/main
srun singularity exec \
    /home/w174dxg/main/singularity/xgboost.sif \
    bash /home/w174dxg/main/scripts/train/xgboost.sh task4
srun echo "Slurm Training Done"
