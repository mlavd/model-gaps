#!/bin/bash
#
#SBATCH --job-name=re-t6
#SBATCH --output=../../logs/regvd_task6.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=30-0:0:0
#SBATCH --gres=gpu:1
#SBATCH -p a100

export SINGULARITYENV_BASE=/home/w174dxg/main
srun -p a100 --gres=gpu:1 singularity exec --nv \
    /home/w174dxg/main/singularity/regvd.sif \
    bash /home/w174dxg/main/scripts/train/regvd.sh task6 1
srun echo "Slurm Training Done"
