#!/bin/bash
#
#SBATCH --job-name=tc-t3
#SBATCH --output=../../logs/textcnn_task3.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=30-0:0:0
#SBATCH --gres=gpu:1
#SBATCH -p a100

export SINGULARITYENV_BASE=/home/w174dxg/main
srun -p a100 --gres=gpu:1 singularity exec --nv \
    /home/w174dxg/main/singularity/lightning.sif \
    bash /home/w174dxg/main/scripts/train/textcnn.sh task3
srun echo "Slurm Training Done"
