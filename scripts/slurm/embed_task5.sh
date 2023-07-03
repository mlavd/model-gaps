#!/bin/bash
#
#SBATCH --job-name=eb-t5
#SBATCH --output=../../logs/embed_task5.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=30-0:0:0
#SBATCH --gres=gpu:1
#SBATCH -p p100

export SINGULARITYENV_BASE=/home/w174dxg/main
srun -p p100 --gres=gpu:1 singularity exec --nv \
    /home/w174dxg/main/singularity/linevul.sif \
    bash /home/w174dxg/main/scripts/create_embeddings.sh task5
srun echo "Slurm Training Done"

