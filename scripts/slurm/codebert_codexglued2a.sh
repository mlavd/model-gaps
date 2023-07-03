#!/bin/bash
#
#SBATCH --job-name=cb-c2
#SBATCH --output=../../logs/codebert_codexglued2a.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=30-0:0:0
#SBATCH --gres=gpu:1
#SBATCH -p a100

export SINGULARITYENV_BASE=/home/w174dxg/main
srun -p a100 --gres=gpu:1 singularity exec --nv \
    /home/w174dxg/main/singularity/linevul.sif \
    bash /home/w174dxg/main/scripts/train/codebert.sh codexglued2a 5
srun echo "Slurm Training Done"
