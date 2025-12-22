#!/bin/bash
#SBATCH --job-name=picai_train
#SBATCH --partition=ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/train_fold_%j.out
#SBATCH --error=logs/train_fold_%j.err

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/11.7.0

source /scratch/brussel/114/vsc11434/venvs/picai/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

cd /scratch/brussel/114/vsc11434/project

python -m src.train data.fold=${FOLD}