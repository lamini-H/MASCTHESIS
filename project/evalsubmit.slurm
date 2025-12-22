#!/bin/bash
#SBATCH --job-name=picai_eval
#SBATCH --partition=ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/eval_fold_%j.out
#SBATCH --error=logs/eval_fold_%j.err

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/11.7.0

source /scratch/brussel/114/vsc11434/venvs/picai/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

cd /scratch/brussel/114/vsc11434/project

if [ -z "$FOLD" ]; then
  echo "ERROR: FOLD is not set. Submit with: sbatch --export=FOLD=0 eval_fold.slurm"
  exit 1
fi

python -m src.eval_val data.fold=${FOLD}
