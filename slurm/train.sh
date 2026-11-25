#!/bin/bash
# A train.py job (its args after the script), e.g.
#   sbatch --gres=gpu:1 --time=1:00:00 slurm/train.sh --run_name=smoke
#   sbatch --gres=gpu:4 --cpus-per-task=16 slurm/train.sh --run_name=x --num_tokens=2_500_000_000
# On SIGTERM (time limit / preemption) train.py saves a checkpoint and exits; resubmitting the same --run_name resumes.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@300
source slurm/common.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
exec python train.py "$@"  # exec so SLURM's SIGTERM reaches python directly
