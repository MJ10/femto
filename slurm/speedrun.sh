#!/bin/bash
# run.sh (the whole recipe) as one job on a 4-GPU node. It resumes where it stopped, so after a time limit or a
# preemption, resubmit it (--requeue does that for preemptions). E.g.
#   sbatch --gres=gpu:4 --cpus-per-task=16 --mem=160G slurm/speedrun.sh
#   sbatch --gres=gpu:1 --time=3:00:00 slurm/speedrun.sh smoke
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --requeue
source slurm/common.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
exec bash run.sh "$@"
