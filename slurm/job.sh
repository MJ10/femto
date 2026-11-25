#!/bin/bash
# One python script as a job (evals, RL, data preparation, ...), from the repo's directory, e.g.
#   sbatch --gres=gpu:1 slurm/job.sh agent_eval.py --run_name=femto
#   sbatch --gres=gpu:1 --time=6:00:00 slurm/job.sh rl.py --run_name=... --init_from=femto-mid
#   sbatch --cpus-per-task=2 --mem=16G slurm/job.sh prepare.py download shell
# (with cluv: cluv submit <cluster> slurm/job.sh <sbatch flags> -- <script> <args>). Set FEMTO_DIR with --export.
# On SIGTERM (time limit / preemption) rl.py saves a checkpoint after the current step; resubmitting resumes.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --signal=B:TERM@300
source slurm/common.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || true
exec python "$@"  # exec so SLURM's SIGTERM reaches python directly
