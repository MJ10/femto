# Sourced by the job scripts. Runs the code at a pinned commit from node-local storage, so a job still waiting
# in the queue is not affected when the project checkout on the cluster moves to a newer commit.
# The commit is $GIT_COMMIT (`sbatch --export=ALL,GIT_COMMIT=<sha>`, or set by `cluv submit`),
# otherwise the current HEAD of the project dir. Branch commits must be pushed to GitHub first.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=$SCRATCH/hf_cache JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cache WANDB_DIR=${FEMTO_DIR:-$SCRATCH/femto}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$SCRATCH/uv_cache}
export PYTHONUNBUFFERED=1  # otherwise the log only shows up when the job ends

project=$PWD
commit=${GIT_COMMIT:-$(git -C "$project" rev-parse HEAD)}
git -C "$project" fetch -q origin || echo "git fetch failed (no internet on this node?), using local commits"
git clone -q --shared "$project" "$SLURM_TMPDIR/code"
cd "$SLURM_TMPDIR/code"
git checkout -q "$commit"
uv sync --frozen -q || { sleep $((RANDOM % 60)); uv sync --frozen -q; }  # retry: jobs starting together can race on the shared uv cache
source .venv/bin/activate
echo "running commit $(git rev-parse --short HEAD) on $(hostname)"
