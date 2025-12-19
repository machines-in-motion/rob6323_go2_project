#!/bin/bash

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --account=rob_gy6323-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --output=../slurm_%j.out
#SBATCH --error=../slurm_%j.err

set -euo pipefail

# -------------------------------
# Project workspace bootstrap
# -------------------------------
PROJECT_NAME="rob6323_go2_project"
REMOTE_HOST="greene-dtn"
LOCAL_PROJECT="${HOME}/${PROJECT_NAME}"
REMOTE_PROJECT="${HOME}/${PROJECT_NAME}"
ISAACLAB_DIR="/scratch/$USER/IsaacLab"

# Ensure local project exists and mirrors remote (first time creates it)
mkdir -p "${LOCAL_PROJECT}"
rsync -az --delete --mkpath -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null" \
    "${REMOTE_HOST}:${REMOTE_PROJECT}/" \
    "${LOCAL_PROJECT}/"

mkdir -p "${ISAACLAB_DIR}"
rsync -az --delete --mkpath -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null" \
    "${REMOTE_HOST}:${ISAACLAB_DIR}/" \
    "${ISAACLAB_DIR}/"

# -------------------------------
# Hardcode your cluster paths
# -------------------------------
SIF_IMAGE="/scratch/$USER/isaac-lab-base.sif"
RUN_DIR="${LOCAL_PROJECT}"                       # run inside the mirrored project
PERSISTENT_CACHE_DIR="/scratch/$USER/docker-isaac-sim"
PERSISTENT_LOGS_DIR="/scratch/$USER/isaaclab/logs/${SLURM_JOB_ID}"

# Isaac Sim container paths
DOCKER_ISAACSIM_ROOT_PATH="/isaac-sim"
DOCKER_USER_HOME="/root"

# Node-local cache & execution
NODE_TMP="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CACHE_ROOT="${NODE_TMP}/docker-isaac-sim"
mkdir -p \
  "${CACHE_ROOT}/cache/kit" \
  "${CACHE_ROOT}/cache/ov" \
  "${CACHE_ROOT}/cache/pip" \
  "${CACHE_ROOT}/cache/glcache" \
  "${CACHE_ROOT}/cache/computecache" \
  "${CACHE_ROOT}/logs" \
  "${CACHE_ROOT}/data" \
  "${CACHE_ROOT}/documents" \
  "${CACHE_ROOT}/kit-data"

mkdir -p "${PERSISTENT_LOGS_DIR}"
touch "${PERSISTENT_LOGS_DIR}/.keep"

# Prefer node-local tmp for apptainer/singularity scratch
export APPTAINER_TMPDIR="${NODE_TMP}"
export APPTAINER_CACHEDIR="${NODE_TMP}/apptainer-cache"

# Forward all user args to eval.py
export ISAAC_ARGS="$*"
echo "$ISAAC_ARGS"

# Execute with GPU and required binds
singularity exec \
  --nv --containall \
  -B "${CACHE_ROOT}/kit-data:${DOCKER_ISAACSIM_ROOT_PATH}/kit/data:rw" \
  -B "${CACHE_ROOT}/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw" \
  -B "${CACHE_ROOT}/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw" \
  -B "${CACHE_ROOT}/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw" \
  -B "${CACHE_ROOT}/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw" \
  -B "${CACHE_ROOT}/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw" \
  -B "${CACHE_ROOT}/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw" \
  -B "${CACHE_ROOT}/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw" \
  -B "${CACHE_ROOT}/documents:${DOCKER_USER_HOME}/Documents:rw" \
  -B "${ISAACLAB_DIR}:/workspace/isaaclab:rw" \
  -B "${PERSISTENT_LOGS_DIR}:/workspace/isaaclab/logs:rw" \
  -B "${RUN_DIR}:/workspace/run:rw" \
  "${SIF_IMAGE}" bash -lc '
set -euo pipefail

cd /workspace/isaaclab
export ISAACLAB_PATH=/workspace/isaaclab

# Ensure the local package is installed in the container Python

/isaac-sim/python.sh -m pip install -e /workspace/run/source/rob6323_go2

/isaac-sim/python.sh /workspace/run/scripts/rsl_rl/train.py \
  --task=Template-Rob6323-Go2-Direct-v0 \
  --headless

# Identify the latest-created/modified subdirectory under go2_flat_direct
LATEST_DIR=$(ls -td /workspace/isaaclab/logs/rsl_rl/go2_flat_direct/*/ 2>/dev/null | head -n 1 || true)
if [[ -z "${LATEST_DIR:-}" ]]; then
  echo "No subdirectories found under /workspace/isaaclab/logs/rsl_rl/go2_flat_direct" >&2
  exit 1
fi
LATEST_DIR="${LATEST_DIR%/}"

# Run evaluation with the discovered checkpoint
/isaac-sim/python.sh /workspace/run/scripts/rsl_rl/play.py \
  --task=Template-Rob6323-Go2-Direct-v0 \
  --checkpoint "${LATEST_DIR}/model_499.pt" \
  --video \
  --video_length 1000 \
  --headless
'

rsync -az --delete \
  --exclude='*.err' \
  --exclude='*.out' \
  -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null" \
  "${PERSISTENT_LOGS_DIR}/" \
  "${REMOTE_HOST}:${REMOTE_PROJECT}/logs/partroughfri${SLURM_JOB_ID}/"
