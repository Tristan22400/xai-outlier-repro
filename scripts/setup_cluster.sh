#!/usr/bin/env bash
# One-shot setup for the gpu-telecom Slurm cluster.
#
#   ssh gpu-telecom
#   bash setup_cluster.sh
#
# Installs miniconda under $HOME, patches ~/.bashrc to auto-activate the
# xai-repro env, creates the env from env/environment.yml, and logs in to
# Weights & Biases. Idempotent: safe to re-run.

set -euo pipefail

CONDA_DIR="${HOME}/miniconda3"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "${CONDA_DIR}" ]]; then
  echo "[setup] installing miniconda to ${CONDA_DIR}"
  tmp_installer="$(mktemp --suffix=.sh)"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "${tmp_installer}"
  bash "${tmp_installer}" -b -p "${CONDA_DIR}"
  rm -f "${tmp_installer}"
else
  echo "[setup] miniconda already present at ${CONDA_DIR}"
fi

# Patch .bashrc once.
BASHRC_MARK="# >>> xai-repro conda bootstrap >>>"
if ! grep -Fq "${BASHRC_MARK}" "${HOME}/.bashrc"; then
  echo "[setup] patching ~/.bashrc"
  {
    echo ""
    echo "${BASHRC_MARK}"
    echo "source ${CONDA_DIR}/etc/profile.d/conda.sh"
    echo "conda activate xai-repro"
    echo "# <<< xai-repro conda bootstrap <<<"
  } >> "${HOME}/.bashrc"
fi

# shellcheck disable=SC1091
source "${CONDA_DIR}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "xai-repro"; then
  echo "[setup] creating conda env from ${REPO_DIR}/env/environment.yml"
  conda env create -f "${REPO_DIR}/env/environment.yml"
else
  echo "[setup] env xai-repro already exists; run 'conda env update -f env/environment.yml' to refresh"
fi

conda activate xai-repro

# Verify P100 sm_60 compatibility.
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    print(f"device: {name}  capability: sm_{cap[0]}{cap[1]}")
    assert cap == (6, 0), f"expected sm_60 (P100), got sm_{cap[0]}{cap[1]}"
else:
    print("WARNING: no CUDA device visible from login node (expected on gpu-telecom head)")
PY

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[setup] run 'wandb login' manually if you have not already"
else
  wandb login --relogin "${WANDB_API_KEY}"
fi

echo "[setup] done"
