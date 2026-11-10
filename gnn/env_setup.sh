#!/usr/bin/env bash
# Sourced by every job script. Builds a venv on top of the cluster's PyTorch module the first time
# (needs internet), then reuses it. Tested with Python 3.14, torch 2.9.1+cu130, PyG 2.8.1, Triton 3.5.1.
# On another cluster, change the module line, or replace this file with your own environment.
module load pytorch/2.9.1 2>/dev/null || true
VENV=${VENV:-$PWD/venv}
if [ ! -x "$VENV/bin/python" ]; then
  echo "[env] creating venv $VENV"
  python -m venv --system-site-packages "$VENV"
  "$VENV/bin/pip" install --quiet torch_geometric pandas
fi
source "$VENV/bin/activate"
python -c "import torch, torch_geometric, sys; print('[env] python', sys.version.split()[0], '| torch', torch.__version__, '| pyg', torch_geometric.__version__, '| cuda', torch.cuda.is_available())"
