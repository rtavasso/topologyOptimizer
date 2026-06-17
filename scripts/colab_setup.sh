#!/usr/bin/env bash
# Colab setup for TAD. Run from the repository root. Does NOT reinstall torch —
# Colab ships a CUDA-enabled PyTorch and replacing it can break the GPU build.
set -euo pipefail

echo "== Installing Python dependencies (excluding torch) =="
pip install -q \
  "numpy>=1.26" "scipy>=1.11" "pandas>=2.0" "pyyaml>=6.0" \
  "zarr>=3.0" "h5py>=3.10" "safetensors>=0.4" "matplotlib>=3.8"

echo "== Installing TAD (editable) =="
pip install -q -e . --no-deps

echo "== Environment check =="
python - <<'PY'
import torch, tad
print("tad", tad.__version__)
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no CUDA GPU detected — set Runtime > Change runtime type > GPU")
PY
echo "== Setup complete =="
