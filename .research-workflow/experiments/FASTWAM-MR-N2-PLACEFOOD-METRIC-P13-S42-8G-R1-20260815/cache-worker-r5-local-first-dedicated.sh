#!/usr/bin/env bash
set -euo pipefail

# The quota exposes eight GPUs as one indivisible dedicated resource shape.
# Rendering is deliberately single-process and sees only physical device 0.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
exec /bin/bash /oss-chengjuntao/artifacts/fastwam-p13-runtime-20260815/cache-worker.sh
