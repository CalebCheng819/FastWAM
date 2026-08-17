#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# The dedicated quota exposes eight GPUs as one indivisible resource shape.
# Rendering is deliberately single-process and sees only physical device 0.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
exec /bin/bash /oss-chengjuntao/artifacts/fastwam-p13-runtime-20260817-r7/run_p13_metric_cache_dlc.sh
