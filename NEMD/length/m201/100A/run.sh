#!/usr/bin/env bash
# ============================================================
#  NEP Training Launcher (with output/ folder)
#  Automatically activates env and runs NEP training cleanly
# ============================================================

set -u
# set -Eeuo pipefail #this will lead to silent failure if nep exits with non-zero code

shopt -s expand_aliases

export CUDA_VISIBLE_DEVICES=0
# ---------------------------------------

rm compute.out thermo.out
# rm nvt.xyz relaxed.xyz thermo.out log neighbor.out
<PATH_TO_DATA>/software/gpumd/GPUMD/src/gpumd > log 2>&1

echo "GPUMD finished with exit code $?"