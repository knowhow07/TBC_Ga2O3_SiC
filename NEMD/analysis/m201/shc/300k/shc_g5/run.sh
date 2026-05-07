#!/usr/bin/env bash
# ============================================================
#  NEP Training Launcher (with output/ folder)
#  Automatically activates env and runs NEP training cleanly
# ============================================================

set -e
# set -Eeuo pipefail #this will lead to silent failure if nep exits with non-zero code

shopt -s expand_aliases

export CUDA_VISIBLE_DEVICES=3
GPUMD_EXE="<PATH_TO_DATA>/software/gpumd/GPUMD/src/gpumd"
TEMPLATE_MODEL="model.xyz"
TEMPLATE_POT="<PATH_TO_DATA>/qe_jobs/nep_training/config/0217/output/nep.txt"
TEMPLATE_RUNIN="run.in"

# ---------------------------------------

cp "$TEMPLATE_POT" .
# rm nvt.xyz relaxed.xyz thermo.out log neighbor.out
"$GPUMD_EXE" > log 2>&1

echo "GPUMD finished with exit code $?"