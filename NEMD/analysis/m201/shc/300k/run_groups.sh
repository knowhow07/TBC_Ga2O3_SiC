#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=3,6
GPUMD_EXE="<PATH_TO_DATA>/software/gpumd/GPUMD/src/gpumd"

TEMPLATE_MODEL="model.xyz"
TEMPLATE_POT="<PATH_TO_DATA>/qe_jobs/nep_training/config/0217/output/nep.txt"
TEMPLATE_RUNIN="run.in"

SHC_GROUPS=(5)

[[ -f "$TEMPLATE_MODEL" ]] || { echo "Missing $TEMPLATE_MODEL"; exit 1; }
[[ -f "$TEMPLATE_RUNIN" ]] || { echo "Missing $TEMPLATE_RUNIN"; exit 1; }
[[ -f "$TEMPLATE_POT"   ]] || { echo "Missing $TEMPLATE_POT"; exit 1; }
[[ -x "$GPUMD_EXE"      ]] || { echo "Missing/Non-exec $GPUMD_EXE"; exit 1; }

echo "SHC_GROUPS = ${SHC_GROUPS[*]}"

for g in "${SHC_GROUPS[@]}"; do
  dir="shc_g${g}"
  mkdir -p "$dir"

  cp -f "$TEMPLATE_MODEL" "$dir/model.xyz"
  cp -f "$TEMPLATE_POT"   "$dir/nep.txt"
  cp -f "$TEMPLATE_RUNIN" "$dir/run.in"

  sed -i \
    -e "s/^compute_shc.*/compute_shc 10 1000 0 4000 320 group 0 ${g}/" \
    "$dir/run.in"

  echo "=== Running GPUMD for group ${g} in ${dir} ==="
  if ! (
    cd "$dir"
    "$GPUMD_EXE" > log 2>&1
  ); then
    echo "ERROR: GPUMD failed for group ${g}. See ${dir}/log"
    exit 1
  fi

  echo "OK: group ${g} finished. Outputs in ${dir}/"
done

echo "All groups (3,4,5) finished."