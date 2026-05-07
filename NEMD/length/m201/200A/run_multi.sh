#!/usr/bin/env bash
# ============================================================
#  GPUMD EMD launcher: N independent jobs on M GPUs
#  This version completely ignores existing GPU workloads.
# ============================================================

set -euo pipefail

# ------------ user settings ---------------------------------

N_JOBS=8
GPUS=(5 6)      # DO NOT auto-detect, DO NOT check nvidia-smi
GPUMD_EXE="<PATH_TO_DATA>/software/gpumd/GPUMD/src/gpumd"

TEMPLATE_MODEL="model.xyz"
TEMPLATE_POT="<PATH_TO_DATA>/qe_jobs/nep_training/config/1202_all/output/nep.txt"
TEMPLATE_RUNIN="run.in"

# ------------ prepare job directories -----------------------

echo "Preparing ${N_JOBS} job directories..."

for ((i=1; i<=N_JOBS; i++)); do
    jobdir=$(printf "job_%02d" "$i")
    mkdir -p "$jobdir"

    cp "$TEMPLATE_MODEL" "$jobdir/"
    cp "$TEMPLATE_POT"   "$jobdir/"
    cp "$TEMPLATE_RUNIN" "$jobdir/"

    rm -f "$jobdir"/{log,thermo.out,neighbor.out,nvt.xyz,relaxed.xyz,computed.out}
done

echo "Job directories ready."

# ------------ internal GPU scheduler only -------------------

declare -A GPU_PIDS=()
next_job=1
total_jobs=$N_JOBS

echo "Launching ${total_jobs} jobs on GPUs: ${GPUS[*]}"
echo "(Ignoring external GPU usage)"

while (( next_job <= total_jobs )) || (( ${#GPU_PIDS[@]} > 0 )); do
  for gpu in "${GPUS[@]}"; do

   # Check only the jobs started by THIS script
    if [[ -n "${GPU_PIDS[$gpu]:-}" ]]; then
      pid="${GPU_PIDS[$gpu]}"

      # If process is gone, reap it and free the GPU slot exactly once
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        echo "GPU $gpu: internal job PID $pid finished."
        unset "GPU_PIDS[$gpu]"
      else
        continue
      fi
    fi

    # If no more jobs left, skip
    if (( next_job > total_jobs )); then
      continue
    fi

    # Start new job on this GPU
    jobdir=$(printf "job_%02d" "$next_job")

    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      cd "$jobdir"
      echo "GPU $gpu: starting $jobdir"
      "$GPUMD_EXE" > log 2>&1
      echo "GPU $gpu: $jobdir finished"
    ) &

    pid=$!
    GPU_PIDS["$gpu"]=$pid

    echo "Launched $jobdir on GPU $gpu (PID $pid)"
    ((next_job++))
  done

  sleep 3
done

echo "All jobs finished."
