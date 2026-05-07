#!/usr/bin/env bash
# ============================================================
#  GPUMD EMD launcher: N new jobs on M GPUs
#  - Does NOT overwrite existing job_* directories
#  - New jobs continue numbering from the last existing index
# ============================================================

set -u

# ------------ user settings ---------------------------------

N_JOBS=4
GPUS=(0 1)      # DO NOT auto-detect, DO NOT check nvidia-smi
GPUMD_EXE="<PATH_TO_DATA>/software/gpumd/GPUMD/src/gpumd"

TEMPLATE_MODEL="model.xyz"
TEMPLATE_POT="<PATH_TO_DATA>/qe_jobs/nep_training/config/0217/output/nep.txt"
TEMPLATE_RUNIN="run.in"

# ------------ prepare NEW job directories -------------------

echo "Preparing ${N_JOBS} NEW job directories..."

shopt -s nullglob
existing=(job_*)
max_idx=0

# Find max existing index: job_01, job_02, ...
for d in "${existing[@]}"; do
  # strip "job_" prefix
  num=${d#job_}
  # ensure numeric
  if [[ "$num" =~ ^[0-9]+$ ]]; then
    idx=$((10#$num))       # force base-10 to avoid octal
    (( idx > max_idx )) && max_idx=$idx
  fi
done

start_idx=$((max_idx + 1))
end_idx=$((start_idx + N_JOBS - 1))

echo "Existing max job index: $max_idx"
echo "Creating jobs: job_$(printf '%02d' "$start_idx") .. job_$(printf '%02d' "$end_idx")"

for ((idx=start_idx; idx<=end_idx; idx++)); do
  jobdir=$(printf "job_%02d" "$idx")
  mkdir -p "$jobdir"

  cp "$TEMPLATE_MODEL" "$jobdir/"
  cp "$TEMPLATE_POT"   "$jobdir/"
  cp "$TEMPLATE_RUNIN" "$jobdir/"

  # IMPORTANT: do NOT rm -f anything here; existing dirs are untouched
done

echo "Job directory preparation done."

# ------------ build job list for scheduler (ONLY new jobs) ------------------

# Only schedule the newly created job directories: job_start .. job_end
jobdirs=()
for ((idx=start_idx; idx<=end_idx; idx++)); do
  jobdirs+=( "$(printf 'job_%02d' "$idx")" )
done

total_jobs=${#jobdirs[@]}

if (( total_jobs == 0 )); then
  echo "No new job directories to run. Nothing to do."
  exit 0
fi

echo "Scheduling ${total_jobs} NEW job directories:"
printf '  %s\n' "${jobdirs[@]}"


# ------------ internal GPU scheduler only -------------------

declare -A GPU_PIDS=()   # gpu -> pid
declare -A PID_GPU=()    # pid -> gpu
next_idx=0

echo "Launching ${total_jobs} jobs on GPUs: ${GPUS[*]}"
echo "(Ignoring external GPU usage)"

launch_job() {
  local gpu="$1"
  local jobdir="$2"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    cd "$jobdir"
    echo "GPU $gpu: starting $jobdir"
    "$GPUMD_EXE" > log 2>&1
    echo "GPU $gpu: $jobdir finished"
  ) &

  local pid=$!
  GPU_PIDS["$gpu"]=$pid
  PID_GPU["$pid"]=$gpu
  echo "Launched $jobdir on GPU $gpu (PID $pid)"
}

# Fill GPUs initially
for gpu in "${GPUS[@]}"; do
  (( next_idx >= total_jobs )) && break
  launch_job "$gpu" "${jobdirs[$next_idx]}"
  ((next_idx++))
done

# As jobs finish, launch next
while (( next_idx < total_jobs )) || (( ${#GPU_PIDS[@]} > 0 )); do
  done_pid=""

  # Bash 5+: wait -n -p returns the PID that finished (best)
  if wait -n -p done_pid 2>/dev/null; then
    :
  else
    # Fallback: plain wait -n (no PID), then we’ll rescan to find finished ones
    wait -n || true
    done_pid=""
  fi

  if [[ -n "$done_pid" ]]; then
    gpu="${PID_GPU[$done_pid]:-}"
    if [[ -n "$gpu" ]]; then
      echo "GPU $gpu: internal job PID $done_pid finished."
      unset 'GPU_PIDS[$gpu]'
      unset 'PID_GPU[$done_pid]'
    fi
  else
    # Fallback cleanup: remove any PIDs that are no longer children
    for gpu in "${!GPU_PIDS[@]}"; do
      pid="${GPU_PIDS[$gpu]}"
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "GPU $gpu: internal job PID $pid finished."
        wait "$pid" 2>/dev/null || true
        unset 'GPU_PIDS[$gpu]'
        unset 'PID_GPU[$pid]'
      fi
    done
  fi

  # Launch new jobs into any free GPU slots
  for gpu in "${GPUS[@]}"; do
    if [[ -z "${GPU_PIDS[$gpu]:-}" ]] && (( next_idx < total_jobs )); then
      launch_job "$gpu" "${jobdirs[$next_idx]}"
      ((next_idx++))
    fi
  done
done

echo "All jobs finished."