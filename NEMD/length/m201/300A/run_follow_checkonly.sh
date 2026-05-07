#!/usr/bin/env bash
# ============================================================
#  GPUMD EMD launcher: N new jobs on M GPUs
#  - Does NOT overwrite existing job_* directories
#  - New jobs continue numbering from the last existing index
# ============================================================

set -u

# ------------ user settings ---------------------------------

N_JOBS=8
GPUS=(2 3)      # DO NOT auto-detect, DO NOT check nvidia-smi
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

declare -A GPU_PIDS=()
next_idx=0  # index into jobdirs[]

echo "Launching ${total_jobs} jobs on GPUs: ${GPUS[*]}"
echo "(Ignoring external GPU usage)"

while (( next_idx < total_jobs )) || (( ${#GPU_PIDS[@]} > 0 )); do
  for gpu in "${GPUS[@]}"; do

    # Check only the jobs started by THIS script
    if [[ -n "${GPU_PIDS[$gpu]:-}" ]]; then
      pid="${GPU_PIDS[$gpu]}"

      # If the PID is gone OR it's a zombie, treat as finished and reap it
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "GPU $gpu: internal job PID $pid finished."
        wait "$pid" 2>/dev/null || true
        unset GPU_PIDS["$gpu"]
      else
        # PID exists; it still might be a zombie -> reap it
        if ps -p "$pid" -o stat= 2>/dev/null | grep -q '^Z'; then
          echo "GPU $gpu: internal job PID $pid finished (zombie -> reaping)."
          wait "$pid" 2>/dev/null || true
          unset GPU_PIDS["$gpu"]
        else
          continue
        fi
      fi
    fi


    # If no more jobs left, skip
    if (( next_idx >= total_jobs )); then
      continue
    fi

    # Start new job on this GPU
    jobdir="${jobdirs[$next_idx]}"

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
    ((next_idx++))
  done

  sleep 3
done

echo "All jobs finished."
