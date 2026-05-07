#!/usr/bin/env bash
# ============================================================
#  VASP single-step launcher: many POSCAR jobs on fixed GPUs
#  - Creates one job folder per POSCAR under ../jobs
#  - Uses fixed GPU list, e.g. GPUS=(2 4)
#  - Runs one job per GPU at a time
#  - When one finishes, automatically launches the next
#  - Does not rely on nvidia-smi / external GPU detection
# ============================================================
#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'
shopt -s nullglob

export LD_LIBRARY_PATH="<PATH_TO_DATA>/software/openblas/install/lib:$LD_LIBRARY_PATH"
# VASP_EXE="<PATH_TO_DATA>/software/VASP/vasp.6.4.2/bin/vasp_std"

unset NV_ACC_NOTIFY NVCOMPILER_ACC_NOTIFY ACC_NOTIFY PGI_ACC_NOTIFY

export NV_ACC_NOTIFY=0
export NVCOMPILER_ACC_NOTIFY=0
export ACC_NOTIFY=0
export PGI_ACC_NOTIFY=0

# ---------------- user settings ----------------
FRAME_ROOT="../poscars"          # where POSCAR-* files live
JOBS_ROOT="../jobs"              # where to create job folders
STATIC_DIR="$(pwd)"              # folder containing INCAR/POTCAR/KPOINTS
COPY_OR_LINK="copy"              # "copy" or "link"
NP="${NP:-1}"                    # MPI ranks per VASP job
VASP_EXE="${VASP_EXE:-vasp_std}" # can override: VASP_EXE=/path/to/vasp_std ./run_all.sh
GPUS=(0 1 4 6)                       # fixed GPU IDs, same logic as NEMD run.sh
# ------------------------------------------------

trap 'echo "🛑 Interrupted — aborting all running jobs."; jobs -pr | xargs -r kill; exit 130' INT TERM

mkdir -p "$JOBS_ROOT"

# Check static files
for f in INCAR POTCAR; do
  [[ -s "$STATIC_DIR/$f" ]] || { echo "❌ Missing $STATIC_DIR/$f"; exit 1; }
done
# KPOINTS optional

# Return 0 if this directory looks finished; 1 otherwise.
vasp_finished() {
  local d="$1"
  local markers=(
    "General timing and accounting informations for this job"
    "reached required accuracy - stopping structural energy minimisation"
    "reached required accuracy - stopping structural energy minimization"
  )

  if [[ -f "$d/OUTCAR" ]]; then
    for m in "${markers[@]}"; do
      if grep -aFqm1 "$m" "$d/OUTCAR"; then
        echo "   ↪ Detected finish marker in OUTCAR: \"$m\""
        return 0
      fi
    done
  fi

  if [[ -f "$d/vasp.out" ]]; then
    for m in "${markers[@]}"; do
      if grep -aFqm1 "$m" "$d/vasp.out"; then
        echo "   ↪ Detected finish marker in vasp.out: \"$m\""
        return 0
      fi
    done
  fi

  return 1
}

# ---------------- prepare / stage jobs ----------------
jobdirs=()
found=0

for pos in "$FRAME_ROOT"/POSCAR-*; do
  [[ -f "$pos" ]] || continue
  found=1

  frame_name="$(basename "$pos")"       # e.g. POSCAR-001
  job_dir="$JOBS_ROOT/$frame_name"
  mkdir -p "$job_dir"

  # POSCAR -> job_dir/POSCAR
  if [[ "$COPY_OR_LINK" == "link" ]]; then
    ln -sf "$(realpath -e "$pos")" "$job_dir/POSCAR"
  else
    cp -f "$pos" "$job_dir/POSCAR"
  fi

  cp -f "$STATIC_DIR/INCAR"  "$job_dir/"
  cp -f "$STATIC_DIR/POTCAR" "$job_dir/"
  [[ -s "$STATIC_DIR/KPOINTS" ]] && cp -f "$STATIC_DIR/KPOINTS" "$job_dir/"

  if vasp_finished "$job_dir"; then
    echo "✅ Already finished — skipping $job_dir"
    continue
  fi

  jobdirs+=( "$job_dir" )
  echo "📦 staged: $job_dir"
done

if [[ "$found" -eq 0 ]]; then
  echo "⚠️  No files matched: $FRAME_ROOT/POSCAR-*"
  exit 0
fi

total_jobs=${#jobdirs[@]}
if (( total_jobs == 0 )); then
  echo "✅ All matched jobs are already finished. Nothing to run."
  exit 0
fi

echo "Scheduling ${total_jobs} unfinished jobs:"
printf '  %s\n' "${jobdirs[@]}"

# ---------------- internal GPU scheduler ----------------
declare -A GPU_PIDS=()   # gpu -> pid
declare -A PID_GPU=()    # pid -> gpu
declare -A PID_JOB=()    # pid -> jobdir
next_idx=0

launch_job() {
  local gpu="$1"
  local jobdir="$2"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    cd "$jobdir"

    echo "GPU $gpu: starting $jobdir"
    echo "GPU $gpu" > gpu_assigned.log
    date      >> gpu_assigned.log
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> gpu_assigned.log
    echo "NP=$NP" >> gpu_assigned.log
    echo "which mpirun: $(which mpirun)" >> gpu_assigned.log
    echo "which vasp: $(which "$VASP_EXE" 2>/dev/null || echo "$VASP_EXE")" >> gpu_assigned.log
    mpirun -np "$NP" "$VASP_EXE" > vasp.out 2>&1
    rc=$?

    if (( rc == 0 )); then
      echo "GPU $gpu: $jobdir finished successfully"
    else
      echo "GPU $gpu: $jobdir failed with exit code $rc"
    fi

    exit "$rc"
  ) &

  local pid=$!
  GPU_PIDS["$gpu"]=$pid
  PID_GPU["$pid"]=$gpu
  PID_JOB["$pid"]=$jobdir

  echo "Launched $jobdir on GPU $gpu (PID $pid)"
}

echo "Launching ${total_jobs} jobs on GPUs: ${GPUS[*]}"
echo "(Ignoring external GPU usage; only using internal fixed GPU sequence)"

# Fill GPUs initially
for gpu in "${GPUS[@]}"; do
  (( next_idx >= total_jobs )) && break
  launch_job "$gpu" "${jobdirs[$next_idx]}"
  next_idx=$((next_idx + 1))
done

# As jobs finish, launch the next one
while (( next_idx < total_jobs )) || (( ${#GPU_PIDS[@]} > 0 )); do
  done_pid=""

  if wait -n -p done_pid 2>/dev/null; then
    :
  else
    wait -n || true
    done_pid=""
  fi

  if [[ -n "$done_pid" ]]; then
    gpu="${PID_GPU[$done_pid]:-}"
    jobdir="${PID_JOB[$done_pid]:-}"

    if [[ -n "$gpu" ]]; then
      rc=0
      wait "$done_pid" || rc=$?
      echo "GPU $gpu: $jobdir finished (PID $done_pid, exit $rc)"
      unset 'GPU_PIDS[$gpu]'
      unset 'PID_GPU[$done_pid]'
      unset 'PID_JOB[$done_pid]'
    fi
  else
    for gpu in "${!GPU_PIDS[@]}"; do
      pid="${GPU_PIDS[$gpu]}"
      if ! kill -0 "$pid" 2>/dev/null; then
        jobdir="${PID_JOB[$pid]:-}"
        rc=0
        wait "$pid" || rc=$?
        echo "GPU $gpu: $jobdir finished (PID $pid, exit $rc)"
        unset 'GPU_PIDS[$gpu]'
        unset 'PID_GPU[$pid]'
        unset 'PID_JOB[$pid]'
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

echo "✅ All jobs finished."