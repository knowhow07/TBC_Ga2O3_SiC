#!/usr/bin/env bash
# ============================================================
#  NEP Training Launcher (with output/ folder)
#  Automatically activates env and runs NEP training cleanly
# ============================================================

set -u
# set -Eeuo pipefail #this will lead to silent failure if nep exits with non-zero code

shopt -s expand_aliases

# ---------- User configuration ----------
ENV_NAME="nep"                         # conda env with GPUMD/nep built
WORK_DIR="$(pwd)"                      # main run directory
DATA_DIR="${WORK_DIR}/data"            # contains train.xyz, val.xyz/test.xyz
OUTPUT_DIR="${WORK_DIR}/output"        # all run results will go here
NEP_EXE="<PATH_TO_DATA>/software/gpumd/GPUMD/src/nep"
LOG_FILE="nep_run.log"   
GENS_DEFAULT=10000
export CUDA_VISIBLE_DEVICES=4,5,6,7
# ---------------------------------------

# ========== Conda activation ==========
CONDA_BASE="<PATH_TO_DATA>/miniconda3"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ -f "${CONDA_SH}" ]]; then
    source "${CONDA_SH}"
else
    echo "❌ Conda initialization script not found at: ${CONDA_SH}"
    exit 1
fi

echo "Activating conda environment: ${ENV_NAME}"
conda activate "${ENV_NAME}" || { echo "❌ Failed to activate ${ENV_NAME}"; exit 1; }



echo "=================================================="
echo ">>> Environment info"
which python || true
# which nep || echo "⚠️ nep not found in PATH; using ${NEP_EXE}"
echo "=================================================="

# ========== Directory prep ==========
mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

# Verify required files
[[ -f "${WORK_DIR}/nep.in" ]] || { echo "❌ nep.in not found in ${WORK_DIR}"; exit 1; }
[[ -f "${DATA_DIR}/train.xyz" ]] || { echo "❌ train.xyz not found in ${DATA_DIR}"; exit 1; }

# Copy or symlink required input files
ln -sf "${WORK_DIR}/nep.in" .
ln -sf "${DATA_DIR}/train.xyz" .

if [[ -f "${DATA_DIR}/val.xyz" ]]; then
    echo "✅ Using val.xyz as validation set"
    ln -sf "${DATA_DIR}/val.xyz" test.xyz
elif [[ -f "${DATA_DIR}/test.xyz" ]]; then
    echo "✅ Using test.xyz as validation set"
    ln -sf "${DATA_DIR}/test.xyz" test.xyz
else
    echo "⚠️ No val.xyz or test.xyz found; training without validation set"
    rm -f test.xyz || true
fi

# Optional: clean old results
if [[ "${1:-}" == "fresh" ]]; then
    echo "🧹 Cleaning previous outputs..."
    rm -f nep.restart loss.out energy_* force_* virial_* model.* || true
fi

# Determine generations
GENS=$(awk '/generation/ {print $2}' "${WORK_DIR}/nep.in" | tail -n1 || echo "${GENS_DEFAULT}")

# ---------- Snapshot (archive) previous outputs but keep nep.restart ----------
SNAP_DIR="${OUTPUT_DIR}/snapshots/$(date +'%Y%m%d_%H%M%S')"
NEP_FILES=(
  nep.txt nep.restart loss.out
  energy_train.out energy_test.out
  force_train.out  force_test.out
  virial_train.out virial_test.out
  neighbor.out model.* plot_results.m thermo.out
)

snapshot_previous() {
  local any_found=0
  for f in "${NEP_FILES[@]}"; do
    [[ -e "$f" ]] && { any_found=1; break; }
  done
  if (( any_found )); then
    mkdir -p "$SNAP_DIR"
    echo "🗂  Snapshotting previous outputs to: $SNAP_DIR"
    # copy (not move) so nep.restart remains for resume
    for f in "${NEP_FILES[@]}"; do
      [[ -e "$f" ]] && cp -p "$f" "$SNAP_DIR"/
    done
  else
    echo "ℹ️  No previous outputs to snapshot."
  fi
}
# If nep.restart exists, we will resume; otherwise this is a fresh start unless user passed 'fresh'
if [[ -f nep.restart && "${1:-}" != "fresh" ]]; then
  echo "🔁 Resume mode detected (nep.restart present)."
  snapshot_previous
else
  if [[ "${1:-}" == "fresh" ]]; then
    echo "🧹 Cleaning previous outputs (fresh start)..."
    rm -f "${NEP_FILES[@]}" 2>/dev/null || true
  fi
fi

echo "=================================================="
echo ">>> Starting NEP training (${GENS} generations)"
echo "CWD        : $(pwd)"
echo "nep.in     : ${WORK_DIR}/nep.in"
echo "Logging to : ${LOG_FILE}"
echo "Output dir : ${OUTPUT_DIR}"
date
echo "=================================================="

# --- Run NEP but DON'T let non-zero exit kill the script ---
set +e   # temporarily disable 'exit on error'

# If you suspect stdbuf might be missing, start with the simpler line below:
#"${NEP_EXE}" 2>&1 | tee -a "${LOG_FILE}"

# Try with stdbuf (if available) for nicer streaming:
stdbuf -oL -eL "${NEP_EXE}" 2>&1 | tee -a "${LOG_FILE}"
rc=${PIPESTATUS[0]:-0}

set -e   # re-enable 'exit on error' for the rest of the script

echo "nep exit code: ${rc}"
echo "=================================================="
echo ">>> Finished training"
date
grep -E 'RMSE|Time used for training' "${LOG_FILE}" | tail -20 || true
echo "=================================================="

echo "📁 Output files stored in: ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}" | grep -E 'nep|loss|model' || true
