#!/usr/bin/env bash
set -Eeuo pipefail

# ===================== User knobs (defaults) =====================
ENV_NAME="phoebe-nep"   # conda env with phonopy, ASE (+ optional pynep). NEP forces via LAMMPS "pair_style nep"
DRIVER_PY="dp_2nd.py"

# Input POSCAR (step 1 source)
POSCAR="POSCAR"

# LAMMPS
LMP_BIN="<PATH_TO_DATA>/software/lammps/lammps_dp/lammps-stable_22Jul2025_update1/src/lmp_mpi"
IN_LAMMPS="in.lammps"
LAMMPS_ATOM_STYLE="atomic"

# NEP + phonon  (keep variable names the same for compatibility)
export DP_NEP_BACKEND="pynep"
MODEL="<PATH_TO_DATA>/qe_jobs/nep_training/config/0217/output/nep.txt"   # <-- set this to your NEP model file (e.g. nep.txt / NEP_SiC.txt)

DIM="1 3 2"
AMP="0.02"
TYPE_MAP="Ga,O"

# Type id mapping for LAMMPS numeric types -> symbols (JSON string)
TYPE_ID_TO_ELEM='{"1":"Ga","2":"O"}'

# Band path for plotting
BAND_POINTS="0.5 0 0.5   0.5 0 0   0.5 0.5 0   0 0 0   0 0 0.5   0.5 0.5 0.5"
BAND_LABELS="L M A Γ Z V"

# Outputs (filenames)
DATA_FROM_POSCAR=""                     # will be "${POSCAR}.data"
POSCAR_PRE="POSCAR_pre.vasp"
POSCAR_RELAX="POSCAR_relax.vasp"
RELAX_DATA="relax.data"
RELAX_DUMP="relax.dump"

# Control
DO_RELAX=false

# ===================== CLI parsing =====================
usage() {
  cat <<EOF
Usage: $0 [options] [--relax]

Core:
  --env <conda_env>             (${ENV_NAME})
  --driver <dp_2nd.py>          (${DRIVER_PY})
  --poscar <POSCAR>             (${POSCAR})

LAMMPS:
  --lmp <path_to_lmp_mpi>       (${LMP_BIN})
  --in <in.lammps>              (${IN_LAMMPS})
  --atom-style <style>          (${LAMMPS_ATOM_STYLE})
  --relax                       run LAMMPS relaxation
  --relax-data <file>           (${RELAX_DATA})
  --relax-dump <file>           (optional; fallback if no data)

NEP + phonon (kept same flag names):
  --model <nep_model_file>      (${MODEL})
  --dim "na nb nc"              ("${DIM}")
  --amp <angstrom>              (${AMP})
  --type-map "Ga,O,..."         ("${TYPE_MAP}")
  --type-id-map <json>          ('${TYPE_ID_TO_ELEM}')

Band path:
  --band-points "<pts>"
  --band-labels "<labels>"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)            ENV_NAME="$2"; shift 2 ;;
    --driver)         DRIVER_PY="$2"; shift 2 ;;
    --poscar)         POSCAR="$2"; shift 2 ;;
    --lmp)            LMP_BIN="$2"; shift 2 ;;
    --in)             IN_LAMMPS="$2"; shift 2 ;;
    --atom-style)     LAMMPS_ATOM_STYLE="$2"; shift 2 ;;
    --relax)          DO_RELAX=true; shift 1 ;;
    --relax-data)     RELAX_DATA="$2"; shift 2 ;;
    --relax-dump)     RELAX_DUMP="$2"; shift 2 ;;
    --model)          MODEL="$2"; shift 2 ;;
    --dim)            DIM="$2 $3 $4"; shift 4 ;;
    --amp)            AMP="$2"; shift 2 ;;
    --type-map)       TYPE_MAP="$2"; shift 2 ;;
    --type-id-map)    TYPE_ID_TO_ELEM="$2"; shift 2 ;;
    --band-points)    BAND_POINTS="$2"; shift 2 ;;
    --band-labels)    BAND_LABELS="$2"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 1 ;;
  esac
done

# ===================== Env activation =====================
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH"; exit 2
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Ensure ASE exists (used for format conversion + NEP force driver writing data files)
python - <<'PY'
import sys
try:
    import ase  # noqa
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "ase"])
PY

# ===================== Helpers =====================
ase_write_poscar () {
  local in_fmt="$1" in_path="$2" out_poscar="$3" style="${4:-atomic}"
  python - "$in_fmt" "$in_path" "$out_poscar" "$style" <<'PY'
import sys
from ase.io import read, write
fmt, ipath, oposcar, style = sys.argv[1:]
if fmt == "poscar":
    atoms = read(ipath, format="vasp")
elif fmt == "lammps-data":
    atoms = read(ipath, format="lammps-data", style=style)
elif fmt == "lammps-dump":
    atoms = read(ipath, format="lammps-dump-text", index=-1)
else:
    raise SystemExit(f"Unknown fmt {fmt}")
write(oposcar, atoms, format="vasp")
print(f"[ASE] Wrote POSCAR: {oposcar}")
PY
}

ase_write_data_from_poscar () {
  local poscar="$1" out_data="$2" style="${3:-atomic}"
  python - "$poscar" "$out_data" "$style" <<'PY'
import sys
from ase.io import read, write
poscar, outdata, style = sys.argv[1:]
atoms = read(poscar, format="vasp")
write(outdata, atoms, format="lammps-data", atom_style=style)
print(f"[ASE] Wrote DATA: {outdata} (style={style})")
PY
}

# Run dp_2nd.py for one POSCAR and then plot bands (yaml/pdf kept separate)
run_fc2_and_plot () {
  local poscar="$1" tag="$2"

  echo "[run] FC2 for ${tag} using ${poscar}"

  rm -f FORCE_CONSTANTS FORCE_SETS band.yaml band.conf

  # ------------ pass env to python driver ------------
  export DP_INPUT_FORMAT="poscar"
  export DP_POSCAR_PATH="${poscar}"
  export DP_LAMMPS_DATA_PATH=""
  export DP_LAMMPS_DATA_STYLE="${LAMMPS_ATOM_STYLE}"
  export DP_LAMMPS_DUMP_PATH=""
  export DP_LAMMPS_DUMP_LAST_FRAME="True"
  export DP_TYPE_ID_TO_ELEM="${TYPE_ID_TO_ELEM}"

  export DP_MODEL_PATH="${MODEL}"              # NEP model file path (text)
  export DP_SUPERCELL_DIM="${DIM}"
  export DP_DISP_AMPLITUDE="${AMP}"
  export DP_TYPE_MAP_MASTER="${TYPE_MAP}"
  export DP_OUTPUT_MODE="force_constants"

  # NEW: tell python driver where LAMMPS is (for NEP force evaluation if needed)
  export DP_LMP_BIN="${LMP_BIN}"

  # ------------ run driver ------------
  python -u "${DRIVER_PY}"
  if [[ ! -s FORCE_CONSTANTS ]]; then
    echo "[ERROR] ${DRIVER_PY} didn't produce FORCE_CONSTANTS for ${tag}"
    exit 10
  fi

  mv -f FORCE_CONSTANTS "FORCE_CONSTANTS-${tag}"

  (
    set -e
    ln -sf "FORCE_CONSTANTS-${tag}" FORCE_CONSTANTS
    phonopy --readfc -c "${poscar}" --dim="${DIM}" \
      --band "${BAND_POINTS}" \
      --band-labels "${BAND_LABELS}" \
      --band-const-interval --band-points 101 -p
    mv -f band.yaml "band-${tag}.yaml"
    phonopy-bandplot -o "band-${tag}.pdf" "band-${tag}.yaml"
  )
  echo "[ok] Wrote: FORCE_CONSTANTS-${tag}, band-${tag}.yaml, band-${tag}.pdf"
}

# ===================== Step 1: POSCAR -> DATA =====================
DATA_FROM_POSCAR="${POSCAR}.data"
echo "[step 1] Convert ${POSCAR} -> ${DATA_FROM_POSCAR}"
ase_write_data_from_poscar "${POSCAR}" "${DATA_FROM_POSCAR}" "${LAMMPS_ATOM_STYLE}"

ase_write_poscar "poscar" "${POSCAR}" "${POSCAR_PRE}" "${LAMMPS_ATOM_STYLE}"

# ===================== Step 2: optional LAMMPS relax =====================
if ${DO_RELAX}; then
  echo "[step 2] Run LAMMPS relaxation with ${LMP_BIN}"
  if [[ ! -x "${LMP_BIN}" ]]; then
    echo "[ERROR] LAMMPS binary not found/executable: ${LMP_BIN}"; exit 3
  fi
  if [[ ! -f "${IN_LAMMPS}" ]]; then
    echo "[ERROR] LAMMPS input not found: ${IN_LAMMPS}"; exit 3
  fi

  export DATA_FROM_POSCAR
  export RELAX_DATA

  "${LMP_BIN}" -in "${IN_LAMMPS}"

  if [[ -f "${RELAX_DATA}" ]]; then
    echo "[step 2] Found relaxed data: ${RELAX_DATA}"
    ase_write_poscar "lammps-data" "${RELAX_DATA}" "${POSCAR_RELAX}" "${LAMMPS_ATOM_STYLE}"
  elif [[ -n "${RELAX_DUMP}" && -f "${RELAX_DUMP}" ]]; then
    echo "[step 2] Found relaxed dump: ${RELAX_DUMP}"
    ase_write_poscar "lammps-dump" "${RELAX_DUMP}" "${POSCAR_RELAX}" "${LAMMPS_ATOM_STYLE}"
  else
    echo "[WARN] No relaxed structure found (expected ${RELAX_DATA} or ${RELAX_DUMP})."
    echo "       Proceeding without a relaxed case."
    DO_RELAX=false
  fi
fi

# ===================== Step 3+4: FC2 + phonons =====================
run_fc2_and_plot "${POSCAR_PRE}" "pre"

if ${DO_RELAX}; then
  run_fc2_and_plot "${POSCAR_RELAX}" "relax"
fi

echo "[done] Outputs:"
echo "  - band-pre.pdf (and band-pre.yaml)"
${DO_RELAX} && echo "  - band-relax.pdf (and band-relax.yaml)"
