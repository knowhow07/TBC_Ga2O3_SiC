#!/bin/bash

set -euo pipefail

# ============================================================
# User knobs
# ============================================================
ROOT_DIR="<PATH_TO_DATA>/qe_jobs/nep_training/ifc/100-106"
WORK_DIR="${ROOT_DIR}"

CONDA_SH="<PATH_TO_DATA>/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="phoebe"

STRUCTURE_FILE="CONTCAR"
JOBS_GLOB="jobs/POSCAR-*/vasprun.xml"

CROSS_IFC_PY="cross_ifc.py"
LOG_FILE="ifc_pipeline.log"
DISP_YAML="./poscars/phonopy_disp.yaml"
# ============================================================

echo "===== IFC post-processing pipeline started at $(date) =====" | tee "${WORK_DIR}/${LOG_FILE}"

# ---- activate conda env ----
if [ -f "${CONDA_SH}" ]; then
    source "${CONDA_SH}"
else
    echo "ERROR: conda.sh not found at ${CONDA_SH}" | tee -a "${WORK_DIR}/${LOG_FILE}"
    exit 1
fi

conda activate "${ENV_NAME}"

echo "Activated conda env: ${ENV_NAME}" | tee -a "${WORK_DIR}/${LOG_FILE}"

# ---- go to work dir ----
cd "${WORK_DIR}"

echo "Working directory: $(pwd)" | tee -a "${LOG_FILE}"

# ---- check required files ----
if [ ! -f "${STRUCTURE_FILE}" ]; then
    echo "ERROR: structure file not found: ${STRUCTURE_FILE}" | tee -a "${LOG_FILE}"
    exit 1
fi

if [ ! -f "${CROSS_IFC_PY}" ]; then
    echo "ERROR: python script not found: ${CROSS_IFC_PY}" | tee -a "${LOG_FILE}"
    exit 1
fi
if [ -f "./poscars/phonopy_disp.yaml" ] && [ ! -f "./phonopy_disp.yaml" ]; then
    cp ./poscars/phonopy_disp.yaml ./phonopy_disp.yaml
fi



# ---- collect forces into FORCE_SETS ----
echo "" | tee -a "${LOG_FILE}"
echo ">>> Step 1: Building FORCE_SETS from displaced VASP runs" | tee -a "${LOG_FILE}"
phonopy -f ${JOBS_GLOB} 2>&1 | tee -a "${LOG_FILE}"

if [ ! -f "FORCE_SETS" ]; then
    echo "ERROR: FORCE_SETS was not generated." | tee -a "${LOG_FILE}"
    exit 1
fi

# ---- write FORCE_CONSTANTS ----
echo "" | tee -a "${LOG_FILE}"
echo ">>> Step 2: Writing FORCE_CONSTANTS" | tee -a "${LOG_FILE}"
phonopy --writefc -c "${STRUCTURE_FILE}" --dim="1 1 1" 2>&1 | tee -a "${LOG_FILE}"

if [ ! -f "FORCE_CONSTANTS" ]; then
    echo "ERROR: FORCE_CONSTANTS was not generated." | tee -a "${LOG_FILE}"
    exit 1
fi

# ---- run cross-interface IFC analysis ----
echo "" | tee -a "${LOG_FILE}"
echo ">>> Step 3: Running cross_ifc.py" | tee -a "${LOG_FILE}"
python "${CROSS_IFC_PY}" 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "===== IFC post-processing pipeline finished at $(date) =====" | tee -a "${LOG_FILE}"