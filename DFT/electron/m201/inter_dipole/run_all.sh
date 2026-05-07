export CUDA_VISIBLE_DEVICES=2,3
np=2


#!/bin/bash

LOCKFILE="job.lock"
# INPUT="Ga2O3_SiC_relax_2.in"
# OUTPUT="relax.out"

# --- Prevent re-submission if already running ---
if [ -e "$LOCKFILE" ]; then
    echo "❌ Job is already running (lock exists: $LOCKFILE). Exiting."
    exit 1
fi

# --- Create lock file ---
echo $$ > "$LOCKFILE"

# --- Run QE job ---
echo "🔄 Starting vasp job..."
mpirun -np ${np} vasp_std > output.out

# --- Check for normal termination ---
if grep -q "ERROR" "$OUTPUT"; then
    echo "✅ Job terminated with errors. Removing lock file."
    rm -f "$LOCKFILE"
else
    echo " Lock file retained for inspection."
fi

chgsum.pl AECCAR0 AECCAR2
bader CHGCAR -ref CHGCAR_sum
