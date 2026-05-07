#!/usr/bin/env bash
set -u

# --------- user knobs ----------
NEP_EXE="<PATH_TO_DATA>/software/gpumd/GPUMD/src/nep"           # nep executable
MODEL=${MODEL:-output/nep.txt}              # trained NEP model
TEST_DIR=${TEST_DIR:-data/tests}     # folder with *.xyz test sets
OUT_ROOT=${OUT_ROOT:-tests}          # where per-set prediction folders live
BASE_NEP_IN=${BASE_NEP_IN:-nep.in}   # your original training nep.in in the ROOT dir
PYTHON=${PYTHON:-python3}            # python interpreter
PLOT_SCRIPT=${PLOT_SCRIPT:-plot_test_nep.py}
# -----------------------------------

mkdir -p "${OUT_ROOT}"
SUMMARY="${OUT_ROOT}/rmse_summary.csv"
echo "dataset,E_RMSE,F_RMSE,V_RMSE" > "${SUMMARY}"

for xyz in "${TEST_DIR}"/*.xyz; do
    [[ -e "$xyz" ]] || continue

    base=$(basename "$xyz" .xyz)
    outdir="${OUT_ROOT}/${base}"
    mkdir -p "${outdir}"

    echo "=== Testing ${base} ==="

    cd "${outdir}"

    # link model and xyz (as train.xyz for prediction mode)
    ln -sf "../../${MODEL}" nep.txt
    ln -sf "../../${xyz}"  train.xyz

    # start from original training nep.in and force prediction 1
    cp "../../${BASE_NEP_IN}" nep.in
    awk '
        BEGIN{done=0}
        /^prediction[[:space:]]+/ {
            print "prediction 1"
            done=1
            next
        }
        {print}
        END{
            if(!done) print "prediction 1"
        }
    ' nep.in > nep.tmp && mv nep.tmp nep.in

    # run prediction
    "${NEP_EXE}" < nep.in > log 2>&1 || {
        echo "  !! nep crashed for ${base} (see ${outdir}/log)" >&2
        cd - > /dev/null
        continue
    }

    # compute RMSEs via Python from *train.out files
    rmse_line=$(${PYTHON} - << 'PY'
import os, math
import numpy as np

def rmse_two_col(fname):
    if not os.path.exists(fname):
        return float('nan')
    a = np.loadtxt(fname)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    diff = a[:, 0] - a[:, 1]   # col0 = pred, col1 = ref (or vice versa, symmetric)
    return math.sqrt((diff**2).mean())

def rmse_force(fname):
    if not os.path.exists(fname):
        return float('nan')
    a = np.loadtxt(fname)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    # Now assume: Fx_pred Fy_pred Fz_pred Fx_ref Fy_ref Fz_ref
    if a.shape[1] != 6:
        raise SystemExit(f"Unexpected force_train.out shape: {a.shape}")
    Fx_pred, Fy_pred, Fz_pred, Fx_ref, Fy_ref, Fz_ref = a.T
    diff = np.concatenate([
        Fx_pred - Fx_ref,
        Fy_pred - Fy_ref,
        Fz_pred - Fz_ref,
    ])
    return math.sqrt((diff**2).mean())

def rmse_virial(fname):
    if not os.path.exists(fname):
        return float('nan')
    a = np.loadtxt(fname)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    # Now assume: [all pred comps][all ref comps]
    if a.shape[1] % 2 != 0:
        raise SystemExit(f"Unexpected virial/stress shape: {a.shape}")
    ncomp = a.shape[1] // 2
    pred = a[:, :ncomp]
    ref  = a[:, ncomp:]
    diff = (pred - ref).ravel()
    return math.sqrt((diff**2).mean())

E = rmse_two_col("energy_train.out")
F = rmse_force("force_train.out")
Vfile = "virial_train.out" if os.path.exists("virial_train.out") else "stress_train.out"
V = rmse_virial(Vfile) if os.path.exists(Vfile) else float('nan')

print(f"{E},{F},{V}")

PY
)

    E_RMSE=$(echo "$rmse_line" | cut -d',' -f1)
    F_RMSE=$(echo "$rmse_line" | cut -d',' -f2)
    V_RMSE=$(echo "$rmse_line" | cut -d',' -f3)

    echo "  -> RMSE: E=${E_RMSE}, F=${F_RMSE}, V=${V_RMSE}"
    echo "${base},${E_RMSE},${F_RMSE},${V_RMSE}" >> "../rmse_summary.csv"

    cd - > /dev/null
done

echo "Done. Summary in ${SUMMARY}"

# ---- Generate parity plots and bar plots for RMSE ----
"${PYTHON}" "${PLOT_SCRIPT}" --root "${OUT_ROOT}" --summary "${SUMMARY}"

# # ---- Generate parity plots and bar plots for RMSE ----
# "${PYTHON}" - <<PY
# import os, math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams["pdf.fonttype"] = 42
# mpl.rcParams["ps.fonttype"] = 42
# mpl.rcParams["font.family"] = "serif"
# mpl.rcParams["font.serif"] = ["DejaVu Serif"]

# root = "${OUT_ROOT}"
# summary_csv = os.path.join(root, "rmse_summary.csv")

# plt.rcParams.update({'font.size': 16})

# # ============================================================
# #  Collect prediction/reference data for parity plots
# # ============================================================

# def collect_energy(root):
#     e_pred, e_ref = [], []
#     for name in os.listdir(root):
#         d = os.path.join(root, name)
#         if not os.path.isdir(d): continue
#         f = os.path.join(d, "energy_train.out")
#         if not os.path.exists(f): continue
#         a = np.loadtxt(f)
#         if a.ndim == 1: a = a.reshape(1, -1)
#         if a.shape[1] < 2: continue
#         e_pred.append(a[:, 0])
#         e_ref.append(a[:, 1])
#     if not e_pred: return None, None
#     return np.concatenate(e_pred), np.concatenate(e_ref)

# def collect_force(root):
#     f_pred, f_ref = [], []
#     for name in os.listdir(root):
#         d = os.path.join(root, name)
#         if not os.path.isdir(d): continue
#         f = os.path.join(d, "force_train.out")
#         if not os.path.exists(f): continue
#         a = np.loadtxt(f)
#         if a.ndim == 1: a = a.reshape(1, -1)
#         if a.shape[1] != 6: continue
#         Fx_p, Fy_p, Fz_p, Fx_r, Fy_r, Fz_r = a.T
#         f_pred.append(np.concatenate([Fx_p, Fy_p, Fz_p]))
#         f_ref.append(np.concatenate([Fx_r, Fy_r, Fz_r]))
#     if not f_pred: return None, None
#     return np.concatenate(f_pred), np.concatenate(f_ref)

# def collect_virial(root):
#     v_pred, v_ref = [], []
#     for name in os.listdir(root):
#         d = os.path.join(root, name)
#         if not os.path.isdir(d): continue
#         vf = os.path.join(d, "virial_train.out")
#         sf = os.path.join(d, "stress_train.out")
#         f = vf if os.path.exists(vf) else sf if os.path.exists(sf) else None
#         if f is None: continue
#         a = np.loadtxt(f)
#         if a.ndim == 1: a = a.reshape(1, -1)
#         if a.shape[1] % 2 != 0: continue
#         ncomp = a.shape[1] // 2
#         v_pred.append(a[:, :ncomp].ravel())
#         v_ref.append(a[:, ncomp:].ravel())
#     if not v_pred: return None, None
#     return np.concatenate(v_pred), np.concatenate(v_ref)

# # ============================================================
# #  Plot helper
# # ============================================================

# def parity_plot(pred, ref, xlabel, ylabel, title, outfile):
#     if pred is None: 
#         print(f"[WARN] No data for {outfile}")
#         return
#     rmse = math.sqrt(np.mean((pred - ref)**2))

#     vmin = min(pred.min(), ref.min())
#     vmax = max(pred.max(), ref.max())
#     margin = 0.05 * (vmax - vmin)
#     lo, hi = vmin - margin, vmax + margin

#     plt.figure(figsize=(6, 6))
#     plt.scatter(ref, pred, s=5, alpha=0.4)
#     plt.plot([lo, hi], [lo, hi], 'k--')
#     plt.xlim(lo, hi)
#     plt.ylim(lo, hi)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.text(0.05, 0.95, f"RMSE={rmse:.3e}", transform=plt.gca().transAxes,
#              ha="left", va="top", fontsize=14)
#     plt.tight_layout()
#     plt.savefig(os.path.join(root, outfile), dpi=200)
#     plt.close()
#     print(f"[OK] Saved {outfile}")

# # ============================================================
# #  Make parity plots
# # ============================================================

# e_p, e_r = collect_energy(root)
# parity_plot(e_p, e_r,
#             "DFT Energy (eV)", "NEP Energy (eV)",
#             "Energy Parity (All Tests)", "parity_energy.png")

# f_p, f_r = collect_force(root)
# parity_plot(f_p, f_r,
#             "DFT Force (eV/Å)", "NEP Force (eV/Å)",
#             "Force Parity (All Tests)", "parity_force.png")

# v_p, v_r = collect_virial(root)
# parity_plot(v_p, v_r,
#             "DFT Virial/Stress (eV/Å³)", "NEP Virial/Stress (eV/Å³)",
#             "Virial Parity (All Tests)", "parity_virial.png")

# # ============================================================
# #  Bar plots from RMSE summary
# # ============================================================

# df = pd.read_csv(summary_csv)

# def plot_bar(df, col, ylabel, outname):
#     plt.figure(figsize=(14, 6))
#     x = df["dataset"]
#     y = df[col]
#     plt.bar(x, y, color="steelblue", edgecolor="black")
#     plt.xticks(rotation=75, fontsize=14, ha="right")
#     plt.ylabel(ylabel)
#     plt.title(f"{col} per dataset")
#     plt.tight_layout()
#     plt.savefig(os.path.join(root, outname), dpi=200)
#     plt.savefig(os.path.join(root, outname,".pdf"), bbox_inches="tight")
#     plt.close()
#     print(f"[OK] Saved {outname}")

# plot_bar(df, "E_RMSE", "Energy RMSE (eV)",   "bar_E_RMSE.png")
# plot_bar(df, "F_RMSE", "Force RMSE (eV/Å)", "bar_F_RMSE.png")
# plot_bar(df, "V_RMSE", "Virial RMSE (eV/Å³)", "bar_V_RMSE.png")

# PY
# # ============================================================
echo "All parity and bar plots generated."