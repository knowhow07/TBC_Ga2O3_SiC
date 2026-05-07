#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
# --- REVISED PART 1: imports (add datetime) ---
from datetime import datetime

# --- REVISED PART: add near the top (user knobs) ---
path = "<PATH_TO_DATA>/vasp_jobs/nep_0217/emd/Ga2O3/750k"      # folders to search
INPUT_GLOB = path + "/job_*" 
HAC_NAME   = "hac.out"    # hac filename inside each folder

# ----------------------------
# Output directory (save everything into input folder)
# ----------------------------
OUT_DIR = os.path.abspath(path)
os.makedirs(OUT_DIR, exist_ok=True)

def outp(fname: str) -> str:
    return os.path.join(OUT_DIR, fname)


# --- REVISED PART 2: add this right AFTER imports (before you start printing) ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = outp(f"postprocess_{timestamp}.log")




# --- REVISED PART: new user knobs (add near MAX_T_PLOT) ---
USE_PLATEAU = True        # True: auto-plateau from kiso_avg(t); False: use last LAST_WINDOW_PS
LAST_WINDOW_PS = 10.0     # only used when USE_PLATEAU=False
# ----------------------------
# User knobs for HFACF columns
# ----------------------------
# >>> NEW: set which columns in hac.out contain the heat-flux autocorrelation
# Indices are 0-based. Adjust these to match your hac.out layout.
# Example: if columns 1,2,3 are Cxx, Cyy, Czz:
HFACF_COLS = (1, 2, 3)   # (Cxx_col, Cyy_col, Czz_col)
# If you're not sure, print a line of hac.out and inspect.
# ----------------------------
# Plotting options
# ----------------------------
# Set the maximum correlation time (in ps) to show in plots.
# Use None to plot the full range.
MAX_T_PLOT = 300      # e.g. 50.0 for 0–50 ps, or None for all

def log(msg: str):
    print(msg)
    with open(logfile, "a") as f:
        f.write(msg + "\n")



# ----------------------------
# Collect all hac.out files
# ----------------------------
# --- REVISED PART: add this function near the top (before collecting files) ---
def collect_hac_files(folder_glob: str, hac_name: str) -> list[str]:
    folders = sorted(glob.glob(folder_glob))
    return [os.path.join(d, hac_name) for d in folders if os.path.exists(os.path.join(d, hac_name))]

# --- REVISED PART: replace the original "Collect all hac.out files" block ---
hac_files = collect_hac_files(INPUT_GLOB, HAC_NAME)

if len(hac_files) == 0:
    raise SystemExit(f"No {HAC_NAME} files found under {INPUT_GLOB}!")

log(f"Found {len(hac_files)} {HAC_NAME} files under {INPUT_GLOB}.")


kx_list = []
ky_list = []
kz_list = []
kiso_list = []
t_ref = None

# >>> NEW: lists for HFACF components
Cxx_list = []
Cyy_list = []
Czz_list = []

# ----------------------------
# Process each hac.out
# ----------------------------
for hac in hac_files:
    data = np.loadtxt(hac)
    t = data[:, 0]  # time in ps (your current convention)

    # Save the reference time array once
    if t_ref is None:
        t_ref = t
    else:
        if len(t) != len(t_ref):
            raise ValueError(f"Time array length mismatch in {hac}")

    # Columns:
    # col6 = k_x_in, col7 = k_x_out, col8 = k_y_in, col9 = k_y_out, col10 = k_z_tot
    k_x_in  = data[:, 6]
    k_x_out = data[:, 7]
    k_y_in  = data[:, 8]
    k_y_out = data[:, 9]
    k_z_tot = data[:, 10]

    k_x_tot = k_x_in + k_x_out
    k_y_tot = k_y_in + k_y_out
    k_iso   = (k_x_tot + k_y_tot + k_z_tot) / 3.0

    kx_list.append(k_x_tot)
    ky_list.append(k_y_tot)
    kz_list.append(k_z_tot)
    kiso_list.append(k_iso)

    # >>> NEW: grab HFACF components (Cxx, Cyy, Czz) per run
    cxx = data[:, HFACF_COLS[0]]
    cyy = data[:, HFACF_COLS[1]]
    czz = data[:, HFACF_COLS[2]]

    Cxx_list.append(cxx)
    Cyy_list.append(cyy)
    Czz_list.append(czz)

# Convert lists → arrays
kx_arr   = np.vstack(kx_list)
ky_arr   = np.vstack(ky_list)
kz_arr   = np.vstack(kz_list)
kiso_arr = np.vstack(kiso_list)

# >>> NEW: HFACF arrays
Cxx_arr  = np.vstack(Cxx_list)
Cyy_arr  = np.vstack(Cyy_list)
Czz_arr  = np.vstack(Czz_list)

# Averages across jobs
kx_avg   = kx_arr.mean(axis=0)
ky_avg   = ky_arr.mean(axis=0)
kz_avg   = kz_arr.mean(axis=0)
kiso_avg = kiso_arr.mean(axis=0)

# >>> NEW: average HFACF
Cxx_avg  = Cxx_arr.mean(axis=0)
Cyy_avg  = Cyy_arr.mean(axis=0)
Czz_avg  = Czz_arr.mean(axis=0)

# ============================================================
# 0. Plot HFACF and its running integral (to inspect correlation time)
# ============================================================

# Time mask for plotting (0–MAX_T_PLOT)
if MAX_T_PLOT is not None:
    plot_mask = t_ref <= MAX_T_PLOT
else:
    plot_mask = np.ones_like(t_ref, dtype=bool)

t_plot = t_ref[plot_mask]


# >>> normalized HFACF (using masked time window)
Cxx_norm = Cxx_avg / Cxx_avg[0]
Cyy_norm = Cyy_avg / Cyy_avg[0]
Czz_norm = Czz_avg / Czz_avg[0]

plt.figure(figsize=(8, 5))
plt.plot(t_plot, Cxx_norm[plot_mask], label="Cxx / Cxx(0)")
plt.plot(t_plot, Cyy_norm[plot_mask], label="Cyy / Cyy(0)")
plt.plot(t_plot, Czz_norm[plot_mask], label="Czz / Czz(0)")
plt.xlabel("Correlation time t (ps)")
plt.ylabel("Normalized HFACF")
plt.title("Heat Flux Autocorrelation (normalized, averaged over runs)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(outp("emd_hfacf_normalized_avg.png"), dpi=300); plt.close()
log("Saved HFACF figure: emd_hfacf_normalized_avg.png")

# >>> running integral of isotropic HFACF (shape only, masked)
C_iso = (Cxx_avg + Cyy_avg + Czz_avg) / 3.0
C_iso_plot = C_iso[plot_mask]

# time step in ps within the plotting window
dt_ps = np.median(np.diff(t_plot))

hfacf_int = np.cumsum(0.5 * (C_iso_plot[1:] + C_iso_plot[:-1]) * dt_ps)
t_mid = 0.5 * (t_plot[1:] + t_plot[:-1])

plt.figure(figsize=(8, 5))
plt.plot(t_mid, hfacf_int)
plt.xlabel("Correlation time t (ps)")
plt.ylabel("Cumulative ∫ C_iso(t) dt (arb. units)")
plt.title("Running integral of HFACF (for correlation time inspection)")
plt.grid(True)
plt.tight_layout()
plt.savefig(outp("emd_hfacf_running_integral.png"), dpi=300); plt.close()
log("Saved HFACF running-integral figure: emd_hfacf_running_integral.png")

# ============================================================
# 1. Find plateau window from kiso_avg(t)
# ============================================================

def find_plateau_indices(t, y,
                         min_fraction=0.3,
                         window_ps=50.0,
                         slope_threshold=5e-4):
    """
    Heuristic plateau finder:
      - Smooth y with a moving average of length ~window_ps
      - Compute dy/dt
      - Find the longest contiguous region at t >= min_fraction * t_max
        where |dy/dt| < slope_threshold.
    Returns: (i_start, i_end) indices (Python slice: [i_start:i_end])
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if len(t) < 5:
        raise RuntimeError("Time array too short to find plateau.")

    # Estimate time step and choose smoothing window length in points
    dt = np.median(np.diff(t))
    nwin = max(int(window_ps / dt), 1)

    # Simple moving average smoothing
    kernel = np.ones(nwin) / nwin
    y_smooth = np.convolve(y, kernel, mode="same")

    # Numerical derivative
    dy_dt = np.gradient(y_smooth, t)

    # Only consider times after a fraction of the total simulation
    t_min = t[0] + min_fraction * (t[-1] - t[0])
    mask = (np.abs(dy_dt) < slope_threshold) & (t >= t_min)

    # Find longest contiguous True segment in mask
    best_len = 0
    best_start = None
    cur_start = None

    for i, m in enumerate(mask):
        if m:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                cur_len = i - cur_start
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
                cur_start = None

    # Handle case where plateau extends to the last point
    if cur_start is not None:
        cur_len = len(mask) - cur_start
        if cur_len > best_len:
            best_len = cur_len
            best_start = cur_start

    if best_len == 0 or best_start is None:
        raise RuntimeError("No plateau region found; try relaxing thresholds.")

    i_start = best_start
    i_end = best_start + best_len
    return i_start, i_end


if USE_PLATEAU:
    # Use kiso_avg to identify plateau region
    i0, i1 = find_plateau_indices(
        t_ref, kiso_avg,
        min_fraction=0.3,
        window_ps=50.0,
        slope_threshold=5e-4
    )
    method_str = "plateau (from k_iso avg)"
else:
    # Use the last LAST_WINDOW_PS (no plateau detection)
    t_end_target = t_ref[-1]
    t_start_target = max(t_ref[0], t_end_target - LAST_WINDOW_PS)
    i0 = int(np.searchsorted(t_ref, t_start_target, side="left"))
    i1 = len(t_ref)
    method_str = f"last {LAST_WINDOW_PS:.2f} ps"

t_start, t_end = t_ref[i0], t_ref[i1 - 1]
log(f"Averaging window [{method_str}]: {t_start:.2f} – {t_end:.2f} ps ({i1 - i0} points)")


# ============================================================
# 2. Compute stats (mean, std, stderr) per component over plateau
# ============================================================

def window_stats(arr, i_start, i_end):
    """
    arr shape: (n_runs, n_t)
    Returns:
        mean_over_runs, std_over_runs, stderr_over_runs
    where each run is first averaged over the plateau indices.
    """
    # Mean κ over plateau for each run
    vals_per_run = arr[:, i_start:i_end].mean(axis=1)
    mean = vals_per_run.mean()
    std = vals_per_run.std(ddof=1)  # sample std
    stderr = std / np.sqrt(arr.shape[0])
    return mean, std, stderr


results = {}
for name, arr in [
    ("kx_tot", kx_arr),
    ("ky_tot", ky_arr),
    ("kz_tot", kz_arr),
    ("k_iso",  kiso_arr),
]:
    mean, std, stderr = window_stats(arr, i0, i1)
    results[name] = (mean, std, stderr)
    log(f"{name:6s}: mean = {mean:8.3f} W/mK, std = {std:8.3f}, stderr = {stderr:8.3f}")


# ============================================================
# 3. Save stats to a text file
# ============================================================

with open("kappa_plateau_stats.txt", "w") as f:
    f.write("# Plateau window (ps): "
            f"{t_start:.6f}  {t_end:.6f}\n")
    f.write("# name    mean(W/mK)      std(W/mK)       stderr(W/mK)\n")
    for name, (mean, std, stderr) in results.items():
        f.write(f"{name:6s}  {mean:13.6f}  {std:13.6f}  {stderr:13.6f}\n")

log("Saved plateau statistics to: kappa_plateau_stats.txt")


# ----------------------------
# Save averaged kappa vs time
# ----------------------------
out_data = np.column_stack([t_ref, kx_avg, ky_avg, kz_avg, kiso_avg])

header = (
    "t(ps)   kx_avg(W/mK)   ky_avg(W/mK)   kz_avg(W/mK)   kiso_avg(W/mK)\n"
    "Averaged over {} runs".format(len(kiso_arr))
)

np.savetxt(outp("kappa_avg_vs_time.txt"), out_data, header=header, fmt="%.6f")
log("Saved averaged kappa curves to: kappa_avg_vs_time.txt")

# ----------------------------
# Also save ONLY the final values
# ----------------------------
final_values = np.array([
    kx_avg[-1],
    ky_avg[-1],
    kz_avg[-1],
    kiso_avg[-1]
])

np.savetxt(
    outp("kappa_final_values.txt"),
    final_values,
    header="kx_avg_final   ky_avg_final   kz_avg_final   kiso_avg_final\n",
    fmt="%.6f"
)

log("Saved final averaged kappa values to: kappa_final_values.txt")


# ----------------------------
# Compute final averaged values + find the max
# ----------------------------
final_values = {
    "kx_tot (avg)": kx_avg[-1],
    "ky_tot (avg)": ky_avg[-1],
    "kz_tot (avg)": kz_avg[-1],
    "k_iso (avg)":  kiso_avg[-1],
}

max_label = max(final_values, key=final_values.get)
max_value = final_values[max_label]

# ----------------------------
# Plot running κ(t)
# ----------------------------
plt.rcParams['font.size'] = 16
plt.figure(figsize=(10, 6))

# Light lines: each job's k_iso, masked
for i in range(kiso_arr.shape[0]):
    plt.plot(t_plot, kiso_arr[i, plot_mask], color="gray", alpha=0.25, linewidth=1)

# Dark averaged lines, masked
plt.plot(t_plot, kx_avg[plot_mask],   label="kx_tot (avg)")
plt.plot(t_plot, ky_avg[plot_mask],   label="ky_tot (avg)")
plt.plot(t_plot, kz_avg[plot_mask],   label="kz_tot (avg)")
plt.plot(t_plot, kiso_avg[plot_mask], label="k_iso (avg)", linewidth=3)

plt.xlabel("Correlation time t (ps)")
plt.ylabel("Running κ(t) [W/mK]")
plt.legend()

# Single annotation at top center (uses full max_final κ, not masked)
plt.text(
    0.5, 1.02,
    f"max_final κ = {max_label}: {max_value:.2f} W/mK",
    transform=plt.gca().transAxes,
    ha="center",
    va="bottom",
    fontsize=16,
)

plt.tight_layout()
plt.savefig(outp("emd_running_kappa_avg.png"), dpi=300); plt.close()
log(f"Saved HFACF figure: {outp('emd_hfacf_normalized_avg.png')}")