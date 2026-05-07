#!/usr/bin/env python3
import numpy as np
import math
import re
import datetime
import sys
import os  # <<< NEW

# ---- create timestamped log file ----
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"tbc_results_{timestamp}.txt"

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

sys.stdout = Logger(log_filename)

# ---- filenames and knobs ----
compute_file = "compute.out"
model_file   = "model.xyz"

JOB_PREFIX = "job_"   # <<< NEW: auto-scan folders with this prefix

skip_fraction    = 0.2     # use last 50% of rows as steady state
time_step_fs     = 1.0     # GPUMD time_step
sample_interval  = 1       # from "compute 0 200 100 temperature"
output_interval  = 100000
# fitting windows for each material (Å) – ADJUST
x_sic_min, x_sic_max = 25.0, 75.0
x_ga_min, x_ga_max   = 125.0, 175.0

x_int = 95        # interface position (Å), adjust from structure
# --------------------


def read_box_from_model_xyz(fname):
    with open(fname, "r") as f:
        _ = f.readline()          # N
        header = f.readline().strip()
    m = re.search(r'Lattice="([^"]+)"', header)
    if m is None:
        raise RuntimeError("Cannot find Lattice in model.xyz header")
    lat_vals = np.fromstring(m.group(1), sep=" ")
    if lat_vals.size != 9:
        raise RuntimeError("Expected 9 numbers in Lattice")
    a1 = lat_vals[0:3]
    a2 = lat_vals[3:6]
    a3 = lat_vals[6:9]
    Lx = np.linalg.norm(a1)
    # cross-sectional area perpendicular to heat direction (a1)
    area = np.linalg.norm(np.cross(a2, a3))    # Å^2
    return Lx, area


def analyze_one(job_dir):  # <<< NEW: factor out analysis for a single folder
    """Run the TBC analysis for one job directory and return (G, TBR)."""

    cf = os.path.join(job_dir, compute_file)
    mf = os.path.join(job_dir, model_file)

    if not os.path.exists(cf):
        raise FileNotFoundError(f"{cf} not found")
    if not os.path.exists(mf):
        raise FileNotFoundError(f"{mf} not found")

    print(f"\n===== Job: {job_dir} =====")

    # box
    Lx_ang, area_A2 = read_box_from_model_xyz(mf)
    print(f"Lx = {Lx_ang:.3f} Å, cross-section area = {area_A2:.3f} Å²")

    # load compute.out
    data = np.loadtxt(cf)
    nrows, ncols = data.shape
    M = ncols - 2                       # number of groups
    T_all   = data[:, :M]               # (nrows, M)
    E_hot   = data[:, -2]               # (nrows,)
    E_cold  = data[:, -1]

    # steady-state rows
    start_idx = int(math.floor(nrows * skip_fraction))
    if start_idx >= nrows:
        start_idx = nrows - 1
    T_ss    = T_all[start_idx:, :]
    E_hot_ss  = E_hot[start_idx:]
    E_cold_ss = E_cold[start_idx:]
    n_ss = T_ss.shape[0]
    print(f"Using last {n_ss} rows out of {nrows} as steady state")

    # ---- print all parameters to logfile ----
    print("==== Thermal analysis run ====")
    print(f"Timestamp: {timestamp}")
    print(f"compute_file       = {cf}")
    print(f"model_file         = {mf}")
    print("")
    print("---- User parameters ----")
    print(f"skip_fraction      = {skip_fraction}")
    print(f"time_step_fs       = {time_step_fs}")
    print(f"sample_interval    = {sample_interval}")
    print(f"output_interval    = {output_interval}")
    print(f"x_sic_min/max      = {x_sic_min}, {x_sic_max}")
    print(f"x_ga_min/max       = {x_ga_min}, {x_ga_max}")
    print(f"x_int              = {x_int}")
    print("==========================\n")

    # time vector for rows (ps)
    dt_row_fs = time_step_fs * sample_interval * output_interval
    dt_row_ps = dt_row_fs * 1e-3
    t_ps_full = np.arange(nrows) * dt_row_ps
    t_ps_ss   = t_ps_full[start_idx:]

    # average temperature per group over time (ignoring NaNs)
    T_avg = np.nanmean(T_ss, axis=0)    # (M,)
    x_bins = (np.arange(M) + 0.5) * Lx_ang / M   # bin centers (Å)

    # ---- fit Ga2O3 and SiC regions ----
    mask_sic = (x_bins >= x_sic_min) & (x_bins <= x_sic_max)   # LEFT
    mask_ga  = (x_bins >= x_ga_min)  & (x_bins <= x_ga_max)    # RIGHT

    if not np.any(mask_sic):
        raise RuntimeError("No bins in SiC fit window")
    if not np.any(mask_ga):
        raise RuntimeError("No bins in Ga2O3 fit window")

    coef_sic = np.polyfit(x_bins[mask_sic], T_avg[mask_sic], 1)
    coef_ga  = np.polyfit(x_bins[mask_ga],  T_avg[mask_ga],  1)

    slope_sic = coef_sic[0]   # K/Å
    slope_ga  = coef_ga[0]    # K/Å

    print(f"slope SiC   dT/dx = {slope_sic:.4f} K/Å")
    print(f"slope Ga2O3 dT/dx = {slope_ga:.4f} K/Å")

    # ---- heat flux from thermostat energies ----
    p_hot  = np.polyfit(t_ps_ss, E_hot_ss,  1)
    p_cold = np.polyfit(t_ps_ss, E_cold_ss, 1)
    dE_hot_dt  = p_hot[0]   # eV/ps
    dE_cold_dt = p_cold[0]  # eV/ps

    print(f"dE_hot/dt  = {dE_hot_dt:.4f} eV/ps")
    print(f"dE_cold/dt = {dE_cold_dt:.4f} eV/ps")

    # power
    P_hot  = -dE_hot_dt      # hot thermostat gives energy to system
    P_cold =  dE_cold_dt     # cold thermostat removes energy from system
    P_eV   = 0.5 * (P_hot + P_cold)   # average power [eV/ps]

    # Use magnitude of the flux
    q_eV = abs(P_eV) / area_A2       # eV/(ps·Å²)
    print(f"heat flux |q| = {q_eV:.6e} eV/(ps·Å²)")

    # ---- thermal conductivity in each material ----
    evtoWatt_over_KperA = 1.60217733e3
    k_sic = evtoWatt_over_KperA * q_eV / abs(slope_sic)
    k_ga  = evtoWatt_over_KperA * q_eV / abs(slope_ga)

    print(f"k_Ga2O3 = {k_ga:.2f} W/mK")
    print(f"k_SiC   = {k_sic:.2f} W/mK")

    # ---- TBC ----
    T_sic_int = np.polyval(coef_sic, x_int)
    T_ga_int  = np.polyval(coef_ga,  x_int)
    dT_int = T_sic_int - T_ga_int   # drop across interface SiC → Ga2O3

    print(f"T_SiC  (interface) = {T_sic_int:.2f} K")
    print(f"T_Ga2O3(interface) = {T_ga_int:.2f} K")
    print(f"ΔT_interface (SiC - Ga2O3) = {dT_int:.3f} K")

    q_Wm2 = q_eV * 1.602176634e13
    G = q_Wm2 / abs(dT_int)
    R = 1.0 / G

    G_MW   = G * 1e-6         # MW/(m²·K)
    R_GW   = R * 1e9         # m²·K/GW
    print(f"TBC G = {G_MW:.3f} MW/(m²·K)")
    print(f"R_K   = {R_GW:.3f} m²·K/GW")


    # ---- TBR (thermal boundary resistance, ΔT / Jq) ----
    Jq_eVpsA2 = q_eV          # already |P| / A  in eV/(ps·Å²)
    Jq_Wm2 = Jq_eVpsA2 * 1.602176634e13    # eV/(ps·Å²) → W/m²
    TBR = abs(dT_int) / Jq_Wm2           # m²·K/W
    TBR_GW = TBR * 1e9                  # m²·K/GW

    print(f"TBR  = {TBR_GW:.3f} m²·K/GW   (ΔT / J_Q)")

    # return TBC & TBR for averaging
    return G_MW, R_GW, k_sic, k_ga


def main():  # <<< NEW main: loop over job_* folders and average
    # find job directories with the chosen prefix
    k_sic_list = []    # <<< NEW
    k_ga_list  = []    # <<< NEW


    job_dirs = [d for d in os.listdir(".")
                if os.path.isdir(d) and d.startswith(JOB_PREFIX)]
    job_dirs = sorted(job_dirs)
    if not job_dirs:
        print(f"No job folders starting with '{JOB_PREFIX}' found.")
        return

    print("Found job folders:", job_dirs)

    G_list = []
    TBR_list = []
    ok_jobs = []

    for jd in job_dirs:
        try:
            G, Rk, ks, kg = analyze_one(jd)   # <<< NEW unpack
            G_list.append(G)
            TBR_list.append(Rk)
            ok_jobs.append(jd)
            k_sic_list.append(ks)             # <<< NEW
            k_ga_list.append(kg)              # <<< NEW

        except Exception as e:
            print(f"[SKIP] {jd}: {e}")

    if not G_list:
        print("\nNo successful jobs to average.")
        return

    G_arr   = np.array(G_list)
    TBR_arr = np.array(TBR_list)

    # --- averages ---
    G_mean    = G_arr.mean()
    TBR_mean  = TBR_arr.mean()
    k_sic_arr = np.array(k_sic_list)
    k_ga_arr  = np.array(k_ga_list)

    # --- standard errors (SEM = SD / sqrt(N)) ---
    n_G   = G_arr.size
    n_TBR = TBR_arr.size
    n_ks  = k_sic_arr.size
    n_kg  = k_ga_arr.size

    G_sem    = G_arr.std(ddof=1)   / np.sqrt(n_G)   if n_G   > 1 else 0.0
    TBR_sem  = TBR_arr.std(ddof=1) / np.sqrt(n_TBR) if n_TBR > 1 else 0.0
    k_sic_sem = k_sic_arr.std(ddof=1) / np.sqrt(n_ks) if n_ks > 1 else 0.0
    k_ga_sem  = k_ga_arr.std(ddof=1)  / np.sqrt(n_kg) if n_kg > 1 else 0.0

    print("\n===== Summary over jobs =====")
    print("Included jobs:", ok_jobs)
    print(f"Avg k_SiC   = {k_sic_arr.mean():.3f} ± {k_sic_sem:.3f} W/mK")
    print(f"Avg k_Ga2O3 = {k_ga_arr.mean():.3f} ± {k_ga_sem:.3f} W/mK")
    print(f"Average TBC G   = {G_mean:.3e} ± {G_sem:.3e} MW/(m²·K)")
    print(f"Average TBR     = {TBR_mean:.3e} ± {TBR_sem:.3e} m²K/GW")
    print("================================")


if __name__ == "__main__":
    main()
