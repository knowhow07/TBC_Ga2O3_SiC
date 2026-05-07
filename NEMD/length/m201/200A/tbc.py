#!/usr/bin/env python3
import numpy as np
import math
import re
import datetime
import sys

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


compute_file = "compute.out"
model_file   = "model.xyz"

# ---- user knobs ----
skip_fraction    = 0.2     # use last 50% of rows as steady state
time_step_fs     = 1.0     # GPUMD time_step
sample_interval  = 1     # from "compute 0 200 100 temperature"
output_interval  = 500000
# fitting windows for each material (Å) – ADJUST
x_sic_min, x_sic_max = 30.0, 90.0
x_ga_min, x_ga_max   = 120.0, 175.0

x_int = 105.0        # interface position (Å), adjust from structure
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


def main():
    # box
    Lx_ang, area_A2 = read_box_from_model_xyz(model_file)
    print(f"Lx = {Lx_ang:.3f} Å, cross-section area = {area_A2:.3f} Å²")

    # load compute.out
    data = np.loadtxt(compute_file)
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
    print(f"Using last {n_ss} rows out of {nrows} as steady state")# ---- print all parameters to logfile ----
    
    print("==== Thermal analysis run ====")
    print(f"Timestamp: {timestamp}")
    print(f"compute_file       = {compute_file}")
    print(f"model_file         = {model_file}")
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

    coef_sic = np.polyfit(x_bins[mask_sic], T_avg[mask_sic], 1)
    coef_ga  = np.polyfit(x_bins[mask_ga],  T_avg[mask_ga],  1)

    slope_sic = coef_sic[0]   # K/Å
    slope_ga  = coef_ga[0]    # K/Å

    print(f"slope SiC   dT/dx = {slope_sic:.4f} K/Å")
    print(f"slope Ga2O3 dT/dx = {slope_ga:.4f} K/Å")


    # ---- heat flux from thermostat energies ----
    # original linear fits:
    p_hot  = np.polyfit(t_ps_ss, E_hot_ss,  1)
    p_cold = np.polyfit(t_ps_ss, E_cold_ss, 1)
    dE_hot_dt  = p_hot[0]   # eV/ps
    dE_cold_dt = p_cold[0]  # eV/ps

    print(f"dE_hot/dt  = {dE_hot_dt:.4f} eV/ps")
    print(f"dE_cold/dt = {dE_cold_dt:.4f} eV/ps")

    # In GPUMD Langevin, if E_hot decreases and E_cold increases,
    # power INTO the system is:
    P_hot  = -dE_hot_dt      # hot thermostat gives energy to system
    P_cold =  dE_cold_dt     # cold thermostat removes energy from system
    P_eV   = 0.5 * (P_hot + P_cold)   # average power [eV/ps]

    # Use magnitude of the flux (direction doesn’t matter for κ)
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


    print(f"TBC G = {G:.3e} W/(m²·K)")
    print(f"R_K   = {R:.3e} m²K/W")
    # ---- TBR (thermal boundary resistance, ΔT / Jq) ----
    # J_Q must use the SIGNED power sum (energy increase)
    # but magnitude for flux
    Jq_eVpsA2 = q_eV          # already |P| / A  in eV/(ps·Å²)

    # Convert Jq to W/m² for TBR
    Jq_Wm2 = Jq_eVpsA2 * 1.602176634e13    # eV/(ps·Å²) → W/m²

    # TBR = ΔT / J_Q   (units: m²·K/W)
    TBR = abs(dT_int) / Jq_Wm2

    print(f"TBR  = {TBR:.3e} m²K/W   (ΔT / J_Q)")



if __name__ == "__main__":
    main()
