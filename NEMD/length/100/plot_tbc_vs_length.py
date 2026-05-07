#!/usr/bin/env python3
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# ------------ Helpers ------------

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

JOB_FOLDERS=["100A","200A","300A","400A","600A"]  # list of thickness folders to analyze
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_name = f"TBC_thickness_analysis_{timestamp}.log"
log = open(log_name, "w")

def log_print(msg):
    print(msg)
    log.write(msg + "\n")

log_print("===== Thickness TBC Analysis =====")
log_print(f"Timestamp: {timestamp}")


def read_Lx_from_model_xyz(path):
    """
    Read Lx (Å) from model.xyz in given folder.
    Assumes header has Lattice="a1x a1y a1z a2x a2y a2z a3x a3y a3z".
    """
    fname = os.path.join(path, "model.xyz")
    with open(fname, "r") as f:
        _ = f.readline()          # N
        header = f.readline().strip()

    m = re.search(r'Lattice="([^"]+)"', header)
    if m is None:
        raise RuntimeError(f"Cannot find Lattice in {fname}")
    lat_vals = np.fromstring(m.group(1), sep=" ")
    if lat_vals.size != 9:
        raise RuntimeError(f"Expected 9 lattice numbers in {fname}")

    a1 = lat_vals[0:3]
    Lx = np.linalg.norm(a1)
    return Lx


def parse_latest_tbc_results(path):
    """
    Find the latest tbc_results_*.txt in this folder and parse:
    Avg k_SiC, Avg k_Ga2O3, Average TBC G (means and sems).
    Units in file:
      k_*   : W/mK
      G     : MW/(m²·K)
    Returns (k_sic_mean, k_sic_sem, k_ga_mean, k_ga_sem, G_mean, G_sem).
    """
    files = sorted(glob.glob(os.path.join(path, "tbc_results_*.txt")))
    if not files:
        raise FileNotFoundError(f"No tbc_results_*.txt in {path}")
    fname = files[-1]  # lexicographically last -> latest timestamp
    k_sic_mean = k_sic_sem = None
    k_ga_mean = k_ga_sem = None
    G_mean = G_sem = None

    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            m = re.search(r"Avg k_SiC\s*=\s*([0-9.eE+-]+)\s*±\s*([0-9.eE+-]+)", line)
            if m:
                k_sic_mean = float(m.group(1))
                k_sic_sem  = float(m.group(2))
                continue
            m = re.search(r"Avg k_Ga2O3\s*=\s*([0-9.eE+-]+)\s*±\s*([0-9.eE+-]+)", line)
            if m:
                k_ga_mean = float(m.group(1))
                k_ga_sem  = float(m.group(2))
                continue
            m = re.search(r"Average TBC G\s*=\s*([0-9.eE+-]+)\s*±\s*([0-9.eE+-]+)", line)
            if m:
                G_mean = float(m.group(1))
                G_sem  = float(m.group(2))
                continue

    if None in (k_sic_mean, k_sic_sem, k_ga_mean, k_ga_sem, G_mean, G_sem):
        raise RuntimeError(f"Failed to parse all values from {fname}")

    return k_sic_mean, k_sic_sem, k_ga_mean, k_ga_sem, G_mean, G_sem


# ------------ Main ------------

def main():
    # use only the folder names you define in JOB_FOLDERS
    folders = [
        d for d in JOB_FOLDERS
        if os.path.isdir(d)
    ]

    if not folders:
        print(f"No valid folders found in JOB_FOLDERS: {JOB_FOLDERS}")
        return

    for folder in folders:
        print(f"Processing folder: {folder}")
        # your existing processing code here

    folders = sorted(folders)
    print("Found folders:", folders)

    Lxs = []
    G_means = []
    G_sems = []
    k_sic_means = []
    k_sic_sems = []
    k_ga_means = []
    k_ga_sems = []
    labels = []

    R_means = []
    R_sems  = []
    invLs   = []

    for d in folders:
        try:
            Lx = read_Lx_from_model_xyz(d)
            k_sic_mean, k_sic_sem, k_ga_mean, k_ga_sem, G_mean, G_sem = parse_latest_tbc_results(d)
                        # inside the for d in folders: loop, right after parsing G_mean, G_sem
            R_mean = 1e3 / G_mean                  # (m²K/GW) since G is in MW/(m²K)
            R_sem  = (1e3 * G_sem) / (G_mean**2)   # error propagation
            invL   = 1.0 / Lx


        except Exception as e:
            print(f"[SKIP] {d}: {e}")
            continue
        # --- REVISED PART C: after the loop, when converting to numpy ---
        R_means.append(R_mean)
        R_sems.append(R_sem)
        invLs.append(invL)


        Lxs.append(Lx)
        G_means.append(G_mean)
        G_sems.append(G_sem)
        k_sic_means.append(k_sic_mean)
        k_sic_sems.append(k_sic_sem)
        k_ga_means.append(k_ga_mean)
        k_ga_sems.append(k_ga_sem)
        labels.append(d)

        R_mean = 1e3 / G_mean
        R_sem  = (1e3 * G_sem) / (G_mean**2)

        invL = 1.0 / Lx

        log_print(
            f"{d}: "
            f"Lx={Lx:.3f} Å, 1/Lx={invL:.6e} 1/Å, "
            f"G={G_mean:.2f}±{G_sem:.2f} MW/m²K, "
            f"Rk={R_mean:.4f}±{R_sem:.4f} m²K/GW, "
            f"k_SiC={k_sic_mean:.2f}±{k_sic_sem:.2f}, "
            f"k_Ga2O3={k_ga_mean:.2f}±{k_ga_sem:.2f} W/mK"
        )




    if not Lxs:
        print("No valid data parsed.")
        return

    # convert to numpy & sort by Lx
    Lxs = np.array(Lxs)
    G_means = np.array(G_means)
    G_sems = np.array(G_sems)
    k_sic_means = np.array(k_sic_means)
    k_sic_sems = np.array(k_sic_sems)
    k_ga_means = np.array(k_ga_means)
    k_ga_sems = np.array(k_ga_sems)
    labels = np.array(labels)

    R_means = np.array(R_means)
    R_sems  = np.array(R_sems)
    invLs   = np.array(invLs)

    order = np.argsort(Lxs)
    Lxs = Lxs[order]
    G_means = G_means[order]
    k_sic_means = k_sic_means[order]
    k_ga_means = k_ga_means[order]
    G_sems = G_sems[order]
    k_sic_sems = k_sic_sems[order]
    k_ga_sems = k_ga_sems[order]
    labels = labels[order]
        # --- REVISED PART D: in the sorting block, include R arrays ---
    R_means = R_means[order]
    R_sems  = R_sems[order]
    invLs   = invLs[order]

    # ---------- Plot 1: TBC vs Lx ----------
    plt.figure(figsize=(7, 5))
    plt.errorbar(Lxs, G_means, yerr=G_sems, fmt="o-", capsize=4)
    plt.xlabel("Box length Lx (Å)")
    plt.ylabel("TBC G (MW/(m²·K))")
    plt.title("TBC vs Lx")
    # optional: annotate each point with folder name
    # for x, y, lab in zip(Lxs, G_means, labels):
    #     plt.text(x, y, lab, fontsize=8, ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(f"TBC_vs_Lx.png", dpi=300)
    plt.close()
    print("Saved TBC_vs_Lx.png")

    # ---------- Plot 2: k_SiC & k_Ga2O3 vs Lx (dual y-axis) ----------
    fig, ax1 = plt.subplots(figsize=(7, 5))

    color_sic = "tab:blue"
    color_ga  = "tab:red"

    # Left axis: k_SiC
    ax1.errorbar(Lxs, k_sic_means, yerr=k_sic_sems, fmt="o-", capsize=4,
                color=color_sic, label="k_SiC")
    ax1.set_xlabel("Box length Lx (Å)")
    ax1.set_ylabel("k_SiC (W/m·K)", color=color_sic)
    ax1.tick_params(axis="y", labelcolor=color_sic)

    # Right axis: k_Ga2O3
    ax2 = ax1.twinx()
    ax2.errorbar(Lxs, k_ga_means, yerr=k_ga_sems, fmt="s--", capsize=4,
                color=color_ga, label="k_Ga2O3")
    ax2.set_ylabel("k_Ga2O3 (W/m·K)", color=color_ga)
    ax2.tick_params(axis="y", labelcolor=color_ga)

    # Combine legends (colors will appear correctly)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Thermal Conductivity vs Lx")
    fig.tight_layout()
    plt.savefig("k_vs_Lx.png", dpi=300)
    plt.close()
    print("Saved k_vs_Lx.png")






    # --- REVISED PART 3: NEW Plot 3: R_K vs 1/Lx ---
    invL = 1.0 / Lxs   # 1/Å

    plt.figure(figsize=(7, 5))
    plt.errorbar(invLs, R_means, yerr=R_sems, fmt="o-", capsize=4)
    plt.xlabel("1 / Lx (1/Å)")
    plt.ylabel("TBR $R_K$ (m$^2$·K/GW)")
    plt.title("TBR $R_K$ vs 1/Lx")

    for x, y, lab in zip(invL, R_means, labels):
        plt.text(x, y, lab, fontsize=8, ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("RK_vs_invLx.png", dpi=300)
    plt.close()
    print("Saved RK_vs_invLx.png")

    log_print("\n--- Sorted Data (for fitting R vs 1/L) ---")
    for Lx, invL, Rm, Rs in zip(Lxs, invLs, R_means, R_sems):
        log_print(f"Lx={Lx:.3f} Å, 1/Lx={invL:.6e}, Rk={Rm:.5f} ± {Rs:.5f}")

    log_print("\nAnalysis complete.")
    log.close()
    print(f"Log saved to {log_name}")





if __name__ == "__main__":
    main()
