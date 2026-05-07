#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import os

# ----------------------------------------------------------------------
# User knobs
# ----------------------------------------------------------------------
compute_file = "compute.out"
model_file   = "model.xyz"
skip_fraction = 0.2   # use last 50% of data as steady state
drop_wall = True      # drop group 0 (frozen wall)
# ----------------------------------------------------------------------


fname = "model.xyz"
with open(fname) as f:
    N = int(f.readline())
    _ = f.readline()

    xs = []
    species = []

    for i in range(N):
        parts = f.readline().split()
        xs.append(float(parts[2]))        # x is 3rd column
        species.append(parts[1])          # Ga O Si C etc.

    xs = np.array(xs)
    species = np.array(species)

    for sp in ["Si","C","Ga","O"]:
        print(sp, xs[species==sp].min(), xs[species==sp].max())

def read_box_from_model_xyz(fname):
    """
    Read Lx, Ly, Lz (Å) from the second line of a GPUMD model.xyz file.
    Assumes Lattice="a1x a1y a1z a2x a2y a2z a3x a3y a3z".
    """
    with open(fname, "r") as f:
        _ = f.readline()          # number of atoms
        header = f.readline().strip()

    # Extract the Lattice="..." part
    m = re.search(r'Lattice="([^"]+)"', header)
    if m is None:
        raise RuntimeError(f"Cannot find Lattice in header of {fname}")

    lat_str = m.group(1)
    lat_vals = np.fromstring(lat_str, sep=" ")
    if lat_vals.size != 9:
        raise RuntimeError("Expected 9 lattice numbers in Lattice=...")

    a1 = lat_vals[0:3]
    a2 = lat_vals[3:6]
    a3 = lat_vals[6:9]

    Lx = np.linalg.norm(a1)
    Ly = np.linalg.norm(a2)
    Lz = np.linalg.norm(a3)
    return Lx, Ly, Lz


def main():
    # --- find all job_* directories ---
    job_dirs = sorted(
        d for d in os.listdir(".")
        if os.path.isdir(d) and d.startswith("shc_")
    )
    if not job_dirs:
        raise RuntimeError("No job_* folders found")

    plt.figure(figsize=(7, 4))

    for jd in job_dirs:
        cf = os.path.join(jd, compute_file)
        mf = os.path.join(jd, model_file)

        if not (os.path.exists(cf) and os.path.exists(mf)):
            print(f"[SKIP] {jd}: missing {compute_file} or {model_file}")
            continue

        # --- read box from model.xyz for this job ---
        Lx_ang, Ly_ang, Lz_ang = read_box_from_model_xyz(mf)
        print(f"[{jd}] Lx={Lx_ang:.3f} Å, Ly={Ly_ang:.3f} Å, Lz={Lz_ang:.3f} Å")

        # --- load compute.out ---
        data = np.loadtxt(cf)

        # Ensure 2D (handles single-row or single-column files)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        nrows, ncols = data.shape

        if ncols < 3 or nrows < 2:
            print(f"[SKIP] {jd}: compute.out has invalid shape {data.shape}")
            continue

        M = ncols - 2      # number of temperature groups/bins
        T_all = data[:, :M]

        # steady-state rows
        start_idx = int(math.floor(nrows * skip_fraction))
        if start_idx >= nrows:
            start_idx = nrows - 1  # at least keep one row

        T_ss = T_all[start_idx:, :]

        # --- average over time, avoiding 'mean of empty slice' ---
        T_avg = np.full(M, np.nan)
        valid_cols = ~np.all(np.isnan(T_ss), axis=0)
        if np.any(valid_cols):
            T_avg[valid_cols] = np.nanmean(T_ss[:, valid_cols], axis=0)

        # bin centers along x (Å)
        x_bins = (np.arange(M) + 0.5) * Lx_ang / M

        # initial mask: drop bins with NaN average
        mask = ~np.isnan(T_avg)

        # drop frozen wall (group 0) if requested
        if drop_wall and M > 0:
            mask[0] = False

        x_plot = x_bins[mask]
        T_plot = T_avg[mask]

        plt.plot(x_plot, T_plot, "o-", linewidth=1.2, label=jd)

    plt.xlabel("Position along heat direction (Å)")
    plt.ylabel("Temperature (K)")
    plt.title(f"Temperature Profile from {compute_file}\n(steady-state average)")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    outname = f"{compute_file}_profile_all_jobs.png"
    plt.savefig(outname, dpi=300)
    plt.close()

    print(f"Saved plot to {outname}")


if __name__ == "__main__":
    main()
