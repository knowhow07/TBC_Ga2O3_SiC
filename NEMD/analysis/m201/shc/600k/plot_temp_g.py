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


def read_group_xranges_from_model_xyz(fname):
    """
    Read x positions and group IDs from GPUMD model.xyz.
    Assumes columns are like:
    id species x y z ... group
    i.e. x is col 2 and group is the last column.
    """
    with open(fname, "r") as f:
        N = int(f.readline())
        _ = f.readline()

        xs = []
        groups = []
        species = []

        for _ in range(N):
            parts = f.readline().split()
            xs.append(float(parts[2]))       # x
            species.append(parts[1])         # species
            groups.append(int(parts[-1]))    # group ID in last column

    xs = np.array(xs)
    groups = np.array(groups)
    species = np.array(species)

    # print species span for checking
    for sp in ["Si", "C", "Ga", "O"]:
        mask = (species == sp)
        if np.any(mask):
            print(sp, xs[mask].min(), xs[mask].max())

    group_ranges = {}
    for g in np.unique(groups):
        gx = xs[groups == g]
        group_ranges[g] = {
            "xmin": np.min(gx),
            "xmax": np.max(gx),
            "xmid": 0.5 * (np.min(gx) + np.max(gx)),
        }

    return group_ranges

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

        # read spatial ranges of each group from model.xyz
        group_ranges = read_group_xranges_from_model_xyz(mf)

        # plot group 2 as a full-bar reference line,
        # but connect the other groups in x order
        x_conn = []
        T_conn = []
        plotted_label_main = False
        plotted_label_g2 = False

        for g in range(M):
            if g not in group_ranges:
                continue
            if np.isnan(T_avg[g]):
                continue
            if drop_wall and g == 0:
                continue

            xmid = group_ranges[g]["xmid"]
            Tg = T_avg[g]

            if g == 2:
                if not plotted_label_g2:
                    plt.hlines(
                        y=Tg, xmin=0.0, xmax=Lx_ang,
                        linestyles="--", linewidth=1.3,
                        colors="gray", alpha=0.8,
                        label=f"{jd} : group 2 ref"
                    )
                    plotted_label_g2 = True
                else:
                    plt.hlines(
                        y=Tg, xmin=0.0, xmax=Lx_ang,
                        linestyles="--", linewidth=1.3,
                        colors="gray", alpha=0.8
                    )
            else:
                x_conn.append(xmid)
                T_conn.append(Tg)

        # connect all groups except group 2
        if x_conn:
            order = np.argsort(x_conn)
            x_conn = np.array(x_conn)[order]
            T_conn = np.array(T_conn)[order]

            if not plotted_label_main:
                plt.plot(x_conn, T_conn, "o-", linewidth=1.5, markersize=4, label=jd)
                plotted_label_main = True
            else:
                plt.plot(x_conn, T_conn, "o-", linewidth=1.5, markersize=4)

    plt.xlabel("Position along x (Å)")
    plt.ylabel("Temperature (K)")
    plt.title(
        f"Temperature profile from {compute_file}\n"
        f"(groups 1→6 transport, group 2 shown as reference line)"
    )
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    outname = f"{compute_file}_profile_all_jobs.png"
    plt.savefig(outname, dpi=300)
    plt.close()

    print(f"Saved plot to {outname}")


if __name__ == "__main__":
    main()
