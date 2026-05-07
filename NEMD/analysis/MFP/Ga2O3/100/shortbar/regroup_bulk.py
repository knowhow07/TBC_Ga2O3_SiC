#!/usr/bin/env python3
"""
Add GPUMD group IDs to a pure bulk bar from extended XYZ
========================================================

Input:
  extended xyz, e.g.
    6720
    Lattice="..." Properties=id:I:1:species:S:1:pos:R:3
    21 Si 11.0882 3.0757 33.7393
    ...

Group definition along x:
  group 0 : fixed atoms at both ends
  group 1 : hot slab (left side)
  group 2 : left dump region
  group 3 : center slab (centered at middle of atomic x positions, user-given width)
  group 4 : right dump region
  group 5 : cold slab (right side)

Layout along x:
  [fixL][hot][left dump][center][right dump][cold][fixR]

Output:
  - model.xyz
  - group_regions.png
  - logs/...
"""

import os
import re
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# User knobs
# ----------------------------------------------------------------------
INPUT_XYZ = "rotated_longx_4+1_Ga2O3.xyz"

# widths in Å along x
FIX_WIDTH    = 5.0
HOT_WIDTH    = 10.0
CENTER_WIDTH = 10.0
COLD_WIDTH   = 10.0

OUT_XYZ = "model.xyz"
OUT_PNG = "group_regions.png"

# shift x so min(x)=0 in output/plot
SHIFT_X_TO_ZERO = True

# set None to plot all atoms
PLOT_MAX_POINTS = None


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
LOG_NAME = f"bulk_bar_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs("logs", exist_ok=True)
LOG_PATH = os.path.join("logs", LOG_NAME)

_original_stdout = sys.stdout


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


log_file = open(LOG_PATH, "w")
sys.stdout = Tee(_original_stdout, log_file)

print(f"\n=== Logging to {LOG_PATH} ===\n")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_lattice_from_comment(comment_line):
    """
    Parse Lattice="a1 a2 ... a9" from extended XYZ comment line.
    Return 3x3 lattice matrix as np.ndarray, or None if not found.
    """
    m = re.search(r'Lattice="([^"]+)"', comment_line)
    if not m:
        return None
    vals = [float(x) for x in m.group(1).split()]
    if len(vals) != 9:
        raise ValueError("Lattice field found, but does not contain 9 numbers.")
    return np.array(vals, dtype=float).reshape(3, 3)


def read_extended_xyz(filename):
    """
    Read extended xyz with formats like:
      id species x y z
    or
      species x y z

    Returns:
      species: list[str]
      coords : (N,3) float array
      lattice: (3,3) float array or None
      comment: original 2nd line
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError("XYZ file is too short.")

    natoms = int(lines[0].strip())
    comment = lines[1].rstrip("\n")
    lattice = parse_lattice_from_comment(comment)

    atom_lines = lines[2:2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"Expected {natoms} atom lines, got {len(atom_lines)}.")

    species = []
    coords = []

    for i, line in enumerate(atom_lines):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Atom line {i+3} has too few columns: {line}")

        # supported:
        #   species x y z
        #   id species x y z
        if len(parts) >= 5:
            # assume: id species x y z ...
            sp = parts[1]
            x, y, z = map(float, parts[2:5])
        else:
            # assume: species x y z
            sp = parts[0]
            x, y, z = map(float, parts[1:4])

        species.append(sp)
        coords.append([x, y, z])

    coords = np.array(coords, dtype=float)

    print(f"Loaded structure from: {filename}")
    print(f"Number of atoms: {natoms}")
    if lattice is not None:
        print("Parsed lattice from XYZ header:")
        print(lattice)
    else:
        print("No lattice found in XYZ header.")

    return species, coords, lattice, comment


def compute_region_bounds(xmin, xmax, fix_w, hot_w, center_w, cold_w):
    """
    Layout:
      [fixL][hot][left dump][center][right dump][cold][fixR]
    """
    xmid = 0.5 * (xmin + xmax)

    fixL_end = xmin + fix_w
    hot_end = fixL_end + hot_w

    fixR_beg = xmax - fix_w
    cold_beg = fixR_beg - cold_w

    center_beg = xmid - 0.5 * center_w
    center_end = xmid + 0.5 * center_w

    if hot_end > center_beg:
        raise ValueError(
            "HOT region overlaps CENTER region. Reduce HOT_WIDTH / FIX_WIDTH / CENTER_WIDTH."
        )
    if center_end > cold_beg:
        raise ValueError(
            "CENTER region overlaps COLD region. Reduce COLD_WIDTH / FIX_WIDTH / CENTER_WIDTH."
        )

    return {
        "xmin": xmin,
        "xmax": xmax,
        "xmid": xmid,
        "fixL_end": fixL_end,
        "hot_end": hot_end,
        "center_beg": center_beg,
        "center_end": center_end,
        "cold_beg": cold_beg,
        "fixR_beg": fixR_beg,
    }


def assign_groups(coords, bounds):
    """
    Assign group IDs:
      0 = fixed ends
      1 = hot
      2 = left dump
      3 = center
      4 = right dump
      5 = cold
    """
    x = coords[:, 0]

    fixL_end = bounds["fixL_end"]
    hot_end = bounds["hot_end"]
    center_beg = bounds["center_beg"]
    center_end = bounds["center_end"]
    cold_beg = bounds["cold_beg"]
    fixR_beg = bounds["fixR_beg"]

    groups = np.full(len(coords), -1, dtype=int)

    for i, xi in enumerate(x):
        if xi <= fixL_end or xi >= fixR_beg:
            groups[i] = 0
        elif fixL_end < xi <= hot_end:
            groups[i] = 1
        elif hot_end < xi < center_beg:
            groups[i] = 2
        elif center_beg <= xi <= center_end:
            groups[i] = 3
        elif center_end < xi < cold_beg:
            groups[i] = 4
        elif cold_beg <= xi < fixR_beg:
            groups[i] = 5
        else:
            raise RuntimeError(f"Atom {i} at x={xi:.6f} Å could not be assigned.")

    return groups


def print_region_summary(bounds):
    print("Region bounds along x (Å):")
    print(f"  xmin       : {bounds['xmin']:.6f}")
    print(f"  fixL_end   : {bounds['fixL_end']:.6f}   -> end of left fixed (group 0)")
    print(f"  hot_end    : {bounds['hot_end']:.6f}   -> end of hot (group 1)")
    print(f"  center_beg : {bounds['center_beg']:.6f}   -> start of center (group 3)")
    print(f"  xmid       : {bounds['xmid']:.6f}   -> center of group 3")
    print(f"  center_end : {bounds['center_end']:.6f}   -> end of center (group 3)")
    print(f"  cold_beg   : {bounds['cold_beg']:.6f}   -> start of cold (group 5)")
    print(f"  fixR_beg   : {bounds['fixR_beg']:.6f}   -> start of right fixed (group 0)")
    print(f"  xmax       : {bounds['xmax']:.6f}")


def print_group_counts(groups):
    unique, counts = np.unique(groups, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    print("\nGroup counts:")
    for g in range(6):
        print(f"  group {g}: {count_map.get(g, 0)}")


def build_output_comment(lattice):
    """
    GPUMD-style comment line.
    """
    if lattice is not None:
        lat_txt = " ".join(f"{v:.10f}" for v in lattice.reshape(-1))
        return f'Lattice="{lat_txt}" Properties=species:S:1:pos:R:3:group:I:1 pbc="T T T"'
    return 'Properties=species:S:1:pos:R:3:group:I:1'


def write_model_xyz(species, coords, groups, lattice, filename, shift_x_to_zero=True):
    """
    Write model.xyz for GPUMD:
      species x y z group
    """
    coords_out = coords.copy()
    if shift_x_to_zero:
        coords_out[:, 0] -= coords_out[:, 0].min()

    comment = build_output_comment(lattice)

    with open(filename, "w") as f:
        f.write(f"{len(species)}\n")
        f.write(comment + "\n")
        for sp, xyz, grp in zip(species, coords_out, groups):
            f.write(
                f"{sp:2s}  {xyz[0]:16.8f} {xyz[1]:16.8f} {xyz[2]:16.8f}  {grp:d}\n"
            )

    print(f"\nWrote GPUMD model XYZ to: {filename}")


def plot_groups_png(coords, groups, bounds, filename, shift_x_to_zero=True, max_points=None):
    """
    Plot x-y scatter colored by group, with boundary lines.
    """
    coords_plot = coords.copy()
    bounds_plot = dict(bounds)

    if shift_x_to_zero:
        x0 = coords_plot[:, 0].min()
        coords_plot[:, 0] -= x0
        for k in bounds_plot:
            bounds_plot[k] -= x0

    x = coords_plot[:, 0]
    y = coords_plot[:, 1]

    if max_points is not None and len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
        x = x[idx]
        y = y[idx]
        groups_plot = groups[idx]
    else:
        groups_plot = groups

    plt.figure(figsize=(12, 4.5))

    for g in range(6):
        mask = (groups_plot == g)
        if np.any(mask):
            plt.scatter(x[mask], y[mask], s=8, alpha=0.8, label=f"group {g}")

    for key in ["fixL_end", "hot_end", "center_beg", "center_end", "cold_beg", "fixR_beg"]:
        plt.axvline(bounds_plot[key], linewidth=1)

    plt.xlabel("x (Å)")
    plt.ylabel("y (Å)")
    plt.title("Bulk bar groups")
    plt.legend(ncol=6, fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Wrote group plot PNG to: {filename}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    species, coords, lattice, comment = read_extended_xyz(INPUT_XYZ)

    x = coords[:, 0]
    xmin, xmax = x.min(), x.max()
    lx = xmax - xmin

    print(f"x range: [{xmin:.6f}, {xmax:.6f}]  (Lx = {lx:.6f} Å)")

    bounds = compute_region_bounds(
        xmin=xmin,
        xmax=xmax,
        fix_w=FIX_WIDTH,
        hot_w=HOT_WIDTH,
        center_w=CENTER_WIDTH,
        cold_w=COLD_WIDTH,
    )

    print_region_summary(bounds)

    groups = assign_groups(coords, bounds)
    print_group_counts(groups)

    write_model_xyz(
        species=species,
        coords=coords,
        groups=groups,
        lattice=lattice,
        filename=OUT_XYZ,
        shift_x_to_zero=SHIFT_X_TO_ZERO,
    )

    plot_groups_png(
        coords=coords,
        groups=groups,
        bounds=bounds,
        filename=OUT_PNG,
        shift_x_to_zero=SHIFT_X_TO_ZERO,
        max_points=PLOT_MAX_POINTS,
    )

    print("\n✅ model.xyz written successfully.")
    print("Group meaning:")
    print("  0 = fixed ends")
    print("  1 = hot")
    print("  2 = left dump")
    print("  3 = center")
    print("  4 = right dump")
    print("  5 = cold")