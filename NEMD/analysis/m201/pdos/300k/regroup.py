#!/usr/bin/env python3
"""
Regroup an existing GPUMD extended-XYZ model.xyz for a PDOS-focused run.

You asked for ONLY 3 analysis groups (+walls):
  - group 0 : fixed walls (both ends)
  - group 1 : left reference region (bulk-like)
  - group 2 : WIDE interface analysis region (adjustable width at top)
  - group 3 : right reference region (bulk-like)

Key design (works for Lx ~ 200–800 Å):
- Interface region (group 2) is centered at the box center (x = Lx/2 by default),
  with width = INTERFACE_W (Å).
- Left (group 1) and right (group 3) are "windows" adjacent to group 2,
  each with width = SIDE_W (Å).
- Any remaining space outside [group1, group2, group3] is assigned to walls (group 0),
  symmetrically, so you don't have to tune for different Lx.

Outputs:
- Backs up the original model.xyz -> model.xyz.bak_TIMESTAMP
- Writes the regrouped file as model.xyz
- Saves a plot: group_ranges_x.png (x vs group ID)

----------------------------------------------------------------
LOCATION TO REVISE:
- Edit the "USER VARIABLES" block below.
----------------------------------------------------------------
"""

import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER VARIABLES (EDIT HERE)
# ============================================================
PATH = "."                 # folder containing xyz files
INFILE = "model.xyz"       # primary input name (the current working file)
ORIGINAL = "model_original.xyz"  # preserved original name (created if missing)
OUTFILE = "model.xyz"      # output regrouped model.xyz
PLOT_FILE = "group_ranges_x.png"


# Width of the wide interface analysis region (Å)  <-- you asked to define this
INTERFACE_W = 100.0

# Width of left/right reference regions (Å)
# (Leave as-is unless you want thinner/thicker bulk reference for PDOS cutoff.)
SIDE_W = 30.0

# Base wall thickness (Å) at each end.
# Extra unused length (if any) is absorbed into walls symmetrically.
WALL_THICK_BASE = 10.0

# If True: interface center is at Lx/2 (recommended)
# If False: interface center uses midpoint of atom x-extent (rarely needed)
USE_BOX_CENTER = True

EPS = 1e-9
# ============================================================


def parse_xyz_line_with_id(line: str):
    """
    Accepts:
      '1 Ga x y z'
      '1 Ga x y z g'
      'Ga x y z'
      'Ga x y z g'
    Returns (id_or_None, species, x, y, z).
    """
    parts = line.split()
    if not parts:
        return None, None, None, None, None

    # try id-first
    try:
        atom_id = int(parts[0])
        species = parts[1]
        x, y, z = map(float, parts[2:5])
        return atom_id, species, x, y, z
    except Exception:
        species = parts[0]
        x, y, z = map(float, parts[1:4])
        return None, species, x, y, z


def ensure_group_in_properties_line(line2: str) -> str:
    """
    Ensure second line includes:
      properties=id:I:1:species:S:1:pos:R:3:group:I:1
    If properties exists, replace it. If not, append it.
    """
    lower = line2.lower()
    key = "properties="
    if key in lower:
        idx = lower.index(key)
        prefix = line2[:idx]
        return prefix + "properties=id:I:1:species:S:1:pos:R:3:group:I:1"
    else:
        if not line2.endswith(" "):
            line2 += " "
        return line2 + "properties=id:I:1:species:S:1:pos:R:3:group:I:1"


def get_Lx_from_lattice(header2: str) -> float:
    """
    Parse Lattice="a1x a1y a1z a2x ... a3z" and return a1x as Lx.
    This assumes your long direction is x and cell is aligned such that a1 is along x.
    """
    m = re.search(r'Lattice="([^"]+)"', header2)
    if m is None:
        raise RuntimeError('Cannot find Lattice="..." in the 2nd line of model.xyz.')
    lat = list(map(float, m.group(1).split()))
    if len(lat) < 9:
        raise RuntimeError("Lattice field does not contain 9 numbers.")
    return float(lat[0])


def plot_group_ranges(x: np.ndarray, groups: np.ndarray, bounds: dict, out_png: str):
    """
    Plot horizontal bars showing x-range per group.
    Split walls (group 0) into left and right pieces.
    """
    plt.figure(figsize=(9, 6))

    # walls split bars
    wallL = bounds["wallL"]
    wallR = bounds["wallR"]
    plt.hlines(y=0, xmin=wallL[0], xmax=wallL[1], linewidth=4)
    plt.hlines(y=0, xmin=wallR[0], xmax=wallR[1], linewidth=4)

    for gid in [1, 2, 3]:
        mask = (groups == gid)
        if not np.any(mask):
            continue
        xmin = x[mask].min()
        xmax = x[mask].max()
        plt.hlines(y=gid, xmin=xmin, xmax=xmax, linewidth=4)

    plt.xlabel("Position along x (Å)")
    plt.ylabel("Group ID")
    plt.title("Group ranges along x-direction (PDOS regroup)")
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    infile = os.path.join(PATH, INFILE)
    outfile = os.path.join(PATH, OUTFILE)
    plot_png = os.path.join(PATH, PLOT_FILE)

    if not os.path.isfile(infile):
        raise FileNotFoundError(f"Input not found: {infile}")

    # ------------------------------------------------------------
    # Backup/read logic:
    # - If ORIGINAL does NOT exist: move INFILE -> ORIGINAL
    # - If ORIGINAL exists: keep it
    # - Always read from ORIGINAL
    # - Always write regrouped output to OUTFILE (typically model.xyz)
    # ------------------------------------------------------------
    infile = os.path.join(PATH, INFILE)
    original = os.path.join(PATH, ORIGINAL)

    if not os.path.exists(original):
        if not os.path.exists(infile):
            raise FileNotFoundError(f"Neither {infile} nor {original} exists.")
        os.rename(infile, original)
        print(f"Renamed {infile} -> {original} (saved original)")

    # Always read from ORIGINAL
    with open(original, "r") as f:
        header1 = f.readline()
        header2 = f.readline().rstrip("\n")
        lines = f.readlines()


    N = int(header1.strip())

    ids = []
    species = []
    coords = []

    idx_assigned = 0
    for line in lines[:N]:
        line = line.strip()
        if not line:
            continue
        atom_id, sp, xi, yi, zi = parse_xyz_line_with_id(line)
        if atom_id is None:
            atom_id = idx_assigned + 1
        idx_assigned += 1
        ids.append(atom_id)
        species.append(sp)
        coords.append([xi, yi, zi])

    coords = np.array(coords, dtype=float)
    x = coords[:, 0]

    # Use Lx from Lattice (robust) and define box [0, Lx]
    Lx = get_Lx_from_lattice(header2)
    xlo_box, xhi_box = 0.0, Lx

    # Choose interface center
    if USE_BOX_CENTER:
        x_center = 0.5 * (xlo_box + xhi_box)
    else:
        x_center = 0.5 * (float(x.min()) + float(x.max()))

    # Required length: two base walls + left + interface + right
    core_len = SIDE_W + INTERFACE_W + SIDE_W
    min_needed = 2.0 * WALL_THICK_BASE + core_len

    print(f"Loaded {N} atoms from {original}")
    print(f"Box Lx (from Lattice) = {Lx:.3f} Å")
    print(f"Interface center x = {x_center:.3f} Å")
    print(f"Requested widths: left={SIDE_W:.3f}, interface={INTERFACE_W:.3f}, right={SIDE_W:.3f} (Å)")
    print(f"WALL_THICK_BASE={WALL_THICK_BASE:.3f} Å each end -> min_needed={min_needed:.3f} Å")

    if Lx + 1e-6 < min_needed:
        raise RuntimeError(
            f"Box too short: Lx={Lx:.3f} Å < min_needed={min_needed:.3f} Å.\n"
            f"Increase Lx or reduce INTERFACE_W/SIDE_W/WALL_THICK_BASE."
        )

    # Extra length goes into walls symmetrically
    extra = Lx - min_needed
    wall_thick = WALL_THICK_BASE + 0.5 * extra

    # Layout:
    # wall0 | left(1) | interface(2) | right(3) | wall0
    x_fixL_hi = xlo_box + wall_thick
    x_fixR_lo = xhi_box - wall_thick

    # We anchor the left/interface/right block within (x_fixL_hi, x_fixR_lo)
    # and center the interface at x_center (usually Lx/2).
    half_int = 0.5 * INTERFACE_W
    int_lo = x_center - half_int
    int_hi = x_center + half_int

    left_lo = int_lo - SIDE_W
    left_hi = int_lo

    right_lo = int_hi
    right_hi = int_hi + SIDE_W

    # Sanity: left/right blocks must lie within the non-wall interior
    # If not, we shift the entire block to fit.
    interior_lo = x_fixL_hi
    interior_hi = x_fixR_lo

    block_lo = left_lo
    block_hi = right_hi

    if block_lo < interior_lo - 1e-6:
        shift = interior_lo - block_lo
        left_lo += shift; left_hi += shift
        int_lo += shift; int_hi += shift
        right_lo += shift; right_hi += shift
        block_lo += shift; block_hi += shift

    if block_hi > interior_hi + 1e-6:
        shift = interior_hi - block_hi
        left_lo += shift; left_hi += shift
        int_lo += shift; int_hi += shift
        right_lo += shift; right_hi += shift
        block_lo += shift; block_hi += shift

    # Final check
    if block_lo < interior_lo - 1e-6 or block_hi > interior_hi + 1e-6:
        raise RuntimeError(
            "Cannot place left/interface/right blocks within interior after shifting.\n"
            "Try reducing INTERFACE_W and/or SIDE_W, or increasing Lx."
        )

    def in_range(xx, lo, hi, include_hi=False):
        if include_hi:
            return (xx >= lo - EPS) & (xx <= hi + EPS)
        return (xx >= lo - EPS) & (xx < hi - EPS)

    groups = np.zeros(N, dtype=int)

    # Assign walls (group 0) by x position relative to interior
    # IMPORTANT: atoms could have x outside [0,Lx] depending on wrapping;
    # for most GPUMD models x should already be inside. If not, wrap is needed.
    groups[in_range(x, x_fixL_hi, x_fixR_lo)] = -1  # mark interior temporarily

    # Default all to wall 0
    groups[:] = 0

    # Assign left/interface/right
    groups[in_range(x, left_lo, left_hi)] = 1
    groups[in_range(x, int_lo, int_hi)] = 2
    groups[in_range(x, right_lo, right_hi)] = 3

    # Everything else stays as 0 (walls/unused buffer)

    # Update properties line
    header2_out = ensure_group_in_properties_line(header2)

    # Write regrouped model.xyz
    with open(outfile, "w") as f:
        f.write(f"{N}\n")
        f.write(header2_out + "\n")
        for atom_id, sp, (xi, yi, zi), g in zip(ids, species, coords, groups):
            f.write(f"{atom_id:d} {sp} {xi:.8f} {yi:.8f} {zi:.8f} {int(g):d}\n")

    print(f"\n✔ Wrote regrouped PDOS model to: {outfile}")

    unique, counts = np.unique(groups, return_counts=True)
    print("\nGroup counts:")
    for gid, c in zip(unique, counts):
        print(f"  group {gid:2d}: {c}")

    bounds = {
        "wallL": (xlo_box, x_fixL_hi),
        "g1_left": (left_lo, left_hi),
        "g2_interface": (int_lo, int_hi),
        "g3_right": (right_lo, right_hi),
        "wallR": (x_fixR_lo, xhi_box),
    }

    print("\nRegion bounds along x (Å):")
    for name, (lo, hi) in bounds.items():
        print(f"  {name:>12s}: {lo:.3f} -> {hi:.3f}")

    plot_group_ranges(x, groups, bounds, plot_png)
    print(f"\n✔ Saved plot: {plot_png}")


if __name__ == "__main__":
    main()
