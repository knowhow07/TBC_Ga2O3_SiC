#!/usr/bin/env python3
"""
Regroup an existing GPUMD extended-XYZ model.xyz for a PDOS-focused run.

You asked for ONLY 3 analysis groups (+walls):
0 = wall
1 = hot slab (left)
2 = dumb slab (all remaining interior sections / gaps)
3 = left SHC slab
4 = interface-core SHC slab
5 = right SHC slab
6 = cold slab (right)

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


# --------- region widths you want (Å) ----------


# --------- NEMD slab widths (Å) ----------
wall_thick  = 10.0    # group 0
hot_thick   = 10.0   # group 1
cold_thick  = 10.0    # group 6

left_w      = 10.0   # group 3, auto-centered in left side
int_core_w  = 10.0   # group 4, centered at interface
right_w     = 10.0   # group 5, auto-centered in right side

EPS = 1e-9

LOG_FILE = "regroup_log.txt"   # or f"regroup_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
# ============================================================

import sys

class Tee:
    """Duplicate prints to both terminal and a log file."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

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

    for gid in [1, 2, 3, 4, 5, 6]:
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

    log_path = os.path.join(PATH, LOG_FILE)
    log_fh = open(log_path, "w", buffering=1)  # line-buffered

    # tee all prints to terminal + log
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.stdout, log_fh)
    sys.stderr = Tee(sys.stderr, log_fh)

    print(f"Logging to: {log_path}")

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

    # ------------------------------------------------------------
    # Choose interface center from chemistry: Si side -> O side
    # Define interface using the last Si layer and first O layer along +x
    # Output:
    #   - avg_x_Si_last_layer
    #   - avg_x_O_first_layer
    #   - interface_center = 0.5*(avg_Si + avg_O)
    # ------------------------------------------------------------

    # thickness (Å) to define a "layer" window near the boundary
    LAYER_W = 2.0  # adjust if your layers are more spread (e.g., 3–5 Å)

    xSi = x[np.array([sp == "Si" for sp in species], dtype=bool)]
    xO  = x[np.array([sp == "O"  for sp in species], dtype=bool)]

    if xSi.size == 0 or xO.size == 0:
        raise RuntimeError("Cannot define interface center: missing Si or O atoms in species list.")

    # crude boundary markers
    xSi_max = float(xSi.max())   # furthest Si toward +x
    xO_min  = float(xO.min())    # closest O toward -x

    # "last Si layer": Si atoms within [xSi_max - LAYER_W, xSi_max]
    mask_Si_layer = (np.array([sp == "Si" for sp in species], dtype=bool)) & (x >= xSi_max - LAYER_W) & (x <= xSi_max + EPS)
    # "first O layer": O atoms within [xO_min, xO_min + LAYER_W]
    mask_O_layer  = (np.array([sp == "O"  for sp in species], dtype=bool)) & (x >= xO_min - EPS) & (x <= xO_min + LAYER_W)

    if not np.any(mask_Si_layer):
        raise RuntimeError(f"No Si atoms found in the last-layer window. Try increasing LAYER_W (currently {LAYER_W} Å).")
    if not np.any(mask_O_layer):
        raise RuntimeError(f"No O atoms found in the first-layer window. Try increasing LAYER_W (currently {LAYER_W} Å).")

    xSi_avg = float(x[mask_Si_layer].mean())
    xO_avg  = float(x[mask_O_layer].mean())

    x_center = 0.5 * (xSi_avg + xO_avg)

    print(f"Interface center from chemistry:")
    print(f"  last Si layer avg x = {xSi_avg:.6f} Å  (Si max x = {xSi_max:.6f} Å, window={LAYER_W} Å)")
    print(f"  first O layer avg x = {xO_avg:.6f} Å  (O  min x = {xO_min:.6f} Å, window={LAYER_W} Å)")
    print(f"  interface center x  = {x_center:.6f} Å")

    # ------------------------------------------------------------
    # New layout:
    # wall | hot | dumb(2) | left(3) | dumb(2) | interface(4) |
    # dumb(2) | right(5) | dumb(2) | cold(6) | wall
    #
    # group 2 is reused for all leftover interior gap sections.
    # group 3 and 5 are automatically centered in their respective sides.
    # ------------------------------------------------------------

    interior_lo = xlo_box + wall_thick
    interior_hi = xhi_box - wall_thick

    # ---- fixed end slabs ----
    g1_lo = interior_lo
    g1_hi = g1_lo + hot_thick

    g6_hi = interior_hi
    g6_lo = g6_hi - cold_thick

    # ---- interface-core centered at chemistry-based interface center ----
    half_core = 0.5 * int_core_w
    g4_lo = x_center - half_core
    g4_hi = x_center + half_core

    # ---- free side regions where left/right SHC slabs will be centered ----
    left_free_lo  = g1_hi
    left_free_hi  = g4_lo

    right_free_lo = g4_hi
    right_free_hi = g6_lo

    left_free_w  = left_free_hi  - left_free_lo
    right_free_w = right_free_hi - right_free_lo

    if left_free_w <= 0:
        raise RuntimeError("No space between hot slab and interface-core.")
    if right_free_w <= 0:
        raise RuntimeError("No space between interface-core and cold slab.")

    if left_w > left_free_w:
        raise RuntimeError(
            f"left_w={left_w} Å is larger than available left-side width={left_free_w:.3f} Å"
        )
    if right_w > right_free_w:
        raise RuntimeError(
            f"right_w={right_w} Å is larger than available right-side width={right_free_w:.3f} Å"
        )

    # ---- group 3 centered in left side ----
    left_mid = 0.5 * (left_free_lo + left_free_hi)
    g3_lo = left_mid - 0.5 * left_w
    g3_hi = left_mid + 0.5 * left_w

    # ---- group 5 centered in right side ----
    right_mid = 0.5 * (right_free_lo + right_free_hi)
    g5_lo = right_mid - 0.5 * right_w
    g5_hi = right_mid + 0.5 * right_w

    if not (g1_lo >= interior_lo and g6_hi <= interior_hi):
        raise RuntimeError("Slabs exceed interior boundaries.")
    

    def in_range(xx, lo, hi, include_hi=False):
        if include_hi:
            return (xx >= lo - EPS) & (xx <= hi + EPS)
        return (xx >= lo - EPS) & (xx < hi - EPS)

    groups = np.zeros(N, dtype=int)

    # Assign walls (group 0) by x position relative to interior
    # IMPORTANT: atoms could have x outside [0,Lx] depending on wrapping;
    # for most GPUMD models x should already be inside. If not, wrap is needed.
    # Default all to wall 0
    groups[:] = 0

    # Default all atoms to wall (0)
    groups[:] = 0

    # Assign NEMD slabs
    # Start from walls / unused outer region
    groups[:] = 0

    # End slabs
    groups[in_range(x, g1_lo, g1_hi)] = 1
    groups[in_range(x, g6_lo, g6_hi)] = 6

    # SHC analysis slabs
    groups[in_range(x, g3_lo, g3_hi)] = 3
    groups[in_range(x, g4_lo, g4_hi)] = 4
    groups[in_range(x, g5_lo, g5_hi)] = 5

    # Group 2 = all remaining interior gaps
    interior_mask = (x >= interior_lo - EPS) & (x < interior_hi - EPS)
    assigned_mask = np.isin(groups, [1, 3, 4, 5, 6])
    groups[interior_mask & (~assigned_mask)] = 2

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
        "wallL": (xlo_box, interior_lo),
        "g1_hot": (g1_lo, g1_hi),

        "g2_gap_1": (g1_hi, g3_lo),
        "g3_left": (g3_lo, g3_hi),
        "g2_gap_2": (g3_hi, g4_lo),
        "g4_core": (g4_lo, g4_hi),
        "g2_gap_3": (g4_hi, g5_lo),
        "g5_right": (g5_lo, g5_hi),
        "g2_gap_4": (g5_hi, g6_lo),

        "g6_cold": (g6_lo, g6_hi),
        "wallR": (interior_hi, xhi_box),
    }

    print("\nRegion bounds along x (Å):")
    for name, (lo, hi) in bounds.items():
        print(f"  {name:>12s}: {lo:.3f} -> {hi:.3f}")

    plot_group_ranges(x, groups, bounds, plot_png)
    print(f"\n✔ Saved plot: {plot_png}")

    # restore stdout/stderr and close log
    sys.stdout, sys.stderr = old_stdout, old_stderr
    log_fh.close()


if __name__ == "__main__":
    main()
