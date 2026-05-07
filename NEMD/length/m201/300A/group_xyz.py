#!/usr/bin/env python3
import numpy as np

infile  = "rotated_longx.xyz"   # input from your rotation step
outfile = "model.xyz"           # GPUMD-ready output

# ---------------- User knobs ----------------
wall_thick   = 10.0   # Å, thickness of frozen walls at both ends
hot_thick    = 10.0   # Å, hot slab thickness near left wall
cold_thick   = 10.0   # Å, cold slab thickness near right wall
N_inter_bins = 20     # number of temperature bins between hot and cold
# -------------------------------------------


def parse_xyz_line_with_id(line):
    """
    Parse lines like:
      '1 Ga 1.23 2.34 3.45'
      '1 Ga 1.23 2.34 3.45 7'
      'Ga 1.23 2.34 3.45'
      'Ga 1.23 2.34 3.45 7'
    Returns: (id, species, x, y, z)
    If no id is present, id is assigned by order later.
    """
    parts = line.split()

    atom_id = None
    try:
        atom_id = int(parts[0])
        species = parts[1]
        x, y, z = map(float, parts[2:5])
        return atom_id, species, x, y, z
    except (ValueError, IndexError):
        species = parts[0]
        x, y, z = map(float, parts[1:4])
        return None, species, x, y, z


# =======================
# Load XYZ
# =======================
with open(infile) as f:
    header = f.readline()
    box    = f.readline().rstrip("\n")  # second line (lattice, properties, etc.)
    lines  = f.readlines()

N = int(header.strip())

ids = []
species = []
coords = []

for idx, line in enumerate(lines[:N]):
    line = line.strip()
    if not line:
        continue
    atom_id, sp, x, y, z = parse_xyz_line_with_id(line)
    if atom_id is None:  # no id found, assign sequentially 1..N
        atom_id = idx + 1
    ids.append(atom_id)
    species.append(sp)
    coords.append([x, y, z])

coords = np.array(coords)
x = coords[:, 0]

xlo, xhi = x.min(), x.max()
Lx = xhi - xlo

print(f"Loaded {N} atoms, x range: [{xlo:.3f}, {xhi:.3f}] (Lx={Lx:.3f} Å)")

# =======================
# Define regions & groups
# =======================

hot_group  = 1
first_inter_group = 2
last_inter_group  = first_inter_group + N_inter_bins - 1
cold_group = last_inter_group + 1   # = N_inter_bins + 2

print(f"Using group IDs: wall=0, hot={hot_group}, "
      f"interior={first_inter_group}..{last_inter_group}, cold={cold_group}")

groups = np.zeros(N, dtype=int)

# region boundaries
x_fixL_hi = xlo + wall_thick
x_fixR_lo = xhi - wall_thick

x_hot_lo  = x_fixL_hi
x_hot_hi  = x_hot_lo + hot_thick

x_cold_hi = x_fixR_lo
x_cold_lo = x_cold_hi - cold_thick

# sanity check: non-overlapping hot/cold
if not (x_hot_hi < x_cold_lo):
    raise RuntimeError("Hot and cold regions overlap: adjust wall/hot/cold thicknesses.")

# --- walls: group 0 ---
wall_mask = (x <= x_fixL_hi) | (x >= x_fixR_lo)
groups[wall_mask] = 0

# --- hot region: group 1 ---
hot_mask = (x >= x_hot_lo) & (x <= x_hot_hi)
groups[hot_mask] = hot_group

# --- cold region: cold_group ---
cold_mask = (x >= x_cold_lo) & (x <= x_cold_hi)
groups[cold_mask] = cold_group

# --- interior region: split into N_inter_bins groups ---
inter_mask = (x > x_hot_hi) & (x < x_cold_lo) & (groups == 0)

x_inter = x[inter_mask]
L_inter = x_cold_lo - x_hot_hi

if N_inter_bins <= 0 or L_inter <= 0.0:
    raise RuntimeError("Interior region invalid or N_inter_bins <= 0.")

frac = (x_inter - x_hot_hi) / L_inter           # in [0,1)
frac = np.clip(frac, 0.0, 1.0 - 1e-12)
bin_idx = (frac * N_inter_bins).astype(int)     # 0..N_inter_bins-1

groups[inter_mask] = first_inter_group + bin_idx  # 2..N_inter_bins+1

# =======================
# Fix properties in box line
# =======================
lower = box.lower()
prop_key = "properties="

if prop_key in lower:
    idx = lower.index(prop_key)
    prefix = box[:idx]
    box = prefix + 'properties=id:I:1:species:S:1:pos:R:3:group:I:1'
else:
    if not box.endswith(" "):
        box += " "
    box += 'properties=id:I:1:species:S:1:pos:R:3:group:I:1'

# =======================
# Write output model.xyz
# =======================
with open(outfile, "w") as f:
    f.write(f"{N}\n")
    f.write(box + "\n")
    for atom_id, sp, (xi, yi, zi), g in zip(ids, species, coords, groups):
        f.write(f"{atom_id:d} {sp} {xi:.8f} {yi:.8f} {zi:.8f} {g:d}\n")

print(f"\n✔ Wrote grouped model to: {outfile}")
unique_groups, counts = np.unique(groups, return_counts=True)
print("Group counts:")
for g, c in zip(unique_groups, counts):
    print(f"  group {g:2d}: {c}")
