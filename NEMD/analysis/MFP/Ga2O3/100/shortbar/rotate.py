#!/usr/bin/env python3
import numpy as np
import re
import os

path = ""
infile  = os.path.join(path,f"Ga2O3_(1, 0, 0)_4H-SiC(0001)_Si-O_150_0.03_4+1.cif")
outfile = os.path.join(path, "rotated_longx.xyz")

# -----------------------------
# 1. Read CIF
# -----------------------------
with open(infile) as f:
    # strip blank lines to simplify parsing
    cif_lines = [ln.strip() for ln in f if ln.strip()]

# -----------------------------
# 2. Parse cell lengths a, b, c
# -----------------------------
a = b = c = None
for ln in cif_lines:
    if ln.startswith("_cell_length_a"):
        a = float(ln.split()[1])
    elif ln.startswith("_cell_length_b"):
        b = float(ln.split()[1])
    elif ln.startswith("_cell_length_c"):
        c = float(ln.split()[1])

if any(v is None for v in (a, b, c)):
    raise RuntimeError("Could not find all _cell_length_* entries in CIF.")

# -----------------------------
# 3. Find atom_site loop and parse fractional coords
# -----------------------------
atom_loop_start = None
atom_labels = None
atom_data_start = None

for i, ln in enumerate(cif_lines):
    if ln.lower().startswith("loop_"):
        j = i + 1
        labels = []
        # collect labels under this loop_
        while j < len(cif_lines) and cif_lines[j].startswith("_"):
            labels.append(cif_lines[j])
            j += 1
        # check if this is the atom_site loop
        if any("_atom_site_" in lab for lab in labels):
            atom_loop_start = i
            atom_labels = labels
            atom_data_start = j
            break

if atom_loop_start is None:
    raise RuntimeError("Could not find atom_site loop in CIF.")

# Map column name → index
label_to_idx = {lab: idx for idx, lab in enumerate(atom_labels)}

def find_col(prefix):
    """Find column whose label starts with given prefix (case-insensitive)."""
    for lab, idx in label_to_idx.items():
        if lab.lower().startswith(prefix.lower()):
            return idx
    raise RuntimeError(f"Column {prefix} not found in atom_site loop.")

ix_type = find_col("_atom_site_type_symbol")
ix_fx   = find_col("_atom_site_fract_x")
ix_fy   = find_col("_atom_site_fract_y")
ix_fz   = find_col("_atom_site_fract_z")

species = []
frac_coords = []

for ln in cif_lines[atom_data_start:]:
    # stop if we hit another loop_ or a new set of definitions
    if ln.startswith("_") or ln.lower().startswith("loop_"):
        break
    parts = ln.split()
    if len(parts) <= max(ix_type, ix_fx, ix_fy, ix_fz):
        continue

    type_symbol = parts[ix_type]
    # strip oxidation state: e.g., "Ga3+" → "Ga"
    m = re.match(r"[A-Za-z]+", type_symbol)
    sp = m.group(0) if m else type_symbol
    species.append(sp)

    fx = float(parts[ix_fx])
    fy = float(parts[ix_fy])
    fz = float(parts[ix_fz])
    frac_coords.append([fx, fy, fz])

frac_coords = np.array(frac_coords, dtype=float)
N = frac_coords.shape[0]

# -----------------------------
# 4. Build old Cartesian coords (orthogonal cell)
#    CIF is orthorhombic here: a→x, b→y, c→z
# -----------------------------
cell_old = np.diag([a, b, c])        # 3×3
coords_cart = frac_coords @ cell_old  # (N,3)

# -----------------------------
# 5. Rotate / permute axes (same as previous script)
#    new_x = old_z, new_y = old_x, new_z = old_y
# -----------------------------
perm = (2, 0, 1)
coords_rot = coords_cart[:, perm]

# Old lengths
La, Lb, Lc = a, b, c
Lx, Ly, Lz = Lc, La, Lb  # after rotation

# -----------------------------
# 6. Replicate 12x along thinnest direction (y after rotation)
# -----------------------------
nrep = 12
coords_rep = []
species_rep = []

for ny in range(nrep):
    shift = np.array([0.0, ny * Ly, 0.0])
    coords_rep.append(coords_rot + shift)
    species_rep.extend(species)

coords_rep = np.vstack(coords_rep)
N_tot = coords_rep.shape[0]

# New lattice: y extended by factor nrep
new_lattice = np.array([
    [Lx,        0.0,       0.0],
    [0.0,  nrep*Ly,       0.0],
    [0.0,        0.0,      Lz],
])

# -----------------------------
# 7. Build header with Lattice / Origin / Properties
# -----------------------------
new_lat_string = (
    f'Lattice="{new_lattice[0,0]:.8f} {new_lattice[0,1]:.8f} {new_lattice[0,2]:.8f} '
    f'{new_lattice[1,0]:.8f} {new_lattice[1,1]:.8f} {new_lattice[1,2]:.8f} '
    f'{new_lattice[2,0]:.8f} {new_lattice[2,1]:.8f} {new_lattice[2,2]:.8f}"'
)
origin_string = 'Origin="0.00000000 0.00000000 0.00000000"'
properties_string = 'Properties=id:I:1:species:S:1:pos:R:3'

header_new = f"{new_lat_string} {origin_string} {properties_string}"

# -----------------------------
# 8. Write XYZ
# -----------------------------
with open(outfile, "w") as f:
    f.write(f"{N_tot}\n")
    f.write(header_new + "\n")
    idx = 1
    for (x, y, z), sp in zip(coords_rep, species_rep):
        f.write(f"{idx:d} {sp} {x:.10f} {y:.10f} {z:.10f}\n")
        idx += 1

# quick sanity check
Lx_out = np.linalg.norm(new_lattice[0])
Ly_out = np.linalg.norm(new_lattice[1])
Lz_out = np.linalg.norm(new_lattice[2])
print("New lattice:")
print(new_lattice)
print(f"Lengths: Lx={Lx_out:.3f}, Ly={Ly_out:.3f}, Lz={Lz_out:.3f}")
