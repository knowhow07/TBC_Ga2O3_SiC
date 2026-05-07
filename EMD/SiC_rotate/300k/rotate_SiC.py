#!/usr/bin/env python3
"""
Rotate atomic coordinates (NOT lattice) by clockwise 60° around +z, keep hex cell.
Replicate to a custom supercell.
Write EXTENDED XYZ (GPUMD-style header with Lattice/Origin/Properties + id).

Requires:
  - pymatgen
  - ase
"""

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write as ase_write

# ===================== user settings =====================

POSCAR_IN = "SiC-mp-11714.vasp"
ANGLE_DEG = -60.0          # clockwise 60° about +z
SUPERCELL = (6, 6, 2)      # replicate (na, nb, nc)

# Origin handling in extxyz header:
# - GPUMD often stores some nonzero origin; for most workflows Origin="0 0 0" is fine.
# - If you want GPUMD-like shift, set ORIGIN_MODE="min" to put min corner at origin.
ORIGIN_MODE = "min"        # "zero" or "min"

PRIM_BEFORE_XYZ = "prim_before.xyz"
PRIM_AFTER_XYZ  = "prim_after.xyz"
SC_BEFORE_XYZ   = "sc_before.xyz"
SC_AFTER_XYZ    = "sc_after.xyz"

# =========================================================

def rot_z_matrix(angle_deg: float) -> np.ndarray:
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def rotate_coords_keep_lattice(struct: Structure, angle_deg: float) -> Structure:
    lat = struct.lattice
    species = struct.species

    cart = lat.get_cartesian_coords(struct.frac_coords)
    R = rot_z_matrix(angle_deg)
    cart_rot = cart @ R.T

    frac_rot = lat.get_fractional_coords(cart_rot)

    return Structure(
        lattice=lat,
        species=species,
        coords=frac_rot,
        coords_are_cartesian=False,
        to_unit_cell=True
    )

def set_origin_and_ids(atoms):
    """
    Make extxyz include:
      - id (integer)
      - Origin in header
    """
    n = len(atoms)
    atoms.arrays["id"] = np.arange(1, n + 1, dtype=int)

    if ORIGIN_MODE == "zero":
        origin = np.array([0.0, 0.0, 0.0])
    elif ORIGIN_MODE == "min":
        # shift so that the minimum x/y/z becomes 0 (common for nicer coordinates)
        origin = atoms.positions.min(axis=0)
    else:
        raise ValueError("ORIGIN_MODE must be 'zero' or 'min'.")

    # Store origin in atoms.info so extxyz writer prints it in the header
    atoms.info["Origin"] = " ".join(f"{x:.10f}" for x in origin)

    # If we used "min", shift positions so origin is exactly that min corner
    if ORIGIN_MODE == "min":
        atoms.positions = atoms.positions - origin

    return atoms

def write_extxyz(struct: Structure, filename: str):
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms = set_origin_and_ids(atoms)

    # Write EXTXYZ (this produces Lattice="..." and Properties=... in line 2)
    ase_write(filename, atoms, format="extxyz")
    print(f"[write] {filename}  (format=extxyz, origin_mode={ORIGIN_MODE})")

def main():
    s0 = Structure.from_file(POSCAR_IN)

    # primitive before/after
    write_extxyz(s0, PRIM_BEFORE_XYZ)
    s_rot = rotate_coords_keep_lattice(s0, ANGLE_DEG)
    write_extxyz(s_rot, PRIM_AFTER_XYZ)

    # supercells before/after
    sc0 = s0.copy()
    sc0.make_supercell(SUPERCELL)
    write_extxyz(sc0, SC_BEFORE_XYZ)

    sc1 = s_rot.copy()
    sc1.make_supercell(SUPERCELL)
    write_extxyz(sc1, SC_AFTER_XYZ)

    print("[done]")

if __name__ == "__main__":
    main()
