#!/usr/bin/env python3
"""
avg_sio_interface_bonds.py
Compute the average Si–O bond length at the interface from a VASP POSCAR.

How it works (default AUTO mode):
1) Parse POSCAR (VASP5) with Direct or Cartesian coords.
2) Build all Si–O pairs using full periodic minimum-image distances.
3) Keep pairs with d <= CUTOFF (Å).
4) For each kept pair, compute the bond midpoint z (Å) along the cell's c-axis
   in a periodic-safe way.
5) Define the interface plane z0 as the median of all Si–O bond midpoints, then
   keep only bonds with |z_mid - z0| <= HALF_THICKNESS (Å).
6) Report average, std, count, and a few nearest bonds.

You can switch to MANUAL mode, setting Z0_ANG and HALF_THICKNESS yourself.

Author: you ✨
"""

import re
import numpy as np
from pathlib import Path

# ------------------ user settings ------------------
POSCAR_PATH = "POSCAR"

# neighbor cutoff for Si–O (Å); tweak if your interface is longer/shorter
CUTOFF = 2.20

# interface selection
AUTO_INTERFACE = True     # True: estimate z0 from bond midpoints; False: use Z0_ANG
HALF_THICKNESS = 2.0      # Å on each side of z0 (window thickness = 2*HALF_THICKNESS)
Z0_ANG = None             # Å (only used if AUTO_INTERFACE=False)

# how many sample bonds to print
N_SHOW = 10
# ---------------------------------------------------

def read_poscar(path="POSCAR"):
    with open(path, "r") as f:
        raw = [l.rstrip() for l in f if l.strip() != ""]
    if len(raw) < 8:
        raise RuntimeError("POSCAR too short or missing lines.")

    scale = float(raw[1].split()[0])
    a = np.fromstring(raw[2], sep=" ")[:3] * scale
    b = np.fromstring(raw[3], sep=" ")[:3] * scale
    c = np.fromstring(raw[4], sep=" ")[:3] * scale
    A = np.vstack([a, b, c]).T  # 3x3 lattice matrix (columns are lattice vectors)

    # detect symbols + counts (VASP5)
    line5 = raw[5].split()
    line6 = raw[6].split()
    if all(re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", t) for t in line5) and all(t.isdigit() for t in line6):
        symbols = line5
        counts  = list(map(int, line6))
        idx = 7
    else:
        raise RuntimeError("POSCAR must be in VASP5 format (symbols + counts).")

    nat = sum(counts)
    mode = raw[idx].strip().lower()
    if mode.startswith("s"):
        idx += 1
        mode = raw[idx].strip().lower()
    direct = mode.startswith("d")
    idx += 1

    fcoords = []
    for i in range(nat):
        parts = raw[idx + i].split()
        if len(parts) < 3:
            raise RuntimeError(f"Malformed coord line {idx+i}: '{raw[idx+i]}'")
        fcoords.append([float(parts[0]), float(parts[1]), float(parts[2])])
    fcoords = np.array(fcoords)

    # wrap to [0,1)
    fcoords = fcoords - np.floor(fcoords)

    # cart coords
    if direct:
        r = fcoords @ A
    else:
        # If given in Cartesian, convert back to fractional to keep both
        r = fcoords
        fcoords = r @ np.linalg.inv(A)
        fcoords = fcoords - np.floor(fcoords)

    # element list
    elems = []
    for s, n in zip(symbols, counts):
        elems += [s] * n

    # useful lengths
    c_len = np.linalg.norm(c)
    area_ab = np.linalg.norm(np.cross(a, b))

    return {
        "A": A, "a": a, "b": b, "c": c, "c_len": c_len, "area_ab": area_ab,
        "fcoords": fcoords, "cart": r, "elems": elems, "symbols": symbols, "counts": counts
    }

def pbc_delta_frac(fj, fi):
    """Minimum-image fractional difference Δf = fj - fi wrapped to [-0.5, 0.5)."""
    df = fj - fi
    return df - np.round(df)

def pair_distance_cart(A, fi, fj):
    """Distance using minimum-image in fractional space -> Cartesian."""
    df = pbc_delta_frac(fj, fi)
    d_cart = df @ A
    return np.linalg.norm(d_cart), df, d_cart

def compute_si_o_interface_avg(poscar_dict):
    A = poscar_dict["A"]
    f = poscar_dict["fcoords"]
    elems = poscar_dict["elems"]
    c_len = poscar_dict["c_len"]

    idx_Si = [i for i,e in enumerate(elems) if e == "Si"]
    idx_O  = [i for i,e in enumerate(elems) if e == "O"]

    if not idx_Si or not idx_O:
        raise RuntimeError("No Si or O atoms found in POSCAR.")

    bonds = []  # list of (d, zmid_A, i_Si, j_O)
    for i in idx_Si:
        fi = f[i]
        for j in idx_O:
            fj = f[j]
            d, df, d_cart = pair_distance_cart(A, fi, fj)
            if d <= CUTOFF:
                # periodic-safe midpoint in fractional coords: fi + df/2
                fmid = fi + 0.5 * df
                fmid = fmid - np.floor(fmid)
                zmid_ang = fmid[2] * c_len
                bonds.append((d, zmid_ang, i, j))

    if not bonds:
        raise RuntimeError(f"No Si–O pairs found within cutoff {CUTOFF} Å. Increase CUTOFF if needed.")

    dists = np.array([b[0] for b in bonds])
    zmid  = np.array([b[1] for b in bonds])

    if AUTO_INTERFACE:
        z0 = float(np.median(zmid))
    else:
        if Z0_ANG is None:
            raise RuntimeError("AUTO_INTERFACE=False but Z0_ANG is not set.")
        z0 = float(Z0_ANG)

    in_window = np.abs(zmid - z0) <= HALF_THICKNESS
    bonds_if = [b for b, keep in zip(bonds, in_window) if keep]

    if not bonds_if:
        raise RuntimeError(
            f"No Si–O bonds fall within |z_mid - z0| <= {HALF_THICKNESS} Å (z0={z0:.3f} Å). "
            "Try increasing HALF_THICKNESS or check AUTO_INTERFACE."
        )

    d_if = np.array([b[0] for b in bonds_if])
    avg = float(d_if.mean())
    std = float(d_if.std(ddof=1)) if len(d_if) > 1 else 0.0

    # sort a few shortest for inspection
    bonds_if_sorted = sorted(bonds_if, key=lambda x: x[0])[:N_SHOW]

    summary = {
        "n_all_pairs_within_cutoff": len(bonds),
        "z0_interface_A": z0,
        "half_thickness_A": HALF_THICKNESS,
        "n_interface_pairs": len(bonds_if),
        "avg_Interface_SiO_A": avg,
        "std_Interface_SiO_A": std,
        "examples_shortest": bonds_if_sorted  # (d, zmid, i_Si, j_O)
    }
    return summary

def main():
    P = read_poscar(POSCAR_PATH)
    res = compute_si_o_interface_avg(P)

    print("=== Si–O Interface Bond Statistics ===")
    print(f"POSCAR: {Path(POSCAR_PATH).resolve()}")
    print(f"Cutoff (Å): {CUTOFF}")
    print(f"AUTO_INTERFACE: {AUTO_INTERFACE}")
    print(f"Interface z0 (Å): {res['z0_interface_A']:.3f}")
    print(f"Window ± (Å): {res['half_thickness_A']:.3f}")
    print(f"Total Si–O pairs within cutoff: {res['n_all_pairs_within_cutoff']}")
    print(f"Interface pairs in window:      {res['n_interface_pairs']}")
    print(f"Average d_Si–O at interface (Å): {res['avg_Interface_SiO_A']:.4f}")
    print(f"Std dev (Å):                     {res['std_Interface_SiO_A']:.4f}")

    print("\nShortest interface Si–O bonds (Å, z_mid[Å], i_Si, j_O):")
    for d, zmid, i, j in res["examples_shortest"]:
        print(f"  {d:7.4f}   {zmid:8.3f}   {i:4d} (Si)  {j:4d} (O)")

if __name__ == "__main__":
    main()
