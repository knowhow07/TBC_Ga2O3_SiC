#!/usr/bin/env python3
import numpy as np
from phonopy import load

# ======================================================================
# User knobs
# ======================================================================
PHONOPY_YAML    = "./phonopy.yaml"
FORCE_CONSTANTS = "./FORCE_CONSTANTS"
STRUCTURE_FILE  = "./CONTCAR"   # fallback structure file
AXIS            = "z"         # interface normal: "x", "y", or "z"
WINDOW          = 4.0         # keep atoms within +/- WINDOW (Angstrom) around interface
PAIR_CUTOFF     = 4.0         # keep cross-interface pairs with distance <= cutoff (Angstrom)
TOP_N           = 20          # print top N strongest pairs
PAIR_OUT        = "cross_ifc_pairs.txt"
# ======================================================================


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-14:
        raise ValueError("Zero-length vector.")
    return v / n


def frob_norm(mat):
    return float(np.linalg.norm(mat, ord="fro"))


def scalar_proj(Phi, a, b):
    return float(np.dot(a, Phi @ b))


def read_structure_and_fc(phonopy_yaml, force_constants, structure_file=None):
    if structure_file is not None:
        ph = load(
            phonopy_yaml=phonopy_yaml,
            force_constants_filename=force_constants,
            unitcell_filename=structure_file,
        )
    else:
        ph = load(
            phonopy_yaml=phonopy_yaml,
            force_constants_filename=force_constants,
        )

    fc = ph.force_constants
    cell = ph.supercell
    positions = np.array(cell.positions, dtype=float)   # Cartesian, Angstrom
    symbols = list(cell.symbols)
    lattice = np.array(cell.cell, dtype=float)          # 3x3, lattice vectors as rows
    return fc, positions, symbols, lattice


def cart_to_frac(cart, lattice):
    return np.linalg.solve(lattice.T, cart.T).T


def frac_to_cart(frac, lattice):
    return frac @ lattice


def get_axis_basis(axis="z"):
    if axis.lower() == "x":
        n = np.array([1.0, 0.0, 0.0])
        t1 = np.array([0.0, 1.0, 0.0])
        t2 = np.array([0.0, 0.0, 1.0])
    elif axis.lower() == "y":
        n = np.array([0.0, 1.0, 0.0])
        t1 = np.array([1.0, 0.0, 0.0])
        t2 = np.array([0.0, 0.0, 1.0])
    elif axis.lower() == "z":
        n = np.array([0.0, 0.0, 1.0])
        t1 = np.array([1.0, 0.0, 0.0])
        t2 = np.array([0.0, 1.0, 0.0])
    else:
        raise ValueError("AXIS must be x, y, or z")
    return n, t1, t2


def interface_area_from_lattice(lattice, axis="z"):
    a, b, c = lattice
    if axis.lower() == "x":
        return float(np.linalg.norm(np.cross(b, c)))
    elif axis.lower() == "y":
        return float(np.linalg.norm(np.cross(a, c)))
    elif axis.lower() == "z":
        return float(np.linalg.norm(np.cross(a, b)))
    else:
        raise ValueError("AXIS must be x, y, or z")


def classify_species(symbol):
    if symbol in ("Si", "C"):
        return "SiC"
    elif symbol in ("Ga", "O"):
        return "Ga2O3"
    else:
        return "OTHER"


def detect_interface_location(positions, symbols, axis="z", verbose=True):
    """
    Auto-detect interface from the species switch along the chosen axis.
    It scans atoms sorted by coordinate and finds Si/C <-> Ga/O transitions.
    Then it chooses the transition with the largest gap.
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    iax = axis_map[axis.lower()]

    coords = positions[:, iax]
    data = []
    for i, (coord, sym) in enumerate(zip(coords, symbols)):
        side = classify_species(sym)
        if side == "OTHER":
            continue
        data.append((i, coord, sym, side))

    data.sort(key=lambda x: x[1])

    switches = []
    for k in range(len(data) - 1):
        i1, c1, s1, side1 = data[k]
        i2, c2, s2, side2 = data[k + 1]
        if side1 != side2:
            mid = 0.5 * (c1 + c2)
            gap = c2 - c1
            switches.append({
                "left_index": i1,
                "right_index": i2,
                "c1": c1,
                "c2": c2,
                "mid": mid,
                "gap": gap,
                "left_symbol": s1,
                "right_symbol": s2,
                "left_side": side1,
                "right_side": side2,
            })

    if not switches:
        raise RuntimeError("No Si/C <-> Ga/O switch found. Check AXIS or structure.")

    best = max(switches, key=lambda d: d["gap"])
    z0 = best["mid"]

    if verbose:
        print("=== Auto-detected interface ===")
        print(f"Axis: {axis}")
        print(f"Chosen switch: {best['left_symbol']}({best['left_side']}) -> "
              f"{best['right_symbol']}({best['right_side']})")
        print(f"coord1 = {best['c1']:.6f} Å, coord2 = {best['c2']:.6f} Å")
        print(f"Interface location = {z0:.6f} Å")
        print(f"Local gap = {best['gap']:.6f} Å")
        print()

    return z0, switches


def minimum_image_displacement(rj, ri, lattice):
    """
    Minimum-image displacement for a general cell.
    """
    df = cart_to_frac(np.array([rj - ri]), lattice)[0]
    df -= np.round(df)
    dr = frac_to_cart(np.array([df]), lattice)[0]
    return dr


def choose_interfacial_atoms(positions, symbols, interface_coord, axis="z", window=4.0):
    axis_map = {"x": 0, "y": 1, "z": 2}
    iax = axis_map[axis.lower()]

    sic_idx = []
    gao_idx = []
    ignored = []

    for i, (r, s) in enumerate(zip(positions, symbols)):
        side = classify_species(s)
        coord = r[iax]

        if abs(coord - interface_coord) > window:
            continue

        if side == "SiC":
            sic_idx.append(i)
        elif side == "Ga2O3":
            gao_idx.append(i)
        else:
            ignored.append(i)

    return sic_idx, gao_idx, ignored


def compute_ifc_metrics(fc, positions, symbols, lattice,
                        axis="z", window=4.0, pair_cutoff=4.0, top_n=20, verbose=True):
    n, t1, t2 = get_axis_basis(axis)
    area = interface_area_from_lattice(lattice, axis=axis)
    interface_coord, switches = detect_interface_location(positions, symbols, axis=axis, verbose=verbose)

    sic_idx, gao_idx, ignored = choose_interfacial_atoms(
        positions, symbols, interface_coord, axis=axis, window=window
    )

    if verbose:
        print("=== Interfacial atom selection ===")
        print(f"Window around interface: +/- {window:.3f} Å")
        print(f"SiC-side atoms kept:    {len(sic_idx)}")
        print(f"Ga2O3-side atoms kept:  {len(gao_idx)}")
        print(f"Ignored OTHER atoms:    {len(ignored)}")
        print(f"Interface area A:       {area:.6f} Å^2")
        print()

    if len(sic_idx) == 0 or len(gao_idx) == 0:
        raise RuntimeError("No interfacial atoms found on one side. Increase WINDOW.")

    Kcross = 0.0
    Knn = 0.0
    Ktt = 0.0
    pair_data = []

    for i in sic_idx:
        for j in gao_idx:
            dr = minimum_image_displacement(positions[j], positions[i], lattice)
            dist = np.linalg.norm(dr)

            if pair_cutoff is not None and dist > pair_cutoff:
                continue

            Phi = fc[i, j]
            fn = frob_norm(Phi)
            nn = abs(scalar_proj(Phi, n, n))
            tt = abs(scalar_proj(Phi, t1, t1)) + abs(scalar_proj(Phi, t2, t2))

            Kcross += fn
            Knn += nn
            Ktt += tt

            pair_data.append({
                "i": i,
                "j": j,
                "sym_i": symbols[i],
                "sym_j": symbols[j],
                "dist": dist,
                "frob": fn,
                "nn": nn,
                "tt": tt,
            })

    pair_data.sort(key=lambda d: d["frob"], reverse=True)

    return {
        "interface_coord": interface_coord,
        "A": area,
        "num_pairs": len(pair_data),
        "Kcross": Kcross,
        "Kcross_per_A": Kcross / area,
        "Knn": Knn,
        "Knn_per_A": Knn / area,
        "Ktt": Ktt,
        "Ktt_per_A": Ktt / area,
        "pairs": pair_data,
        "top_pairs": pair_data[:top_n],
    }


def print_results(res, top_n=20, axis="z"):
    coord_name = axis.lower()
    print("=== Cross-interface IFC metrics ===")
    print(f"Interface location {coord_name}0 (Å): {res['interface_coord']:.6f}")
    print(f"Interface area A (Å^2):            {res['A']:.6f}")
    print(f"Number of cross pairs used:        {res['num_pairs']}")
    print()
    print(f"Kcross      = {res['Kcross']:.10f}")
    print(f"Kcross / A  = {res['Kcross_per_A']:.10f}")
    print(f"Knn         = {res['Knn']:.10f}")
    print(f"Knn / A     = {res['Knn_per_A']:.10f}")
    print(f"Ktt         = {res['Ktt']:.10f}")
    print(f"Ktt / A     = {res['Ktt_per_A']:.10f}")
    print()

    nprint = min(top_n, len(res["top_pairs"]))
    print(f"=== Top {nprint} strongest cross-interface pairs by ||Phi_ij||_F ===")
    print(f"{'rank':>4s} {'i':>4s} {'j':>4s} {'pair':>8s} {'dist(Å)':>10s} {'||Phi||_F':>14s} {'nn':>12s} {'tt':>12s}")
    for rank, d in enumerate(res["top_pairs"], start=1):
        pair_name = f"{d['sym_i']}-{d['sym_j']}"
        print(f"{rank:4d} {d['i']+1:4d} {d['j']+1:4d} {pair_name:>8s} "
              f"{d['dist']:10.4f} {d['frob']:14.6f} {d['nn']:12.6f} {d['tt']:12.6f}")


def save_pair_table(outfile, pair_data):
    with open(outfile, "w") as f:
        f.write("# i  j  sym_i  sym_j  dist_A  frob  nn  tt\n")
        for d in pair_data:
            f.write(
                f"{d['i']+1:4d} {d['j']+1:4d} "
                f"{d['sym_i']:>3s} {d['sym_j']:>3s} "
                f"{d['dist']:12.6f} {d['frob']:16.8f} {d['nn']:16.8f} {d['tt']:16.8f}\n"
            )


def main():
    fc, positions, symbols, lattice = read_structure_and_fc(
        phonopy_yaml=PHONOPY_YAML,
        force_constants=FORCE_CONSTANTS,
        structure_file=STRUCTURE_FILE,
    )

    res = compute_ifc_metrics(
        fc=fc,
        positions=positions,
        symbols=symbols,
        lattice=lattice,
        axis=AXIS,
        window=WINDOW,
        pair_cutoff=PAIR_CUTOFF,
        top_n=TOP_N,
        verbose=True,
    )

    print_results(res, top_n=TOP_N, axis=AXIS)
    save_pair_table(PAIR_OUT, res["pairs"])
    print()
    print(f"Saved full pair table to: {PAIR_OUT}")


if __name__ == "__main__":
    main()