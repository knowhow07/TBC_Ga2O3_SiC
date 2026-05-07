import re, numpy as np

# ---- user settings (confirm with POTCARs) ----
VALENCE = {"Si": 4, "C": 4, "Ga": 13, "O": 6, "H": 1}
SIDE_BY_SPECIES = True              # or False to split by z
Z0_MANUAL = None                    # used only if SIDE_BY_SPECIES=False
EXCLUDE_FROM_TRANSFER = {"H"}       # e.g., passivating H
# ------------------------------------------------

def read_poscar(path="POSCAR"):
    with open(path, "r") as f:
        raw = [l.rstrip() for l in f if l.strip() != ""]
    if len(raw) < 8:
        raise RuntimeError("POSCAR too short or missing lines.")

    scale = float(raw[1].split()[0])
    a = np.fromstring(raw[2], sep=" ")[:3] * scale
    b = np.fromstring(raw[3], sep=" ")[:3] * scale
    c = np.fromstring(raw[4], sep=" ")[:3] * scale

    line5, line6 = raw[5].split(), raw[6].split()
    if all(re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", t) for t in line5) and all(t.isdigit() for t in line6):
        symbols = line5; counts = list(map(int, line6)); idx = 7
    else:
        symbols = None; counts = list(map(int, line5)); idx = 6

    natoms = sum(counts)
    mode = raw[idx].strip().lower()
    if mode.startswith("s"):
        idx += 1
        mode = raw[idx].strip().lower()
    direct = mode.startswith("d")
    idx += 1

    coords_frac = []
    for i in range(natoms):
        parts = raw[idx + i].split()
        if len(parts) < 3:
            raise RuntimeError(f"Coordinate line {idx+i} malformed: '{raw[idx+i]}'")
        coords_frac.append(list(map(float, parts[:3])))
    coords_frac = np.array(coords_frac)

    cart = coords_frac @ np.vstack([a, b, c]) if direct else coords_frac

    if symbols is None:
        raise RuntimeError("POSCAR lacks symbols line; add it or map indices to elements.")

    elems = []
    for s, n in zip(symbols, counts):
        elems += [s] * n

    area = np.linalg.norm(np.cross(a, b))  # Å^2
    return elems, cart, area

def read_acf(path="ACF.dat"):
    z_list, e_list = [], []
    with open(path, "r") as f:
        for line in f:
            if re.match(r"\s*\d+\s", line):
                parts = line.split()
                z_list.append(float(parts[3]))   # Z
                e_list.append(float(parts[4]))   # Bader electrons
    if not e_list:
        raise RuntimeError("No atom lines found in ACF.dat.")
    return np.array(z_list), np.array(e_list)

def main():
    elems, coords, area = read_poscar("POSCAR")
    zpos = coords[:, 2]
    _, bader_e = read_acf("ACF.dat")

    if len(bader_e) != len(elems):
        raise RuntimeError(f"Atom count mismatch: POSCAR {len(elems)} vs ACF.dat {len(bader_e)}")

    try:
        val = np.array([VALENCE[e] for e in elems], dtype=float)
    except KeyError as e:
        raise RuntimeError(f"Element '{e.args[0]}' missing in VALENCE dict.") from None

    q = val - bader_e  # net charge per atom (e)

    # Masks
    include_mask = np.array([e not in EXCLUDE_FROM_TRANSFER for e in elems])
    if SIDE_BY_SPECIES:
        sic_mask_base = np.array([e in {"Si","C"} for e in elems])
        gao_mask_base = np.array([e in {"Ga","O"} for e in elems])
    else:
        z0 = Z0_MANUAL if Z0_MANUAL is not None else float(np.median(zpos))
        sic_mask_base = (zpos <= z0)
        gao_mask_base = (zpos >  z0)

    sic_mask = sic_mask_base & include_mask
    gao_mask = gao_mask_base & include_mask

    Q_sic = float(q[sic_mask].sum())
    Q_gao = float(q[gao_mask].sum())

    print("=== Bader summary ===")
    print(f"Atoms: {len(elems)}")
    print(f"Total net charge (should ~0): {q.sum(): .6f} e")
    print(f"SiC side total charge:         {Q_sic: .6f} e")
    print(f"Ga2O3 side total charge:       {Q_gao: .6f} e")
    print(f"Charge transfer magnitude:     {abs(Q_sic): .6f} e")
    print(f"Interface area:                {area: .3f} Å^2")
    print(f"Charge transfer per area:      {abs(Q_sic)/area: .6e} e/Å^2")

    print("\nFirst 5 atoms: elem  N_bader  q=Zval-N_bader")
    for i in range(min(5, len(elems))):
        print(f"{i+1:4d}  {elems[i]:>2s}  {bader_e[i]:8.4f}  {q[i]:8.4f}")

if __name__ == "__main__":
    main()
