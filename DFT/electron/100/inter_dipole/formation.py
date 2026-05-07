#!/usr/bin/env python3
import re, json
from pathlib import Path
import numpy as np

# ==== Paths (relative to this script's run directory) ====
interface_dir = Path(".")
slab2_dir = Path("../GaO_dipole")   # Ga2O3 slab
slab1_dir = Path("../SiCH_dipole_4H")  # SiC slab

# ==== Group definitions & exclusions ====
group1 = {"Si","C"}  # SiC side
group2 = {"Ga","O"}  # Ga2O3 side
exclude = {"H"}      # ignore passivating H for scaling

# ==== Helper functions ====
def read_energy_from_folder(folder):
    outcar = folder / "OUTCAR"
    oszicar = folder / "OSZICAR"
    if outcar.exists():
        E = None
        with open(outcar, "r", errors="ignore") as f:
            for line in f:
                if "free  energy   TOTEN  =" in line:
                    try:
                        E = float(line.split("=")[1].split()[0])
                    except Exception:
                        pass
        if E is not None:
            return E
    if oszicar.exists():
        E = None
        with open(oszicar, "r", errors="ignore") as f:
            for line in f:
                if "E0=" in line:
                    toks = line.strip().split()
                    for t in toks:
                        if t.startswith("E0="):
                            try:
                                E = float(t.split("=",1)[1])
                            except Exception:
                                pass
        if E is not None:
            return E
    raise RuntimeError(f"No final energy found in {folder}")

def read_poscar(folder):
    poscar = folder / "POSCAR"
    with open(poscar, "r") as f:
        raw = [l.rstrip() for l in f if l.strip()]
    scale = float(raw[1].split()[0])
    a = np.fromstring(raw[2], sep=" ")[:3] * scale
    b = np.fromstring(raw[3], sep=" ")[:3] * scale
    l5, l6 = raw[5].split(), raw[6].split()
    if all(re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", t) for t in l5) and all(t.isdigit() for t in l6):
        symbols = l5; counts = list(map(int, l6))
    else:
        raise RuntimeError("POSCAR missing symbols line (need VASP5 format)")
    elems = []
    for s, n in zip(symbols, counts):
        elems += [s] * n
    comp = {}
    for e in elems:
        comp[e] = comp.get(e, 0) + 1
    area = np.linalg.norm(np.cross(a, b))
    return comp, area

def infer_scale(comp_iface, comp_slab, group, exclude):
    n_iface = sum(comp_iface.get(el,0) for el in group if el not in exclude)
    n_slab  = sum(comp_slab.get(el,0)  for el in group if el not in exclude)
    if n_slab == 0:
        return 1.0
    return n_iface / n_slab

# ==== Read data ====
E_int = read_energy_from_folder(interface_dir)
E_s1  = read_energy_from_folder(slab1_dir)
E_s2  = read_energy_from_folder(slab2_dir)

comp_iface, area = read_poscar(interface_dir)
comp_s1, _ = read_poscar(slab1_dir)
comp_s2, _ = read_poscar(slab2_dir)

scale1 = infer_scale(comp_iface, comp_s1, group1, exclude)
scale2 = infer_scale(comp_iface, comp_s2, group2, exclude)

# ==== Compute gamma ====
gamma_evA2 = (E_int - (scale1 * E_s1 + scale2 * E_s2)) / area
gamma_Jm2 = gamma_evA2 * 16.02176565  # eV/Å² → J/m²

# ==== Output ====
print("=== Interface Binding Energy per Area ===")
print(f"Interface energy (eV): {E_int:.6f}")
print(f"SiC slab energy  (eV): {E_s1:.6f}  scale1={scale1:.3f}")
print(f"Ga2O3 slab energy(eV): {E_s2:.6f}  scale2={scale2:.3f}")
print(f"Interface area   (Å²): {area:.6f}")
print(f"Gamma_int (eV/Å²): {gamma_evA2:.6f}")
print(f"Gamma_int (J/m²):  {gamma_Jm2:.6f}")
