#!/usr/bin/env python3
"""
Compute 2nd-order force constants (FC2) using phonopy + NEP.

This script keeps the *same* environment-variable interface as your original
DeepMD version (DP_MODEL_PATH, DP_TYPE_MAP_MASTER, ...), but interprets:

  DP_MODEL_PATH  -> NEP model file (e.g., nep.txt / NEP_*.txt)

Force backend (auto):
  1) PyNEP (if importable): pynep.calculate.NEP as an ASE calculator
  2) LAMMPS python module (if importable)
  3) External LAMMPS binary (DP_LMP_BIN), using pair_style nep

LAMMPS NEP interface expects (typical):
  pair_style nep
  pair_coeff * * NEP_FILE <elem1> <elem2> ...
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.file_IO import write_FORCE_SETS, write_FORCE_CONSTANTS

# ----------------- defaults (overridden by env) -----------------
INPUT_FORMAT       = "lammps-data"      # "poscar" | "lammps-data" | "lammps-dump"
POSCAR_PATH        = "CONTCAR"
LAMMPS_DATA_PATH   = "relax.data"
LAMMPS_DATA_STYLE  = "atomic"
LAMMPS_DUMP_PATH   = "relax.dump"
LAMMPS_DUMP_LAST_FRAME = True
TYPE_ID_TO_ELEM    = {1: "Ga", 2: "O"}

MODEL_PATH         = "nep.txt"          # NEP model text file
SUPERCELL_DIM      = [1, 1, 1]
DISP_AMPLITUDE     = 0.01
TYPE_MAP_MASTER    = ["Ga", "O"]
OUTPUT_MODE        = "force_constants"  # or "force_sets"

# Optional controls for NEP backend
NEP_BACKEND        = "auto"             # auto | pynep | lammps | lmp_subprocess
LMP_BIN            = "lmp"              # used for subprocess backend
NEP_PAIR_STYLE     = "nep"              # pair_style name in LAMMPS

# ----------------- helpers to read env -----------------
def _getenv_bool(name, default):
    v = os.getenv(name, None)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _maybe_json(s, default):
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default

INPUT_FORMAT      = os.getenv("DP_INPUT_FORMAT", INPUT_FORMAT)
POSCAR_PATH       = os.getenv("DP_POSCAR_PATH", POSCAR_PATH)
LAMMPS_DATA_PATH  = os.getenv("DP_LAMMPS_DATA_PATH", LAMMPS_DATA_PATH) or LAMMPS_DATA_PATH
LAMMPS_DATA_STYLE = os.getenv("DP_LAMMPS_DATA_STYLE", LAMMPS_DATA_STYLE)
LAMMPS_DUMP_PATH  = os.getenv("DP_LAMMPS_DUMP_PATH", LAMMPS_DUMP_PATH)
LAMMPS_DUMP_LAST_FRAME = _getenv_bool("DP_LAMMPS_DUMP_LAST_FRAME", LAMMPS_DUMP_LAST_FRAME)

TYPE_ID_TO_ELEM_S = os.getenv("DP_TYPE_ID_TO_ELEM", "")
if TYPE_ID_TO_ELEM_S:
    _tmp = _maybe_json(TYPE_ID_TO_ELEM_S, {})
    TYPE_ID_TO_ELEM = {int(k): str(v) for k, v in _tmp.items()} if _tmp else TYPE_ID_TO_ELEM

MODEL_PATH        = os.getenv("DP_MODEL_PATH", MODEL_PATH)

DIM_ENV = os.getenv("DP_SUPERCELL_DIM", "")
if DIM_ENV.strip():
    SUPERCELL_DIM = [int(x) for x in DIM_ENV.split()]

DISP_AMPLITUDE    = float(os.getenv("DP_DISP_AMPLITUDE", str(DISP_AMPLITUDE)))

TMAP_ENV = os.getenv("DP_TYPE_MAP_MASTER", "")
if TMAP_ENV.strip():
    TYPE_MAP_MASTER = [t.strip() for t in TMAP_ENV.split(",") if t.strip()]

OUTPUT_MODE       = os.getenv("DP_OUTPUT_MODE", OUTPUT_MODE)

NEP_BACKEND       = os.getenv("DP_NEP_BACKEND", os.getenv("NEP_BACKEND", NEP_BACKEND)).strip().lower()
LMP_BIN           = os.getenv("DP_LMP_BIN", LMP_BIN)
NEP_PAIR_STYLE    = os.getenv("DP_NEP_PAIR_STYLE", NEP_PAIR_STYLE)

# ----------------- structure readers -----------------
def read_poscar(path="POSCAR"):
    with open(path) as f:
        raw = [l.strip() for l in f if l.strip()]
    scale = float(raw[1])
    lat = np.array([[float(x) for x in raw[i].split()] for i in range(2, 5)], float) * scale

    def is_ints(tokens):
        try:
            _ = [int(t) for t in tokens]
            return True
        except Exception:
            return False

    if is_ints(raw[6].split()):
        elements = raw[5].split()
        numbers = [int(x) for x in raw[6].split()]
        ptr = 7
    else:
        try:
            numbers  = [int(x) for x in raw[5].split()]
            elements = raw[0].split() or [f"X{i+1}" for i in range(len(numbers))]
            ptr = 6
        except Exception:
            elements = raw[5].split()
            numbers = [int(x) for x in raw[6].split()]
            ptr = 7

    tag = raw[ptr].lower()
    if tag.startswith("s"):
        ptr += 1
        tag = raw[ptr].lower()
    direct = tag.startswith("d")
    ptr += 1

    nat = sum(numbers)
    coords = np.array([[float(x) for x in raw[ptr+i].split()[:3]] for i in range(nat)])
    pos = coords @ lat if direct else coords * scale

    symbols = []
    for el, n in zip(elements, numbers):
        symbols.extend([el] * n)

    return PhonopyAtoms(symbols=symbols, cell=lat, positions=pos)

def _ensure_ase():
    try:
        import ase  # noqa
        from ase.io import read  # noqa
    except Exception as e:
        raise SystemExit(
            "ASE is required for LAMMPS inputs + NEP force backend.\n"
            "Install in this env:\n"
            "    python -m pip install ase\n"
            f"Original import error: {e}"
        )

def read_lammps_data(path, style, type_id_to_elem):
    _ensure_ase()
    from ase.io import read as ase_read
    atoms = ase_read(path, format="lammps-data", style=style)
    types = None
    try:
        types = atoms.get_array("type")
    except Exception:
        pass

    symbols = atoms.get_chemical_symbols()
    if types is not None:
        mapped = []
        for t in types:
            if int(t) in type_id_to_elem:
                mapped.append(type_id_to_elem[int(t)])
            else:
                raise SystemExit(f"Type {t} not in TYPE_ID_TO_ELEM {type_id_to_elem}")
        symbols = mapped

    cell = np.asarray(atoms.cell.array, float)
    pos  = np.asarray(atoms.get_positions(), float)
    return PhonopyAtoms(symbols=symbols, cell=cell, positions=pos)

def read_lammps_dump(path, last_frame, type_id_to_elem):
    _ensure_ase()
    from ase.io import read as ase_read
    index = -1 if last_frame else 0
    atoms = ase_read(path, format="lammps-dump-text", index=index)

    types = None
    try:
        types = atoms.get_array("type")
    except Exception:
        pass

    symbols = atoms.get_chemical_symbols()
    if types is not None:
        mapped = []
        for t in types:
            if int(t) in type_id_to_elem:
                mapped.append(type_id_to_elem[int(t)])
            else:
                raise SystemExit(f"Type {t} not in TYPE_ID_TO_ELEM {type_id_to_elem}")
        symbols = mapped

    cell = np.asarray(atoms.cell.array, float)
    pos  = np.asarray(atoms.get_positions(), float)
    return PhonopyAtoms(symbols=symbols, cell=cell, positions=pos)

def load_structure():
    fmt = INPUT_FORMAT.lower()
    if fmt == "poscar":
        return read_poscar(POSCAR_PATH)
    if fmt == "lammps-data":
        return read_lammps_data(LAMMPS_DATA_PATH, LAMMPS_DATA_STYLE, TYPE_ID_TO_ELEM)
    if fmt == "lammps-dump":
        return read_lammps_dump(LAMMPS_DUMP_PATH, LAMMPS_DUMP_LAST_FRAME, TYPE_ID_TO_ELEM)
    raise SystemExit(f"Unknown DP_INPUT_FORMAT: {INPUT_FORMAT}")

# ----------------- NEP force backends -----------------
def _phonopyatoms_to_ase(pa: PhonopyAtoms):
    _ensure_ase()
    from ase import Atoms
    return Atoms(
        symbols=list(pa.symbols),
        positions=np.asarray(pa.positions, float),
        cell=np.asarray(pa.cell, float),
        pbc=True,
    )

def _try_pynep_forces(pa: PhonopyAtoms) -> np.ndarray:
    # PyNEP provides an ASE calculator: pynep.calculate.NEP(model_file=...)
    from pynep.calculate import NEP as NEPCalc  # type: ignore
    atoms = _phonopyatoms_to_ase(pa)
    atoms.calc = NEPCalc(MODEL_PATH)
    f = atoms.get_forces()
    return np.asarray(f, float)

def _try_lammps_python_forces(pa: PhonopyAtoms) -> np.ndarray:
    # Use python-lammps if available
    from lammps import lammps  # type: ignore
    _ensure_ase()
    from ase.io import write as ase_write

    nat = pa.get_number_of_atoms()
    elems = TYPE_MAP_MASTER[:]  # mapping LAMMPS type 1..N -> elements
    if not elems:
        raise RuntimeError("TYPE_MAP_MASTER is empty; needed for NEP/LAMMPS type ordering.")

    with tempfile.TemporaryDirectory(prefix="nep_fc2_") as td:
        td = Path(td)
        data_path = td / "sc.data"

        ase_atoms = _phonopyatoms_to_ase(pa)
        # IMPORTANT: enforce LAMMPS type order consistent with TYPE_MAP_MASTER
        ase_write(str(data_path), ase_atoms, format="lammps-data", atom_style="atomic", specorder=elems)

        lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
        try:
            lmp.command("clear")
            lmp.command("units metal")
            lmp.command("atom_style atomic")
            lmp.command("boundary p p p")
            lmp.command(f"read_data {data_path}")
            lmp.command(f"pair_style {NEP_PAIR_STYLE}")
            lmp.command(f"pair_coeff * * {MODEL_PATH} " + " ".join(elems))
            lmp.command("thermo 0")
            lmp.command("run 0")

            # Extract forces (nat x 3)
            f_ptr = lmp.extract_atom("f", 3)
            forces = np.zeros((nat, 3), float)
            for i in range(nat):
                forces[i, 0] = f_ptr[i][0]
                forces[i, 1] = f_ptr[i][1]
                forces[i, 2] = f_ptr[i][2]
            return forces
        finally:
            try:
                lmp.close()
            except Exception:
                pass

def _parse_dump_forces(dump_path: Path, nat: int) -> np.ndarray:
    txt = dump_path.read_text().splitlines()
    # Find last "ITEM: ATOMS"
    idx = None
    for i in range(len(txt) - 1, -1, -1):
        if txt[i].startswith("ITEM: ATOMS"):
            idx = i
            break
    if idx is None:
        raise RuntimeError(f"Could not find 'ITEM: ATOMS' in {dump_path}")

    header = txt[idx].split()
    cols = header[2:]  # after ITEM: ATOMS
    # Expect id fx fy fz at least
    col_id = cols.index("id")
    col_fx = cols.index("fx")
    col_fy = cols.index("fy")
    col_fz = cols.index("fz")

    forces = np.zeros((nat, 3), float)
    lines = txt[idx + 1 : idx + 1 + nat]
    if len(lines) < nat:
        raise RuntimeError(f"Dump {dump_path} has only {len(lines)} atom lines; expected {nat}")

    for line in lines:
        parts = line.split()
        aid = int(float(parts[col_id]))
        forces[aid - 1, 0] = float(parts[col_fx])
        forces[aid - 1, 1] = float(parts[col_fy])
        forces[aid - 1, 2] = float(parts[col_fz])
    return forces

def _lammps_subprocess_forces(pa: PhonopyAtoms) -> np.ndarray:
    _ensure_ase()
    from ase.io import write as ase_write

    nat = pa.get_number_of_atoms()
    elems = TYPE_MAP_MASTER[:]
    if not elems:
        raise RuntimeError("TYPE_MAP_MASTER is empty; needed for NEP/LAMMPS type ordering.")
    if not Path(LMP_BIN).exists():
        raise RuntimeError(f"DP_LMP_BIN not found: {LMP_BIN}")

    with tempfile.TemporaryDirectory(prefix="nep_fc2_") as td:
        td = Path(td)
        data_path = td / "sc.data"
        dump_path = td / "forces.dump"
        in_path = td / "in.force"

        ase_atoms = _phonopyatoms_to_ase(pa)
        ase_write(str(data_path), ase_atoms, format="lammps-data", atom_style="atomic", specorder=elems)

        in_path.write_text(
            "\n".join(
                [
                    "units metal",
                    "atom_style atomic",
                    "boundary p p p",
                    f"read_data {data_path.name}",
                    f"pair_style {NEP_PAIR_STYLE}",
                    f"pair_coeff * * {Path(MODEL_PATH).name} " + " ".join(elems),
                    "thermo 0",
                    "run 0",
                    f"write_dump all custom {dump_path.name} id fx fy fz",
                    "",
                ]
            )
        )

        # Copy model file into workdir so pair_coeff path is simple and robust
        model_src = Path(MODEL_PATH).resolve()
        if not model_src.exists():
            raise RuntimeError(f"NEP model file not found: {MODEL_PATH}")
        (td / model_src.name).write_bytes(model_src.read_bytes())

        cmd = [LMP_BIN, "-in", in_path.name]
        subprocess.check_call(cmd, cwd=str(td))
        if not dump_path.exists():
            raise RuntimeError("LAMMPS did not write forces.dump")
        return _parse_dump_forces(dump_path, nat)

def nep_eval_forces(pa: PhonopyAtoms) -> np.ndarray:
    # Auto backend selection
    backend = NEP_BACKEND
    if backend not in ("auto", "pynep", "lammps", "lmp_subprocess"):
        raise SystemExit(f"Unknown NEP_BACKEND={backend}")

    if backend in ("auto", "pynep"):
        try:
            return _try_pynep_forces(pa)
        except Exception as e:
            if backend == "pynep":
                raise
            print(f"[nep][auto] PyNEP not available / failed: {e}")

    if backend in ("auto", "lammps"):
        try:
            return _try_lammps_python_forces(pa)
        except Exception as e:
            if backend == "lammps":
                raise
            print(f"[nep][auto] python-lammps not available / failed: {e}")

    # Fallback
    return _lammps_subprocess_forces(pa)

# ----------------- phonopy dataset helpers -----------------
def get_dataset_and_supercells(phonon: Phonopy):
    dataset = getattr(phonon, "get_displacement_dataset", lambda: None)() or getattr(phonon, "dataset", None)
    if dataset is None:
        raise AttributeError("phonopy displacement dataset API not found.")
    supercells = getattr(phonon, "get_supercells_with_displacements", lambda: None)() or getattr(
        phonon, "supercells_with_displacements", None
    )
    if supercells is None:
        raise AttributeError("phonopy API for supercells not found.")
    return dataset, supercells

# ----------------- main -----------------
def main():
    print("=== FC2 via NEP (env-driven; DeepMD removed) ===")
    print(f"Input format: {INPUT_FORMAT}")
    print(f"POSCAR_PATH: {POSCAR_PATH}")
    print(f"DATA_PATH:   {LAMMPS_DATA_PATH} (style={LAMMPS_DATA_STYLE})")
    print(f"DUMP_PATH:   {LAMMPS_DUMP_PATH} (last={LAMMPS_DUMP_LAST_FRAME})")
    print(f"TYPE_ID_MAP: {TYPE_ID_TO_ELEM}")
    print(f"NEP model:   {MODEL_PATH}")
    print(f"Dim:         {SUPERCELL_DIM}")
    print(f"Amplitude:   {DISP_AMPLITUDE}")
    print(f"Type map:    {TYPE_MAP_MASTER}")
    print(f"Backend:     {NEP_BACKEND}  (LMP_BIN={LMP_BIN})")
    print(f"Output:      {OUTPUT_MODE}")

    unit = load_structure()
    sc_mat = np.diag(SUPERCELL_DIM)
    phonon = Phonopy(unitcell=unit, supercell_matrix=sc_mat)
    phonon.generate_displacements(distance=DISP_AMPLITUDE)

    dataset, supercells = get_dataset_and_supercells(phonon)

    forces_all = []
    for i, sc in enumerate(supercells, start=1):
        f = nep_eval_forces(sc)
        forces_all.append(f)
        rms = np.sqrt((f**2).sum() / f.size)
        print(f"[{i:4d}/{len(supercells)}] RMS(force) = {rms:.4e} eV/Å")

    if OUTPUT_MODE == "force_sets":
        fs = {"natom": unit.get_number_of_atoms(), "first_atoms": []}
        for (fa, f) in zip(dataset["first_atoms"], forces_all):
            fs["first_atoms"].append(
                {
                    "number": int(fa["number"]),
                    "displacement": [float(x) for x in fa["displacement"]],
                    "forces": f.tolist(),
                }
            )
        write_FORCE_SETS(fs, filename="FORCE_SETS")
        print("✅ Wrote FORCE_SETS")
        return

    phonon.produce_force_constants(forces=forces_all)
    fc2 = phonon.force_constants
    if fc2 is None:
        raise SystemExit("phonon.force_constants is None — FC build failed.")
    write_FORCE_CONSTANTS(fc2, filename="FORCE_CONSTANTS")
    print("✅ Wrote FORCE_CONSTANTS")

if __name__ == "__main__":
    main()
