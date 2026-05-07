#!/usr/bin/env python3
"""
Per-folder TEST; global TRAIN/VAL split for NEP extended XYZ (with VIRIAL).

For each folder:
  - Read all frames (OUTCAR/vasprun.xml).
  - Pick a per-folder TEST subset (e.g., 10%) -> write tests/<tag>.xyz
  - Remaining frames go to a GLOBAL pool.

Then:
  - Shuffle global pool and split into TRAIN / VAL (e.g., 80/20).
  - Write train.xyz and val.xyz.

Also emits manifest.csv with detailed counts.

Units/format:
  lattice (Å), energy (eV), forces (eV/Å), virial (eV) with order:
  vxx vxy vxz vyx vyy vyz vzx vzy vzz
  virial = -stress(eV/Å^3) * volume(Å^3)

Requires: ASE
"""

import os
import re
import csv
import glob
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from ase.io import read
from ase.atoms import Atoms
import datetime
from collections import defaultdict


# ------------------- EDIT DEFAULTS -------------------
ROOTS = [
    # e.g.
    # "<PATH_TO_DATA>/qe_jobs/single_step/SiC/displacement/more/small",
    # "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/md-300k",
    # "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/md-450k"
    # "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/md-600k",
    # "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/displace/more/small",
    # "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/melt/output/job*"
]
ROOT_LABELS = [
#     "SiC-dis",
#     "Ga2O3-dis",
]

ROOT_GROUPS = [
    {
        "label": "Ga2O3-scale",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-0.960",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-0.970",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-0.980",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-0.990",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-1.000",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-1.010",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-1.020",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-1.030",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-1.040",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale2/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-0.960",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale2/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-0.970",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale2/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-0.980",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale2/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-0.990",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale2/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-1.000",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale2/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-1.010",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale2/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-1.020",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale2/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-1.030",
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/scale2/POSCAR.01x03x02/01.scale_pert/sys-0048-0072/scale-1.040",

        ],
    },
    {
        "label": "Ga2O3-md300-1000k",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/md-300to1000/job_1",
        ],
    },
    {
        "label": "Ga2O3-md300k",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/md-300/job_1",
        ],
    },
    {
        "label": "Ga2O3-md600k",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/md-600/job_1",
        ],
    },
    {
        "label": "Ga2O3-md900k",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/Ga2O3/sym/md-900/job_1",
        ],
    },
    {
        "label": "SiC-scale",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/SiC/sym/scale/CONTCAR_finerelax_mp.03x03x02/01.scale_pert/sys-0072-0072/scale-0.960",
            "<PATH_TO_DATA>/qe_jobs/single_step/SiC/sym/scale/CONTCAR_finerelax_mp.03x03x02/01.scale_pert/sys-0072-0072/scale-0.970",
            "<PATH_TO_DATA>/qe_jobs/single_step/SiC/sym/scale/CONTCAR_finerelax_mp.03x03x02/01.scale_pert/sys-0072-0072/scale-0.980",
            "<PATH_TO_DATA>/qe_jobs/single_step/SiC/sym/scale/CONTCAR_finerelax_mp.03x03x02/01.scale_pert/sys-0072-0072/scale-0.990",
            "<PATH_TO_DATA>/qe_jobs/single_step/SiC/sym/scale/CONTCAR_finerelax_mp.03x03x02/01.scale_pert/sys-0072-0072/scale-1.000",
            "<PATH_TO_DATA>/qe_jobs/single_step/SiC/sym/scale/CONTCAR_finerelax_mp.03x03x02/01.scale_pert/sys-0072-0072/scale-1.010",
            "<PATH_TO_DATA>/qe_jobs/single_step/SiC/sym/scale/CONTCAR_finerelax_mp.03x03x02/01.scale_pert/sys-0072-0072/scale-1.020",
            "<PATH_TO_DATA>/qe_jobs/single_step/SiC/sym/scale/CONTCAR_finerelax_mp.03x03x02/01.scale_pert/sys-0072-0072/scale-1.030",
            "<PATH_TO_DATA>/qe_jobs/single_step/SiC/sym/scale/CONTCAR_finerelax_mp.03x03x02/01.scale_pert/sys-0072-0072/scale-1.040",

        ],
    },
    {
        "label": "SiC-md300-1000k",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/SiC/sym/md-300to1000/job_1",
        ],
    },
    {
        "label": "SiC-md300k",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/SiC/sym/md300k/job_1",
        ],
    },
    {
        "label": "SiC-md600k",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/SiC/sym/md600k/job_1",
        ],
    },
    {
        "label": "SiC-md900k",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/SiC/sym/md900k/job_1",
        ],
    },
    {
        "label": "100-168-md300k",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/100-168atoms/md300k/single",
        ],
    },
    {
        "label": "100-168-md600k",
        "paths": [
             "<PATH_TO_DATA>/qe_jobs/single_step/100-168atoms/md600k/single",
        ],
    },
    {
        "label": "100-168-md900k",
        "paths": [
             "<PATH_TO_DATA>/qe_jobs/single_step/100-168atoms/md900k/single/job_1",
        ],
    },
    {
        "label": "100-168-dis",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_168/jobs/sigma_0.010",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_168/jobs/sigma_0.015",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_168/jobs/sigma_0.020",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_168/jobs/sigma_0.025",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_168/jobs/sigma_0.030",
        ],
    },
    {
        "label": "100-84-md300k",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/100-bei/single/md300k",
        ],
    },
    {
        "label": "100-84-md450k",
        "paths": [
             "<PATH_TO_DATA>/qe_jobs/single_step/100-bei/single/md450k",
        ],
    },
    {
        "label": "100-84-md600k",
        "paths": [
             "<PATH_TO_DATA>/qe_jobs/single_step/100-bei/single/md600k",
        ],
    },
    {
        "label": "100-84-md900k",
        "paths": [
             "<PATH_TO_DATA>/qe_jobs/single_step/100-bei/single/md900k/job_1",
        ],
    },
    {
        "label": "100-84-random",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/100-bei/random/displacement/output",
            "<PATH_TO_DATA>/qe_jobs/single_step/100-bei/random/compress/output",
            "<PATH_TO_DATA>/qe_jobs/single_step/100-bei/random/strain/output",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_bei/jobs/sigma_0.010",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_bei/jobs/sigma_0.015",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_bei/jobs/sigma_0.020",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_bei/jobs/sigma_0.025",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_bei/jobs/sigma_0.030",
        ],
    },
        {
        "label": "100-84-melt",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/100-bei/Si-O-melt/job1",
        ],
    },
    {
        "label": "201-69-md300k",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/201-SiO-63atoms/md-300k",
        ],
    },
    {
        "label": "201-69-md450k",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/201-SiO-63atoms/md-450k",
        ],
    },
    {
        "label": "201-69-md600k",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/201-SiO-63atoms/md-600k",
        ],
    },
    {
        "label": "201-69-md900k",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/201-SiO-63atoms/md-900k",
        ],
    },
    {
        "label": "201-69-random",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/201-SiO-63atoms/random/compress/output",
            "<PATH_TO_DATA>/qe_jobs/single_step/201-SiO-63atoms/random/displacement/output",
            "<PATH_TO_DATA>/qe_jobs/single_step/201-SiO-63atoms/random/strain/output",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_201_69atoms/jobs/sigma_0.010",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_201_69atoms/jobs/sigma_0.015",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_201_69atoms/jobs/sigma_0.020",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_201_69atoms/jobs/sigma_0.025",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_201_69atoms/jobs/sigma_0.030",
        ],
    },
    {
        "label": "201-106-md300k",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/201-SiO-106atoms/md-300k",
        ],
    },
    {
        "label": "201-106-md600k",
        "paths": [
             "<PATH_TO_DATA>/qe_jobs/single_step/201-SiO-106atoms/md-600k",
        ],
    },
    {
        "label": "201-106-md900k",
        "paths": [
             "<PATH_TO_DATA>/qe_jobs/single_step/201-SiO-106atoms/md-900k",
        ],
    },
    {
        "label": "201_106-dis",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_201_106_atoms/jobs/sigma_0.010",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_201_106_atoms/jobs/sigma_0.015",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_201_106_atoms/jobs/sigma_0.020",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_201_106_atoms/jobs/sigma_0.025",
            "<PATH_TO_DATA>/qe_jobs_pre/inter_displace_201_106_atoms/jobs/sigma_0.030",
        ],
    },
    {
        "label": "201_106-melt",
        "paths": [
            "<PATH_TO_DATA>/qe_jobs/single_step/201-SiO-106atoms/m3g_novac/job1",
        ],
    },

]

OUT_DIR = "data"
TEST_RATIO_PER_FOLDER = 0.10   # 10% from each folder reserved for that folder's TEST
GLOBAL_TRAIN_RATIO = 0.80      # of the remaining global pool
GLOBAL_VAL_RATIO   = 0.20      # of the remaining global pool
SHUFFLE_SEED = 42
GLOB_PATTERN = "**"            # recurse
# -----------------------------------------------------

def find_run_dirs(root: str, pattern: str = "**") -> List[str]:
    cand = sorted(glob.glob(os.path.join(root, pattern), recursive=True))
    runs = []
    for d in cand:
        if not os.path.isdir(d):
            continue
        if os.path.isfile(os.path.join(d, "OUTCAR")) or os.path.isfile(os.path.join(d, "vasprun.xml")):
            runs.append(d)
    return runs

def read_all_images(run_dir: str) -> List[Atoms]:
    """Read ALL frames from OUTCAR or vasprun.xml; keep only those with energy+forces."""
    imgs: List[Atoms] = []
    for src in (os.path.join(run_dir, "OUTCAR"), os.path.join(run_dir, "vasprun.xml")):
        if os.path.isfile(src):
            try:
                obj = read(src, index=":")
                seq = obj if isinstance(obj, list) else ([obj] if obj is not None else [])
            except Exception:
                seq = []
            for a in seq:
                try:
                    _ = a.get_potential_energy()
                    _ = a.get_forces()
                    imgs.append(a)
                except Exception:
                    continue
            if imgs:
                break
    return imgs

def sanitize_tag(path: str) -> str:
    tag = re.sub(r"[^\w\-\.]+", "_", path.strip("/"))
    return tag[-200:]

def compute_virial_from_stress(atoms: Atoms) -> Optional[np.ndarray]:
    """virial (eV) = - stress(eV/Å^3) * volume(Å^3)"""
    try:
        s = atoms.get_stress(voigt=False)  # 3x3
        V = atoms.get_volume()
        return -s * V
    except Exception:
        return None

def xyz_comment(cell3x3: np.ndarray, energy: float, virial3x3: Optional[np.ndarray]) -> str:
    lat = " ".join(f"{x:.10f}" for x in cell3x3.reshape(-1))
    parts = [
        f'lattice="{lat}"',
        f'Properties=species:S:1:pos:R:3:force:R:3',
        f'energy={energy:.10f}',
    ]
    if virial3x3 is not None:
        v = virial3x3
        parts.append(
            'virial="' +
            " ".join(f"{x:.10f}" for x in [
                v[0,0], v[0,1], v[0,2],
                v[1,0], v[1,1], v[1,2],
                v[2,0], v[2,1], v[2,2],
            ]) +
            '"'
        )
    return " ".join(parts)
def expand_paths(paths: List[str]) -> List[str]:
    """Expand glob patterns in a list of paths (e.g., job[1-2])."""
    out = []
    for p in paths:
        hits = glob.glob(p)
        out.extend(hits if hits else [p])  # keep literal if no match
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq

def build_grouped_from_groups(groups_cfg: List[dict], glob_pat: str) -> List[dict]:
    """Build grouped runs from ROOT_GROUPS (label + multiple paths)."""
    grouped = []
    for g in groups_cfg:
        label = g["label"].strip()
        paths = expand_paths(g["paths"])
        runs = []
        for root in paths:
            for d in find_run_dirs(root, glob_pat):
                imgs = read_all_images(d)
                if imgs:
                    runs.append((d, sanitize_tag(d), imgs))
        if runs:
            grouped.append({"root": ",".join(paths), "root_tag": label, "runs": runs})
    return grouped

def build_grouped_from_roots(roots: List[str], labels: List[str], glob_pat: str) -> List[dict]:
    """Fallback: one root ↔ one label (original behavior)."""
    grouped = []
    for i, root in enumerate(roots, start=1):
        label = labels[i-1].strip() if (i <= len(labels) and labels[i-1].strip()) else f"{i}"
        runs = []
        for d in find_run_dirs(root, glob_pat):
            imgs = read_all_images(d)
            if imgs:
                runs.append((d, sanitize_tag(d), imgs))
        if runs:
            grouped.append({"root": root, "root_tag": label, "runs": runs})
    return grouped

def write_frame(fh, atoms: Atoms):
    pos = atoms.get_positions()
    cell = atoms.get_cell().array
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    virial = compute_virial_from_stress(atoms)

    fh.write(f"{len(atoms)}\n")
    fh.write(xyz_comment(cell, energy, virial) + "\n")
    for sym, r, F in zip(atoms.get_chemical_symbols(), pos, forces):
        fh.write(f"{sym} {r[0]:.10f} {r[1]:.10f} {r[2]:.10f} {F[0]:.10f} {F[1]:.10f} {F[2]:.10f}\n")

def main():
    ap = argparse.ArgumentParser(description="Per-folder TEST; global TRAIN/VAL split (with VIRIAL).")
    ap.add_argument("--roots", nargs="*", help="Override ROOTS.")
    ap.add_argument("--out", default=OUT_DIR, help="Output directory.")
    ap.add_argument("--test-per-folder", type=float, default=TEST_RATIO_PER_FOLDER, help="Per-folder TEST ratio.")
    ap.add_argument("--train-ratio", type=float, default=GLOBAL_TRAIN_RATIO, help="Global TRAIN ratio of remaining pool.")
    ap.add_argument("--val-ratio", type=float, default=GLOBAL_VAL_RATIO, help="Global VAL ratio of remaining pool.")
    ap.add_argument("--seed", type=int, default=SHUFFLE_SEED, help="Shuffle seed.")
    ap.add_argument("--glob", default=GLOB_PATTERN, help='Glob pattern (default: "**").')
    args = ap.parse_args()
    # --- Sanity check: ensure all configured paths exist ---
    print("=== Path sanity check ===")

    missing_paths = []  # collect missing or unmatched ones

    def check_path(p: str) -> bool:
        """Check existence or pattern match for a path; return True if ok."""
        if any(ch in p for ch in "*?["):
            hits = glob.glob(p)
            if not hits:
                print(f"⚠️  Pattern not matched: {p}")
                missing_paths.append(p)
                return False
            for h in hits:
                if not os.path.exists(h):
                    print(f"⚠️  Missing expanded path: {h}")
                    missing_paths.append(h)
                    return False
            return True
        else:
            if not os.path.exists(p):
                print(f"⚠️  Missing path: {p}")
                missing_paths.append(p)
                return False
            return True
    roots = args.roots if args.roots else ROOTS

    if ROOT_GROUPS:
        for g in ROOT_GROUPS:
            print(f"[{g['label']}]")
            for p in g["paths"]:
                check_path(p)
    else:
        for p in roots:
            check_path(p)

    if missing_paths:
        print("\n❌ The following paths are invalid or unmatched:")
        for p in missing_paths:
            print(f"   - {p}")
        print("\nPlease fix them before running.\n")
        exit(1)
    else:
        print("✅ All root paths exist. Proceeding...\n")

    
    out_dir = Path(args.out).resolve()
    tests_dir = out_dir / "tests"         # per-folder test
    out_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    # ------------------- logging -------------------
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"prep_nep_{ts}.log"
    log_fh = open(log_path, "w")

    def log_print(msg: str = ""):
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    # counts[root_tag]["test"/"train"/"val"] = number of frames
    counts = defaultdict(lambda: {"test": 0, "train": 0, "val": 0})


    # Collect frames per run-folder (leaf dirs containing OUTCAR/vasprun.xml)

    # === Build groups ===
    # If ROOT_GROUPS is non-empty, it OVERRIDES ROOTS/ROOT_LABELS.
    if ROOT_GROUPS:
        grouped = build_grouped_from_groups(ROOT_GROUPS, args.glob)
    else:
        grouped = build_grouped_from_roots(roots, ROOT_LABELS, args.glob)

    if not grouped:
        log_print("No folders with valid frames found after grouping.")
        return

    # Show mapping summary
    log_print("=== Group label → source roots ===")
    for g in grouped:
        log_print(f"{g['root_tag']:>12s}  ←  {g['root']}")
    log_print()



    # === ROOT-LEVEL (A/B/...) TEST split; remaining -> global pool ===
    # Interpret each ROOT in --roots (or ROOTS list) as a category (A, B, ...)
    random.seed(args.seed)
    global_pool = []   # (root_tag, folder_path, run_tag, atoms)
    manifest_rows = []


    # 2) per-root selection: choose some run dirs as TEST and write ONE file per root
    tests_dir = out_dir / "tests"   # keep same folder name
    tests_dir.mkdir(parents=True, exist_ok=True)

    for _, g in enumerate(grouped, start=1):
        root_tag = g["root_tag"]       # <-- use the label from the grouped entry
        runs = g["runs"]


        run_indices = list(range(len(runs)))
        random.shuffle(run_indices)

        # use args.test_per_folder as FRACTION OF RUNS within this root
        n_test_runs = int(round(len(runs) * args.test_per_folder))
        if n_test_runs < 0:
            n_test_runs = 0
        if n_test_runs > len(runs):
            n_test_runs = len(runs)

        test_run_set = set(run_indices[:n_test_runs])

        # open a SINGLE test file per root IF there are any test runs selected
        test_path = ""
        ftest = None
        if n_test_runs > 0:
            test_path = str((tests_dir / f"{root_tag}.xyz").resolve())
            ftest = open(test_path, "w")

        # iterate runs; write ALL frames from selected test runs to that SINGLE file
        for i, (folder, run_tag, imgs) in enumerate(runs):
            if i in test_run_set:
                # write all frames from this run into the root-level test file
                for a in imgs:
                    write_frame(ftest, a)
                counts[root_tag]["test"] += len(imgs)
                manifest_rows.append({
                    "root": g["root"],
                    "root_tag": root_tag,
                    "folder": folder,
                    "run_tag": run_tag,
                    "frames_total": len(imgs),
                    "split": "test(root)",
                    "per_root_test_xyz": test_path,
                })
            else:
                # non-test runs go entirely to the global pool (for train/val later)
                for a in imgs:
                    global_pool.append((root_tag, folder, run_tag, a))
                manifest_rows.append({
                    "root": g["root"],
                    "root_tag": root_tag,
                    "folder": folder,
                    "run_tag": run_tag,
                    "frames_total": len(imgs),
                    "split": "pool",
                    "per_root_test_xyz": "",
                })

        # --- summary log for this group ---
        if ftest is not None:
            ftest.close()
            test_count = sum(len(imgs) for i, (_, _, imgs) in enumerate(runs) if i in test_run_set)
            log_print(f"[{root_tag}] test: {test_count:6d} frames → {os.path.basename(test_path)}")
        else:
            log_print(f"[{root_tag}] test:      0 frames (all to global pool)")



    total_pool = len(global_pool)
    if total_pool == 0:
        log_print("After per-folder test selection, global pool is empty. Reduce --test-per-folder.")
        return

    # Global TRAIN/VAL split on the pool
    # Normalize requested ratios (just in case they don't sum to 1)
    tv_sum = args.train_ratio + args.val_ratio
    train_ratio = args.train_ratio / tv_sum
    val_ratio = args.val_ratio / tv_sum

    idxs = list(range(total_pool))
    random.shuffle(idxs)

    n_train = int(round(total_pool * train_ratio))
    n_val   = total_pool - n_train

    train_xyz = out_dir / "train.xyz"
    val_xyz   = out_dir / "val.xyz"
    for p in (train_xyz, val_xyz):
        if p.exists(): p.unlink()

    with open(train_xyz, "a") as ftr, open(val_xyz, "a") as fva:
        for i, idx in enumerate(idxs):
            root_tag, folder, run_tag, atoms = global_pool[idx]
            if i < n_train:
                write_frame(ftr, atoms)
                counts[root_tag]["train"] += 1
            else:
                write_frame(fva, atoms)
                counts[root_tag]["val"] += 1



    # Write manifest
    with open(out_dir / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["root","root_tag","folder","run_tag","frames_total","split","per_root_test_xyz"],
        )
        w.writeheader()
        w.writerows(manifest_rows)


    log_print("\n=== DONE ===")
    log_print(f"train: {train_xyz}   (~{n_train} frames)")
    log_print(f"val  : {val_xyz}     (~{n_val} frames)")
    log_print(f"Per-folder test files in: {tests_dir}")
    log_print(f"Manifest: {out_dir / 'manifest.csv'}")
    log_print(f"Log: {log_path}")

    log_print("\n=== Frame counts by label ===")
    labels = sorted(counts.keys())
    for lab in labels:
        c = counts[lab]
        total = c["train"] + c["val"] + c["test"]
        log_print(f"{lab:>18s}  train={c['train']:7d}  val={c['val']:7d}  test={c['test']:7d}  total={total:7d}")

    log_fh.close()


if __name__ == "__main__":
    main()
