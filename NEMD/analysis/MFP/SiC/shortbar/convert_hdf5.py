#!/usr/bin/env python3
import re
from pathlib import Path
import numpy as np
import h5py

ROOT = Path(".")
JOB_GLOB = "job_*"

VEL_NAME = "velocity.out"
MODEL_NAME = "model.xyz"
OUT_NAME = "velocity_f32.h5"

COMPRESSION = "gzip"      # "gzip" or None
COMPRESSION_OPTS = 4      # 1-9 for gzip
DELETE_ORIGINAL = False   # set True only after verification
OVERWRITE = False         # overwrite existing h5 or skip


def read_natoms_from_model(model_path: Path) -> int:
    with open(model_path, "r") as f:
        first = f.readline().strip()
    try:
        return int(first)
    except ValueError:
        raise RuntimeError(f"Cannot parse natoms from first line of {model_path}")


def count_lines(file_path: Path) -> int:
    n = 0
    with open(file_path, "r") as f:
        for _ in f:
            n += 1
    return n


def count_columns(file_path: Path) -> int:
    with open(file_path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                return len(s.split())
    raise RuntimeError(f"Empty file: {file_path}")


def convert_velocity_to_hdf5(job_dir: Path):
    vel_path = job_dir / VEL_NAME
    model_path = job_dir / MODEL_NAME
    out_path = job_dir / OUT_NAME

    if not vel_path.is_file():
        print(f"[SKIP] {job_dir}: missing {VEL_NAME}")
        return
    if not model_path.is_file():
        print(f"[SKIP] {job_dir}: missing {MODEL_NAME}")
        return
    if out_path.exists() and not OVERWRITE:
        print(f"[SKIP] {job_dir}: {OUT_NAME} already exists")
        return

    natoms = read_natoms_from_model(model_path)
    ncol = count_columns(vel_path)
    if ncol != 3:
        raise RuntimeError(f"{vel_path} has {ncol} columns, expected 3")

    nlines = count_lines(vel_path)
    if nlines % natoms != 0:
        raise RuntimeError(
            f"{vel_path}: total lines = {nlines}, not divisible by natoms = {natoms}"
        )
    nframes = nlines // natoms

    print(f"[INFO] {job_dir}")
    print(f"       natoms  = {natoms}")
    print(f"       nframes = {nframes}")
    print(f"       output  = {out_path}")

    # read whole text file once; simplest and usually OK for one job at a time
    arr = np.loadtxt(vel_path, dtype=np.float32)
    if arr.shape != (nlines, 3):
        raise RuntimeError(f"Unexpected loaded shape for {vel_path}: {arr.shape}")

    arr = arr.reshape(nframes, natoms, 3)

    with h5py.File(out_path, "w") as h5:
        dset = h5.create_dataset(
            "velocity",
            data=arr,
            dtype=np.float32,
            chunks=(1, natoms, 3),
            compression=COMPRESSION,
            compression_opts=COMPRESSION_OPTS if COMPRESSION == "gzip" else None,
        )
        dset.attrs["units"] = "Angstrom/fs"
        dset.attrs["shape_meaning"] = "(nframe, natom, xyz)"
        dset.attrs["natoms"] = natoms
        dset.attrs["nframes"] = nframes
        dset.attrs["source_file"] = str(vel_path.name)

    size_txt = vel_path.stat().st_size / (1024**3)
    size_h5 = out_path.stat().st_size / (1024**3)
    print(f"       size txt = {size_txt:.2f} GB")
    print(f"       size h5  = {size_h5:.2f} GB")

    if DELETE_ORIGINAL:
        vel_path.unlink()
        print(f"       deleted original {vel_path}")


def main():
    job_dirs = sorted(
        [p for p in ROOT.glob(JOB_GLOB) if p.is_dir() and re.match(r"job_\d+$", p.name)]
    )

    if not job_dirs:
        print("No job_* folders found.")
        return

    print(f"Found {len(job_dirs)} job folders.\n")
    for job_dir in job_dirs:
        try:
            convert_velocity_to_hdf5(job_dir)
        except Exception as e:
            print(f"[ERROR] {job_dir}: {e}")
        print()


if __name__ == "__main__":
    main()