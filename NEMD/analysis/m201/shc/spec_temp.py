#!/usr/bin/env python3
import os
import sys
import time
import re
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# User settings
# ============================================================
# Customize these paths
# TEMP_PATHS = {
#     "300K": "./300k",
#     "600K": "./600k-fix",
#     "900K": "./900k",
# }

TEMP_PATHS = {
    "20+10": "./300k",
    "60+30": "./60+30",
}

GROUPS_TO_PROCESS = [3, 4, 5]

RESULTS_DIR = "results_length"

# ---- analysis knobs ----
SKIP_FRACTION = 0.2
M = 7                  # number of temperature groups in compute.out
G_LEFT = 2             # for DeltaT = <Tg2> - <Tg6>
G_RIGHT = 6
NUM_OMEGA = 4000
SMOOTH_W = 101          # moving-average window for G_like
FIG_DPI = 300

# file names inside each shc_g* folder
COMPUTE = "compute.out"
SHC = "shc.out"
MODEL = "model.xyz"

# ============================================================
# Logging helper
# ============================================================
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ============================================================
# Basic helpers
# ============================================================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def moving_average(y, w):
    if w is None or w < 3:
        return y.copy()
    if w % 2 == 0:
        w += 1
    if len(y) < w:
        w = max(3, (len(y) // 2) * 2 - 1)
        if w < 3:
            return y.copy()
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def integrate_trapz_cumulative(x, y):
    if len(x) == 0:
        return np.array([])
    if len(x) == 1:
        return np.array([0.0])
    cum = np.cumsum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
    return np.insert(cum, 0, 0.0)


# ============================================================
# File readers
# ============================================================
def read_group_temps_from_compute(compute_file: str, M: int) -> np.ndarray:
    data = np.loadtxt(compute_file)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < M:
        raise RuntimeError(f"{compute_file} has too few columns: {data.shape[1]} < M={M}")
    return data[:, :M]


def read_shc_omega_block(shc_file: str, num_omega: int):
    rows = []
    with open(shc_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    arr = np.array(rows, dtype=float)
    if arr.size == 0:
        raise RuntimeError(f"No numeric rows parsed from {shc_file}")

    mask = np.isfinite(arr[:, 0]) & (arr[:, 0] >= 0.0)
    arr = arr[mask]

    if arr.shape[0] < num_omega:
        raise RuntimeError(
            f"Only {arr.shape[0]} usable omega rows found in {shc_file}, "
            f"but NUM_OMEGA={num_omega}"
        )

    arr = arr[-num_omega:, :]
    omega = arr[:, 0] / (2.0 * np.pi)   # convert to omega/2pi in THz
    Jin = arr[:, 1]
    Jout = arr[:, 2]

    good = np.isfinite(omega) & np.isfinite(Jin) & np.isfinite(Jout)
    omega = omega[good]
    Jin = Jin[good]
    Jout = Jout[good]

    if len(omega) == 0:
        raise RuntimeError(f"All omega/J values are invalid in {shc_file}")

    return omega, Jin, Jout


def read_LyLz_area_A2(model_xyz: str) -> float:
    with open(model_xyz, "r") as f:
        _ = f.readline()
        header = f.readline().strip()

    m = re.search(r'Lattice="([^"]+)"', header)
    if m is None:
        raise RuntimeError(f"Cannot find Lattice in header of {model_xyz}")

    lat = np.fromstring(m.group(1), sep=" ")
    if lat.size != 9:
        raise RuntimeError(f"Expected 9 lattice numbers in {model_xyz}")

    a1 = lat[0:3]
    a2 = lat[3:6]
    a3 = lat[6:9]

    area = np.linalg.norm(np.cross(a2, a3))
    return float(area)


def read_group_x_range(model_xyz: str, group_id: int):
    with open(model_xyz, "r") as f:
        n_atoms = int(f.readline().strip())
        _ = f.readline()

        xs = []
        for _ in range(n_atoms):
            line = f.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) < 6:
                continue
            x = float(parts[2])
            gid = int(parts[-1])
            if gid == group_id:
                xs.append(x)

    if len(xs) == 0:
        raise RuntimeError(f"No atoms found for group {group_id} in {model_xyz}")

    xs = np.asarray(xs, dtype=float)
    return float(xs.min()), float(xs.max()), float(xs.mean())


# ============================================================
# One case analysis
# ============================================================
def analyze_one_case(temp_label: str, temp_root: str, group_id: int):
    folder = Path(temp_root) / f"shc_g{group_id}"
    if not folder.is_dir():
        warnings.warn(f"[{temp_label}] Missing folder: {folder}")
        return None

    compute_file = folder / COMPUTE
    shc_file = folder / SHC
    model_file = folder / MODEL

    for f in [compute_file, shc_file, model_file]:
        if not f.is_file():
            warnings.warn(f"[{temp_label}] Missing file: {f}")
            return None

    # ---- DeltaT ----
    Tg = read_group_temps_from_compute(str(compute_file), M)
    n = Tg.shape[0]
    start = int(np.floor(n * SKIP_FRACTION))
    Tg_ss = Tg[start:, :]
    Tavg = np.nanmean(Tg_ss, axis=0)

    T_left = Tavg[G_LEFT - 1]
    T_right = Tavg[G_RIGHT - 1]
    dT = T_left - T_right

    if abs(dT) < 1e-12:
        raise RuntimeError(f"[{temp_label}] DeltaT ~ 0 for group {group_id}")

    # ---- SHC ----
    omega, Jin, Jout = read_shc_omega_block(str(shc_file), NUM_OMEGA)
    Jsum = Jin + Jout

    G_like = Jsum / dT

    area = read_LyLz_area_A2(str(model_file))
    G_like_A = G_like / area
    G_like_A_smooth = moving_average(G_like_A, SMOOTH_W)
    G_like_cum = integrate_trapz_cumulative(omega, G_like_A_smooth)

    xmin, xmax, xmean = read_group_x_range(str(model_file), group_id)

    return {
        "temp_label": temp_label,
        "temp_root": str(temp_root),
        "group_id": group_id,
        "folder": str(folder),
        "compute_file": str(compute_file),
        "shc_file": str(shc_file),
        "model_file": str(model_file),
        "omega": omega,
        "Jin": Jin,
        "Jout": Jout,
        "Jsum": Jsum,
        "G_like": G_like,
        "G_like_A": G_like_A,
        "G_like_A_smooth": G_like_A_smooth,
        "G_like_cum": G_like_cum,
        "Tavg": Tavg,
        "dT": dT,
        "area_A2": area,
        "xmin": xmin,
        "xmax": xmax,
        "xmean": xmean,
    }


# ============================================================
# Plotting by group across temperatures
# ============================================================
def plot_group_jsum(group_id, case_list, results_dir):
    plt.figure()
    for c in case_list:
        plt.plot(c["omega"], c["Jsum"], label=c["temp_label"])
    plt.xlabel(r"Frequency ($\omega / 2\pi$) (THz)")
    plt.ylabel("J_in + J_out (native)")
    plt.title(f"Group {group_id}: Jsum at different temperatures")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(results_dir) / f"group_{group_id}_Jsum_vs_temperature.png", dpi=FIG_DPI)
    plt.close()


def plot_group_glike(group_id, case_list, results_dir):
    plt.figure()
    for c in case_list:
        plt.plot(c["omega"], c["G_like_A_smooth"], label=c["temp_label"])
    plt.xlabel("Frequency (THz)")
    plt.ylabel("G_like(ω) / Å² (smoothed)")
    plt.title(f"Group {group_id}: G_like at different temperatures")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(results_dir) / f"group_{group_id}_Glike_vs_temperature.png", dpi=FIG_DPI)
    plt.close()


def plot_group_glike_cum(group_id, case_list, results_dir):
    plt.figure()
    for c in case_list:
        plt.plot(c["omega"], c["G_like_cum"], label=c["temp_label"])
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Cumulative ∫ G_like dω (native)")
    plt.title(f"Group {group_id}: cumulative G_like at different temperatures")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(results_dir) / f"group_{group_id}_GlikeCum_vs_temperature.png", dpi=FIG_DPI)
    plt.close()


# ============================================================
# Output summaries
# ============================================================
def save_case_data(results_by_group, results_dir):
    for group_id, case_list in results_by_group.items():
        for c in case_list:
            out = Path(results_dir) / f"group_{group_id}_{c['temp_label']}_spectral_data.txt"
            header = (
                "omega_THz  Jin_native  Jout_native  Jsum_native  "
                "G_like_native  G_like_per_A2_native  "
                "G_like_per_A2_smoothed  cumulative_integral_native"
            )
            np.savetxt(
                out,
                np.column_stack([
                    c["omega"],
                    c["Jin"],
                    c["Jout"],
                    c["Jsum"],
                    c["G_like"],
                    c["G_like_A"],
                    c["G_like_A_smooth"],
                    c["G_like_cum"]
                ]),
                header=header
            )


def save_summary(results_by_group, results_dir):
    out = Path(results_dir) / "summary_by_group_and_temperature.txt"
    with open(out, "w") as f:
        f.write("# group  temp  dT_K  area_A2  x_min_A  x_max_A  x_center_A  T_left_K  T_right_K  folder\n")
        for group_id, case_list in results_by_group.items():
            for c in case_list:
                f.write(
                    f"{group_id:d}  {c['temp_label']:>6s}  "
                    f"{c['dT']:.8e}  {c['area_A2']:.8e}  "
                    f"{c['xmin']:.8f}  {c['xmax']:.8f}  {c['xmean']:.8f}  "
                    f"{c['Tavg'][G_LEFT-1]:.8f}  {c['Tavg'][G_RIGHT-1]:.8f}  "
                    f"{c['folder']}\n"
                )


# ============================================================
# Main
# ============================================================
def main():
    root = Path(".").resolve()
    results_dir = root / RESULTS_DIR
    ensure_dir(results_dir)

    log_file = results_dir / f"compare_temp_by_group_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_fh = open(log_file, "w", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh)

    print(f"[INFO] Current path   : {root}")
    print(f"[INFO] Results folder : {results_dir}")
    print(f"[INFO] Temperature paths:")
    for k, v in TEMP_PATHS.items():
        print(f"  {k} -> {Path(v).resolve()}")

    results_by_group = {gid: [] for gid in GROUPS_TO_PROCESS}

    for group_id in GROUPS_TO_PROCESS:
        print(f"\n===== Processing group {group_id} across temperatures =====")
        for temp_label, temp_root in TEMP_PATHS.items():
            try:
                case = analyze_one_case(temp_label, temp_root, group_id)
                if case is None:
                    continue
                results_by_group[group_id].append(case)
                print(
                    f"[OK] g{group_id} {temp_label}: "
                    f"dT={case['dT']:.6f} K, area={case['area_A2']:.6f} Å², "
                    f"x=({case['xmin']:.3f},{case['xmax']:.3f})"
                )
            except Exception as e:
                warnings.warn(f"[{temp_label}] group {group_id} failed: {e}")

        results_by_group[group_id].sort(key=lambda c: c["temp_label"])

    # save all spectral txt files
    save_case_data(results_by_group, results_dir)
    save_summary(results_by_group, results_dir)

    # make 3 figures for each group
    made_figures = []
    for group_id, case_list in results_by_group.items():
        if len(case_list) == 0:
            warnings.warn(f"No valid cases for group {group_id}; skip plotting.")
            continue

        plot_group_jsum(group_id, case_list, results_dir)
        plot_group_glike(group_id, case_list, results_dir)
        plot_group_glike_cum(group_id, case_list, results_dir)

        made_figures.extend([
            f"group_{group_id}_Jsum_vs_temperature.png",
            f"group_{group_id}_Glike_vs_temperature.png",
            f"group_{group_id}_GlikeCum_vs_temperature.png",
        ])

    print("\nSaved figures:")
    for name in made_figures:
        print(f"  {name}")

    print("\nSaved text outputs:")
    print("  summary_by_group_and_temperature.txt")
    print("  group_<gid>_<temp>_spectral_data.txt")
    print(f"  log: {log_file}")

    log_fh.close()


if __name__ == "__main__":
    main()