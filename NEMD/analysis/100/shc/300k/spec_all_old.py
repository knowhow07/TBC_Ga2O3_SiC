#!/usr/bin/env python3
import os
import sys
import time
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------- user knobs ----------------
ROOT = "."                      # current folder containing shc_g3/, shc_g4/, shc_g5/
RESULTS_DIR = "results"

SKIP_FRACTION = 0.2            # use last (1-skip) as steady state
M = 7                          # number of temperature groups output by compute keyword
G_LEFT = 2                     # left dumb group id used for DeltaT
G_RIGHT = 6                    # right dumb group id used for DeltaT
Nc = 1000
num_omega = 4000
SMOOTH_W = 71                  # moving average window (odd)

GROUPS_TO_PROCESS = [3, 4, 5]

# file names inside each shc_g* folder
COMPUTE = "compute.out"
SHC = "shc.out"
MODEL = "model.xyz"

# plotting
FIG_DPI = 300
# -------------------------------------------

LOG_FILE = None


class Tee:
    """Duplicate writes to multiple streams (e.g., terminal + log file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def ensure_results_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_group_x_range(model_xyz: str, group_id: int):
    """
    Return (xmin, xmax, xmean) for atoms in a given group_id from model.xyz.
    Assumes format:
      id species x y z group
    and group id is the last column.
    """
    with open(model_xyz, "r") as f:
        n_atoms = int(f.readline().strip())
        _ = f.readline()
        xs = []
        for _ in range(n_atoms):
            parts = f.readline().split()
            if len(parts) < 6:
                continue
            x = float(parts[2])
            g = int(parts[-1])
            if g == group_id:
                xs.append(x)

    if len(xs) == 0:
        raise RuntimeError(f"No atoms found for group {group_id} in {model_xyz}")

    xs = np.asarray(xs, float)
    return float(xs.min()), float(xs.max()), float(xs.mean())


def read_LyLz_area_A2(model_xyz: str) -> float:
    with open(model_xyz, "r") as f:
        _ = f.readline()
        header = f.readline().strip()

    m = re.search(r'Lattice="([^"]+)"', header)
    if m is None:
        raise RuntimeError(f"Cannot find Lattice in {model_xyz} header line 2")

    lat = np.fromstring(m.group(1), sep=" ")
    if lat.size != 9:
        raise RuntimeError(f"Expected 9 lattice numbers in {model_xyz}")

    a1 = lat[0:3]
    a2 = lat[3:6]
    a3 = lat[6:9]

    # Cross section for transport along x is |a2 x a3|
    area = np.linalg.norm(np.cross(a2, a3))
    return float(area)


def moving_average(y, w):
    if w is None or w < 3:
        return y
    if w % 2 == 0:
        w += 1
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def read_group_temps_from_compute(compute_file: str, M: int) -> np.ndarray:
    """
    compute.out format for temperature:
      columns 1..M : group temperatures (K), left->right
      extra columns may follow (e.g. thermostat energies)
    We only need the first M columns.
    """
    data = np.loadtxt(compute_file)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < M:
        raise RuntimeError(f"{compute_file} has too few columns: {data.shape[1]} < M={M}")
    return data[:, :M]


def read_shc_omega_block(shc_file: str, num_omega: int):
    rows = []
    with open(shc_file) as f:
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

    A = np.array(rows, float)
    if A.size == 0:
        raise RuntimeError(f"No numeric rows parsed from {shc_file}")

    # keep only rows that look like omega rows (omega >= 0)
    mask = np.isfinite(A[:, 0]) & (A[:, 0] >= 0.0)
    W = A[mask]

    # take the LAST num_omega rows (omega-block is at the end)
    if W.shape[0] < num_omega:
        raise RuntimeError(
            f"Found only {W.shape[0]} non-negative omega rows in {shc_file}, "
            f"but need num_omega={num_omega}"
        )

    Jw = W[-num_omega:, :]
    omega = Jw[:, 0] / (2.0 * np.pi)
    Jin = Jw[:, 1]
    Jout = Jw[:, 2]

    good = np.isfinite(omega) & np.isfinite(Jin) & np.isfinite(Jout)
    omega, Jin, Jout = omega[good], Jin[good], Jout[good]

    if omega.size == 0:
        raise RuntimeError(f"Omega-block exists in {shc_file}, but Jin/Jout are all NaN")

    return omega, Jin, Jout


def integrate_trapz_cumulative(x, y):
    if len(x) == 0:
        return np.array([])
    if len(x) == 1:
        return np.array([0.0])
    cum = np.cumsum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
    cum = np.insert(cum, 0, 0.0)
    return cum


def analyze_one_group(group_id, folder, results_dir):
    compute_file = os.path.join(folder, COMPUTE)
    shc_file = os.path.join(folder, SHC)
    model_file = os.path.join(folder, MODEL)

    if not os.path.isfile(compute_file):
        raise FileNotFoundError(f"Missing {compute_file}")
    if not os.path.isfile(shc_file):
        raise FileNotFoundError(f"Missing {shc_file}")
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Missing {model_file}")

    print(f"\n===== Processing group {group_id} in {folder} =====")

    # ---- DeltaT from compute.out ----
    Tg = read_group_temps_from_compute(compute_file, M)
    n = Tg.shape[0]
    start = int(np.floor(n * SKIP_FRACTION))
    Tg_ss = Tg[start:, :]

    Tavg = np.nanmean(Tg_ss, axis=0)
    print("Steady-state group temperatures (K):")
    for i, t in enumerate(Tavg, start=1):
        print(f"  g{i}: {t:.6f}")

    T_left = Tavg[G_LEFT - 1]
    T_right = Tavg[G_RIGHT - 1]
    dT = T_left - T_right
    print(f"DeltaT = <Tg{G_LEFT}> - <Tg{G_RIGHT}> = {dT:.6f} K")

    if abs(dT) < 1e-12:
        raise RuntimeError("DeltaT is ~0; cannot normalize SHC")

    # ---- SHC spectrum ----
    omega, Jin, Jout = read_shc_omega_block(shc_file, num_omega)
    Jtot = Jin + Jout

    # Native "conductance-like" normalization
    G_like = Jtot / dT

    # normalize by area
    A_A2 = read_LyLz_area_A2(model_file)
    G_like_A = G_like / A_A2

    # smoothing
    w = SMOOTH_W
    if w is not None and w >= G_like_A.size:
        w = max(3, (G_like_A.size // 2) * 2 - 1)
        if w < 3:
            w = None
    Gs = moving_average(G_like_A, w)

    cum = integrate_trapz_cumulative(omega, Gs)

    xmin, xmax, xmean = read_group_x_range(model_file, group_id)
    print(f"SHC group {group_id} x-range (Å): {xmin:.3f} -> {xmax:.3f} (center ~ {xmean:.3f})")
    print(f"Area = {A_A2:.6f} Å^2")
    print(f"Omega points = {len(omega)}")

    # ---- write summary txt ----
    summary_txt = os.path.join(results_dir, f"group_{group_id}_summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"group_id = {group_id}\n")
        f.write(f"folder = {folder}\n")
        f.write(f"compute_file = {compute_file}\n")
        f.write(f"shc_file = {shc_file}\n")
        f.write(f"model_file = {model_file}\n")
        f.write(f"skip_fraction = {SKIP_FRACTION}\n")
        f.write(f"M = {M}\n")
        f.write(f"G_LEFT = {G_LEFT}\n")
        f.write(f"G_RIGHT = {G_RIGHT}\n")
        f.write(f"DeltaT_K = {dT:.12e}\n")
        f.write(f"Area_A2 = {A_A2:.12e}\n")
        f.write(f"xmin_A = {xmin:.12e}\n")
        f.write(f"xmax_A = {xmax:.12e}\n")
        f.write(f"xmean_A = {xmean:.12e}\n")
        f.write("Steady_state_group_temperatures_K:\n")
        for i, t in enumerate(Tavg, start=1):
            f.write(f"g{i} = {t:.12e}\n")

    # ---- save spectral data ----
    data_txt = os.path.join(results_dir, f"group_{group_id}_spectral_data.txt")
    header = (
        "omega_THz  Jin_native  Jout_native  Jsum_native  "
        "G_like_native  G_like_per_A2_native  G_like_per_A2_smoothed  cumulative_integral_native"
    )
    np.savetxt(
        data_txt,
        np.column_stack([omega, Jin, Jout, Jtot, G_like, G_like_A, Gs, cum]),
        header=header
    )

    # ---- original figures (kept) ----
    plt.figure()
    plt.plot(omega, Jtot, label="J_in + J_out (raw)")
    plt.xlabel(r"Frequency ($\omega / 2\pi$) (THz)")
    plt.ylabel("J_q(ω) (from shc.out)")
    plt.title(f"Spectral Heat Current (group {group_id}, x: {xmin:.3f} -> {xmax:.3f} Å)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"group_{group_id}_shc_Jsum_S{SMOOTH_W}.png"), dpi=FIG_DPI)
    plt.close()

    plt.figure()
    plt.plot(omega, Gs, label=f"smoothed (w={SMOOTH_W})")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("G_like(ω) (native units) / Å²")
    plt.title(f"SHC normalized by ΔT and area (group {group_id})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"group_{group_id}_shc_Glike_S{SMOOTH_W}.png"), dpi=FIG_DPI)
    plt.close()

    plt.figure()
    plt.plot(omega, cum)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Cumulative ∫ G_like dω (native units)")
    plt.title(f"Cumulative spectral contribution (group {group_id})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"group_{group_id}_shc_Glike_cum_S{SMOOTH_W}.png"), dpi=FIG_DPI)
    plt.close()

    return {
        "group_id": group_id,
        "folder": folder,
        "omega": omega,
        "Jin": Jin,
        "Jout": Jout,
        "Jtot": Jtot,
        "G_like": G_like,
        "G_like_A": G_like_A,
        "Gs": Gs,
        "cum": cum,
        "dT": dT,
        "area_A2": A_A2,
        "xmin": xmin,
        "xmax": xmax,
        "xmean": xmean,
        "Tavg": Tavg,
    }


def make_combined_plots(results, results_dir):
    if len(results) == 0:
        return

    # Combined Jsum
    plt.figure()
    for r in results:
        plt.plot(r["omega"], r["Jtot"], label=f"g{r['group_id']}")
    plt.xlabel(r"Frequency ($\omega / 2\pi$) (THz)")
    plt.ylabel("J_q(ω) (from shc.out)")
    plt.title("Spectral Heat Current Comparison")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"combined_shc_Jsum_S{SMOOTH_W}.png"), dpi=FIG_DPI)
    plt.close()

    # Combined G_like
    plt.figure()
    for r in results:
        plt.plot(r["omega"], r["Gs"], label=f"g{r['group_id']}")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("G_like(ω) (native units) / Å²")
    plt.title("SHC normalized by ΔT and area: comparison")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"combined_shc_Glike_S{SMOOTH_W}.png"), dpi=FIG_DPI)
    plt.close()

    # Combined cumulative
    plt.figure()
    for r in results:
        plt.plot(r["omega"], r["cum"], label=f"g{r['group_id']}")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Cumulative ∫ G_like dω (native units)")
    plt.title("Cumulative spectral contribution: comparison")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"combined_shc_Glike_cum_S{SMOOTH_W}.png"), dpi=FIG_DPI)
    plt.close()

    # Summary table
    summary_file = os.path.join(results_dir, "combined_group_summary.txt")
    with open(summary_file, "w") as f:
        f.write("# group_id  x_min_A  x_max_A  x_center_A  DeltaT_K  area_A2  T_left_K  T_right_K\n")
        for r in results:
            f.write(
                f"{r['group_id']:d}  "
                f"{r['xmin']:.8f}  {r['xmax']:.8f}  {r['xmean']:.8f}  "
                f"{r['dT']:.8e}  {r['area_A2']:.8e}  "
                f"{r['Tavg'][G_LEFT-1]:.8f}  {r['Tavg'][G_RIGHT-1]:.8f}\n"
            )


def main():
    root = Path(ROOT).resolve()
    results_dir = root / RESULTS_DIR
    ensure_results_dir(results_dir)

    global LOG_FILE
    LOG_FILE = results_dir / f"spec_thermal_multi_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_fh = open(LOG_FILE, "w", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh)

    print(f"[LOG] Writing to {LOG_FILE}")
    print(f"[INFO] Root directory: {root}")
    print(f"[INFO] Results directory: {results_dir}")
    print(f"[INFO] Groups to process: {GROUPS_TO_PROCESS}")

    results = []

    for gid in GROUPS_TO_PROCESS:
        folder = root / f"shc_g{gid}"
        if not folder.is_dir():
            print(f"[WARN] Missing folder: {folder} ; skip group {gid}")
            continue

        try:
            res = analyze_one_group(gid, str(folder), str(results_dir))
            results.append(res)
        except Exception as e:
            print(f"[ERROR] Failed for group {gid}: {e}")

    if len(results) == 0:
        raise RuntimeError("No valid SHC group was processed successfully.")

    make_combined_plots(results, str(results_dir))

    print("\nSaved per-group figures:")
    for r in results:
        gid = r["group_id"]
        print(f"  group_{gid}_shc_Jsum_S{SMOOTH_W}.png")
        print(f"  group_{gid}_shc_Glike_S{SMOOTH_W}.png")
        print(f"  group_{gid}_shc_Glike_cum_S{SMOOTH_W}.png")
        print(f"  group_{gid}_summary.txt")
        print(f"  group_{gid}_spectral_data.txt")

    print("\nSaved combined figures:")
    print(f"  combined_shc_Jsum_S{SMOOTH_W}.png")
    print(f"  combined_shc_Glike_S{SMOOTH_W}.png")
    print(f"  combined_shc_Glike_cum_S{SMOOTH_W}.png")
    print(f"  combined_group_summary.txt")

    try:
        log_fh.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()