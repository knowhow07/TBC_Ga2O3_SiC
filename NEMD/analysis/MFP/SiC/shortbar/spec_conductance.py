#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
import sys
import time
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif"]

plt.rcParams.update({'mathtext.default': 'regular'})
mpl.rcParams["font.size"] = 22

# =========================
# USER KNOBS
# =========================
PATH_ROOT   = "<PATH_TO_DATA>/qe_jobs/nep_training/NEMD/0217/post/MFP/SiC/shortbar"
INPUT_GLOB  = os.path.join(PATH_ROOT, "job_*")

EV_PER_PS_A2_TO_MW_PER_M2 = 1.602176634e7

COMPUTE_NAME = "compute.out"
SHC_NAME     = "shc.out"
MODEL_NAME   = "model.xyz"

SKIP_FRACTION = 0.2      # use last (1 - SKIP_FRACTION) of compute.out as steady state
M = 5                    # number of temperature groups in compute.out
G_LEFT  = 2              # left temperature group id (1-based)
G_RIGHT = 4              # right temperature group id (1-based)

NUM_OMEGA = 4000         # from compute_shc
SHC_GROUP_ID = 3         # group used in compute_shc

SMOOTH_W = 21            # odd moving-average window for plotting only
PLOT_W_OVER_2PI = True  # True: plot/store freq = omega/(2*pi); False: raw omega
MAX_W_PLOT = None        # THz, or None

OUT_PREFIX = "bulk_SiC_shortbar"

# =========================
# LOGGING
# =========================
LOG_FILE = os.path.join(PATH_ROOT, f"{OUT_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}.log")

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

_log_fh = open(LOG_FILE, "w", buffering=1)
sys.stdout = Tee(sys.__stdout__, _log_fh)
sys.stderr = Tee(sys.__stderr__, _log_fh)

print(f"[LOG] Writing to {LOG_FILE}")

# plt.rcParams.update({
#     "font.size": 16,
#     "axes.labelsize": 16,
#     "axes.titlesize": 16,
#     "xtick.labelsize": 16,
#     "ytick.labelsize": 16,
#     "legend.fontsize": 16,
#     "figure.titlesize": 16,
# })
# =========================
# HELPERS
# =========================
def moving_average(y, w):
    if w is None or w < 3:
        return y.copy()
    if w % 2 == 0:
        w += 1
    if w >= len(y):
        w = max(3, (len(y) // 2) * 2 - 1)
    if w < 3:
        return y.copy()
    k = np.ones(w, dtype=float) / w
    return np.convolve(y, k, mode="same")

def collect_job_dirs(folder_glob):
    jobs = sorted(glob.glob(folder_glob))
    jobs = [d for d in jobs if os.path.isdir(d)]
    if not jobs:
        raise RuntimeError(f"No job folders found under: {folder_glob}")
    return jobs

def read_group_temps_from_compute(compute_file: str, M: int) -> np.ndarray:
    """
    compute.out:
      first M columns are group temperatures (K)
    """
    data = np.loadtxt(compute_file)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < M:
        raise RuntimeError(f"{compute_file}: too few columns ({data.shape[1]}) for M={M}")
    return data[:, :M]

def read_shc_omega_block(shc_file: str, num_omega: int):
    """
    shc.out format:
      last num_omega rows are omega block:
        col0 = omega (raw GPUMD spectral axis)
        col1 = J_in(omega)
        col2 = J_out(omega)
    """
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

    A = np.asarray(rows, dtype=float)
    if A.size == 0:
        raise RuntimeError(f"{shc_file}: no numeric rows parsed")

    # keep rows with non-negative first column; omega block is at the end
    mask = np.isfinite(A[:, 0]) & (A[:, 0] >= 0.0)
    W = A[mask]
    if W.shape[0] < num_omega:
        raise RuntimeError(
            f"{shc_file}: found only {W.shape[0]} non-negative rows, need num_omega={num_omega}"
        )

    Jw = W[-num_omega:, :]
    omega = Jw[:, 0]
    Jin   = Jw[:, 1]
    Jout  = Jw[:, 2]

    good = np.isfinite(omega) & np.isfinite(Jin) & np.isfinite(Jout)
    omega, Jin, Jout = omega[good], Jin[good], Jout[good]
    if omega.size == 0:
        raise RuntimeError(f"{shc_file}: omega block exists but all Jin/Jout are invalid")

    return omega, Jin, Jout

def read_LyLz_area_A2(model_xyz: str) -> float:
    with open(model_xyz, "r") as f:
        _ = f.readline()
        header = f.readline().strip()

    m = re.search(r'Lattice="([^"]+)"', header)
    if m is None:
        raise RuntimeError(f"{model_xyz}: cannot find Lattice= in header")

    lat = np.fromstring(m.group(1), sep=" ")
    if lat.size != 9:
        raise RuntimeError(f"{model_xyz}: expected 9 lattice numbers, got {lat.size}")

    a1 = lat[0:3]
    a2 = lat[3:6]
    a3 = lat[6:9]

    # transport is along x; area is |a2 x a3| in general
    area = np.linalg.norm(np.cross(a2, a3))
    return float(area)

def read_group_x_range(model_xyz: str, group_id: int):
    """
    Return (xmin, xmax, xmean) for atoms in a given group_id from model.xyz.

    Supports both common formats:
      1) species x y z group
      2) id species x y z group
    """
    with open(model_xyz, "r") as f:
        N = int(f.readline().strip())
        _ = f.readline()
        xs = []

        for _ in range(N):
            parts = f.readline().split()
            if not parts:
                continue

            # format: species x y z group
            if len(parts) == 5:
                x = float(parts[1])
                g = int(parts[4])

            # format: id species x y z group
            elif len(parts) >= 6:
                x = float(parts[2])
                g = int(parts[-1])

            else:
                continue

            if g == group_id:
                xs.append(x)

    if len(xs) == 0:
        raise RuntimeError(f"{model_xyz}: no atoms found for group {group_id}")

    xs = np.asarray(xs, dtype=float)
    return float(xs.min()), float(xs.max()), float(xs.mean())

def stderr_from_runs(arr_2d):
    n = arr_2d.shape[0]
    if n <= 1:
        return np.zeros(arr_2d.shape[1], dtype=float)
    return arr_2d.std(axis=0, ddof=1) / np.sqrt(n)

# =========================
# MAIN
# =========================
def main():
    job_dirs = collect_job_dirs(INPUT_GLOB)
    print(f"Found {len(job_dirs)} job folders under {INPUT_GLOB}")

    dT_list = []
    area_list = []
    xinfo_list = []

    omega_ref = None
    G_native_runs = []
    G_paper_runs = []
    J_area_runs = []   # optional diagnostic only
    L_shc_list = []
    V_shc_list = []

    for job in job_dirs:
        compute_file = os.path.join(job, COMPUTE_NAME)
        shc_file     = os.path.join(job, SHC_NAME)
        model_file   = os.path.join(job, MODEL_NAME)

        if not os.path.exists(compute_file):
            print(f"[WARN] skip {job}: missing {COMPUTE_NAME}")
            continue
        if not os.path.exists(shc_file):
            print(f"[WARN] skip {job}: missing {SHC_NAME}")
            continue
        if not os.path.exists(model_file):
            print(f"[WARN] skip {job}: missing {MODEL_NAME}")
            continue

        # ---- Delta T from compute.out ----
        Tg = read_group_temps_from_compute(compute_file, M)
        n = Tg.shape[0]
        start = int(np.floor(n * SKIP_FRACTION))
        Tg_ss = Tg[start:, :]
        Tavg = np.nanmean(Tg_ss, axis=0)

        T_left  = Tavg[G_LEFT - 1]
        T_right = Tavg[G_RIGHT - 1]
        dT = T_left - T_right

        if abs(dT) < 1e-12:
            print(f"[WARN] skip {job}: ΔT ~ 0")
            continue

        # ---- SHC spectrum ----
        omega, Jin, Jout = read_shc_omega_block(shc_file, NUM_OMEGA)
        Jtot = Jin + Jout   # native GPUMD spectral current

        # ---- Area ----
        area_A2 = read_LyLz_area_A2(model_file)

        # ---- SHC group x-range ----
        xmin, xmax, xmean = read_group_x_range(model_file, SHC_GROUP_ID)
        L_shc_A = xmax - xmin
        V_shc_A3 = area_A2 * L_shc_A

        # ---- Conductance in native GPUMD-consistent unit ----
        # unit: eV ps^-1 THz^-1 Å^-2 K^-1
        G_native = Jtot / (V_shc_A3 * dT)

        # ---- Paper unit ----
        # unit: MW m^-2 K^-1 THz^-1
        G_paper = G_native * EV_PER_PS_A2_TO_MW_PER_M2

        # ---- Area-normalized spectral heat current and conductance ----
        J_area = Jtot / area_A2
        # G_area = J_area / dT

        # ---- Consistency checks ----
        if omega_ref is None:
            omega_ref = omega.copy()
        else:
            if len(omega) != len(omega_ref) or not np.allclose(omega, omega_ref):
                raise RuntimeError(f"{job}: omega grid mismatch with previous jobs")

        dT_list.append(dT)
        area_list.append(area_A2)
        xinfo_list.append((job, xmin, xmax, xmean))
        J_area_runs.append(J_area)
        # G_area_runs.append(G_area)
        G_native_runs.append(G_native)
        G_paper_runs.append(G_paper)
        L_shc_list.append(L_shc_A)
        V_shc_list.append(V_shc_A3)

        print(f"\n[{os.path.basename(job)}]")
        print("Steady-state group temperatures (K):")
        for i, t in enumerate(Tavg, start=1):
            print(f"  g{i}: {t:.6f}")
        print(f"DeltaT = <Tg{G_LEFT}> - <Tg{G_RIGHT}> = {dT:.6f} K")
        print(f"Area = {area_A2:.6f} Å^2")
        print(f"SHC group {SHC_GROUP_ID} x-range (Å): {xmin:.3f} -> {xmax:.3f} (center ~ {xmean:.3f})")

    if len(G_native_runs) == 0:
        raise RuntimeError("No valid jobs parsed.")

    # ---- Stack and average ----
    J_area_arr = np.vstack(J_area_runs) if len(J_area_runs) > 0 else None

    G_native_arr = np.vstack(G_native_runs)
    G_paper_arr  = np.vstack(G_paper_runs)

    G_native_avg = G_native_arr.mean(axis=0)
    G_paper_avg  = G_paper_arr.mean(axis=0)

    G_native_stderr = stderr_from_runs(G_native_arr)
    G_paper_stderr  = stderr_from_runs(G_paper_arr)

    if J_area_arr is not None:
        J_area_avg = J_area_arr.mean(axis=0)
        J_area_stderr = stderr_from_runs(J_area_arr)

    # ---- Frequency axis for plotting/output ----
    if PLOT_W_OVER_2PI:
        freq_plot = omega_ref / (2.0 * np.pi)
        xlab = r"Frequency $\omega/2\pi$ (THz)"
        freq_name = "freq_w_over_2pi_THz"
    else:
        freq_plot = omega_ref.copy()
        xlab = r"Angular frequency $\omega$ (THz)"
        freq_name = "omega_raw_THz"

    # ---- Optional smoothing for plotting only ----
    Gs_native = moving_average(G_native_avg, SMOOTH_W)
    Gs_paper  = moving_average(G_paper_avg, SMOOTH_W)

    if J_area_arr is not None:
        Js = moving_average(J_area_avg, SMOOTH_W)

    # ---- Cumulative integral (plotting only) ----
    # Use plotted x-axis for visual convenience
    cum_G_native = np.zeros_like(Gs_native)
    cum_G_paper  = np.zeros_like(Gs_paper)

    if len(freq_plot) > 1:
        cum_G_native[1:] = np.cumsum(
            0.5 * (Gs_native[1:] + Gs_native[:-1]) * np.diff(freq_plot)
        )
        cum_G_paper[1:] = np.cumsum(
            0.5 * (Gs_paper[1:] + Gs_paper[:-1]) * np.diff(freq_plot)
        )

    # ---- Plot mask ----
    if MAX_W_PLOT is not None:
        mask_w = freq_plot <= MAX_W_PLOT
    else:
        mask_w = np.ones_like(freq_plot, dtype=bool)

    # ---- Summary ----
    print("\n================ Summary ================")
    print(f"Valid jobs used: {len(G_native_runs)}")
    print(f"Mean ΔT over jobs: {np.mean(dT_list):.6f} K")
    print(f"Std  ΔT over jobs: {np.std(dT_list, ddof=1) if len(dT_list) > 1 else 0.0:.6f} K")
    print(f"Mean area over jobs: {np.mean(area_list):.6f} Å^2")
    print("SHC region x-ranges:")
    for job, xmin, xmax, xmean in xinfo_list:
        print(f"  {os.path.basename(job)}: {xmin:.3f} -> {xmax:.3f} Å (center ~ {xmean:.3f})")

    # ---- Save averaged spectral conductance ----
    # ---- Save native averaged spectral conductance ----
    out_avg_native = np.column_stack([
        freq_plot,
        G_native_avg,
        G_native_stderr,
    ])
    out_avg_native_file = os.path.join(PATH_ROOT, f"{OUT_PREFIX}_Gomega_avg_native.txt")
    np.savetxt(
        out_avg_native_file,
        out_avg_native,
        header=(
            f"{freq_name}  "
            "G_native_avg_eV_ps^-1_THz^-1_A^-2_K^-1  "
            "G_native_stderr_eV_ps^-1_THz^-1_A^-2_K^-1\n"
            f"Averaged over {len(G_native_runs)} jobs"
        ),
        fmt="%.10e"
    )
    print(f"Saved: {out_avg_native_file}")

    # ---- Save paper averaged spectral conductance ----
    out_avg_paper = np.column_stack([
        freq_plot,
        G_paper_avg,
        G_paper_stderr,
    ])
    out_avg_paper_file = os.path.join(PATH_ROOT, f"{OUT_PREFIX}_Gomega_avg_paper.txt")
    np.savetxt(
        out_avg_paper_file,
        out_avg_paper,
        header=(
            f"{freq_name}  "
            "G_paper_avg_MW_m^-2_K^-1_THz^-1  "
            "G_paper_stderr_MW_m^-2_K^-1_THz^-1\n"
            f"Averaged over {len(G_paper_runs)} jobs"
        ),
        fmt="%.10e"
    )
    print(f"Saved: {out_avg_paper_file}")

   #-- Save cumulative integrals data (plotting only) ----
    cum_native_file = os.path.join(PATH_ROOT, f"{OUT_PREFIX}_Gomega_cumulative_native.txt")
    np.savetxt(
        cum_native_file,
        np.column_stack([freq_plot, cum_G_native]),
        header=f"{freq_name}  cumulative_G_native_eV_ps^-1_A^-2_K^-1",
        fmt="%.10e"
    )
    print(f"Saved: {cum_native_file}")

    cum_paper_file = os.path.join(PATH_ROOT, f"{OUT_PREFIX}_Gomega_cumulative_paper.txt")
    np.savetxt(
        cum_paper_file,
        np.column_stack([freq_plot, cum_G_paper]),
        header=f"{freq_name}  cumulative_G_paper_MW_m^-2_K^-1",
        fmt="%.10e"
    )
    print(f"Saved: {cum_paper_file}")

    # ---- Save per-run conductance for next MFP step ----
    out_runs_native = np.column_stack([freq_plot] + [G_native_arr[i] for i in range(G_native_arr.shape[0])])
    run_cols_native = "  ".join([f"G_native_run{i+1:02d}_eV_ps^-1_THz^-1_A^-2_K^-1" for i in range(G_native_arr.shape[0])])
    out_runs_native_file = os.path.join(PATH_ROOT, f"{OUT_PREFIX}_Gomega_all_runs_native.txt")
    np.savetxt(
        out_runs_native_file,
        out_runs_native,
        header=f"{freq_name}  {run_cols_native}",
        fmt="%.10e"
    )
    print(f"Saved: {out_runs_native_file}")

    # ---- Save scalar run summary ----
    summary_file = os.path.join(PATH_ROOT, f"{OUT_PREFIX}_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"n_jobs_used = {len(G_native_runs)}\n")
        f.write(f"G_LEFT = {G_LEFT}\n")
        f.write(f"G_RIGHT = {G_RIGHT}\n")
        f.write(f"SHC_GROUP_ID = {SHC_GROUP_ID}\n")
        f.write(f"mean_dT_K = {np.mean(dT_list):.10f}\n")
        f.write(f"std_dT_K = {np.std(dT_list, ddof=1) if len(dT_list) > 1 else 0.0:.10f}\n")
        f.write(f"mean_area_A2 = {np.mean(area_list):.10f}\n")
    print(f"Saved: {summary_file}")

    # ---- Plot spectral heat current / area ----
    plt.figure(figsize=(10, 6))
    for i in range(J_area_arr.shape[0]):
        plt.plot(freq_plot[mask_w], J_area_arr[i, mask_w], color="gray", alpha=0.20, linewidth=1)
    plt.plot(freq_plot[mask_w], J_area_avg[mask_w], color="C0", linewidth=2, label="avg raw")
    plt.plot(freq_plot[mask_w], Js[mask_w], color="C1", linewidth=2, label=f"avg smooth (w={SMOOTH_W})")
    if J_area_arr.shape[0] > 1:
        plt.fill_between(
            freq_plot[mask_w],
            (J_area_avg - J_area_stderr)[mask_w],
            (J_area_avg + J_area_stderr)[mask_w],
            color="C0",
            alpha=0.20,
            label="± stderr"
        )
    plt.xlabel(xlab)
    plt.ylabel("Spectral heat current / area (native units / Å$^2$)")
    plt.title(f"Bulk SiC short-bar SHC / area (group {SHC_GROUP_ID})")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    fig1 = os.path.join(PATH_ROOT, f"{OUT_PREFIX}_Jarea_avg.png")
    plt.savefig(fig1, dpi=300)
    print(f"Saved: {fig1}")

    # ---- Plot spectral conductance / area ----
    plt.figure(figsize=(10, 6))
    for i in range(G_native_arr.shape[0]):
        plt.plot(freq_plot[mask_w], G_native_arr[i, mask_w], color="gray", alpha=0.20, linewidth=1)
    plt.plot(freq_plot[mask_w], G_native_avg[mask_w], color="C0", linewidth=2, label="avg raw")
    plt.plot(freq_plot[mask_w], Gs_native[mask_w], color="C1", linewidth=2, label=f"avg smooth (w={SMOOTH_W})")
    if G_native_arr.shape[0] > 1:
        plt.fill_between(
            freq_plot[mask_w],
            (G_native_avg - G_native_stderr)[mask_w],
            (G_native_avg + G_native_stderr)[mask_w],
            color="C0", alpha=0.20, label="± stderr"
        )
    plt.xlabel(xlab)
    plt.ylabel(r"Spectral conductance $G(\omega)$ (eV ps$^{-1}$ THz$^{-1}$ Å$^{-2}$ K$^{-1}$)")
    plt.title(f"Bulk SiC short-bar spectral conductance (native, group {SHC_GROUP_ID})")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    fig_native = os.path.join(PATH_ROOT, f"{OUT_PREFIX}_Gomega_avg_native.png")
    plt.savefig(fig_native, dpi=300)
    print(f"Saved: {fig_native}")

    plt.figure(figsize=(10, 8))
    # for i in range(G_paper_arr.shape[0]):
    #     plt.plot(freq_plot[mask_w], G_paper_arr[i, mask_w], color="gray", alpha=0.20, linewidth=1)
    plt.plot(freq_plot[mask_w], G_paper_avg[mask_w], color="C0", linewidth=2, label="avg raw")
    # plt.plot(freq_plot[mask_w], Gs_paper[mask_w], color="C1", linewidth=2, label=f"avg smooth (w={SMOOTH_W})")
    if G_paper_arr.shape[0] > 1:
        plt.fill_between(
            freq_plot[mask_w],
            (G_paper_avg - G_paper_stderr)[mask_w],
            (G_paper_avg + G_paper_stderr)[mask_w],
            color="C0", alpha=0.20, label="± stderr"
        )
    plt.xlabel(xlab)
    plt.xlim(0, 35)
    plt.ylabel(r"$g(\omega)$ (MW m$^{-2}$ K$^{-1}$ THz$^{-1}$)")
    plt.title(f"Bulk SiC (0001) short-bar spectral conductance", fontsize=20)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    fig_paper = os.path.join(PATH_ROOT, f"{OUT_PREFIX}_Gomega_avg_paper.png")
    plt.savefig(fig_paper, dpi=300)
    plt.savefig(f"{fig_paper}.pdf", bbox_inches="tight")
    print(f"Saved: {fig_paper}")

    # ---- Plot cumulative spectral conductance ----
    plt.figure(figsize=(10, 6))
    plt.plot(freq_plot[mask_w], cum_G_native[mask_w], color="C0", linewidth=2)
    plt.xlabel(xlab)
    plt.ylabel(r"Cumulative $\int G(\omega)\,d\omega$ (eV ps$^{-1}$ Å$^{-2}$ K$^{-1}$)")
    plt.title("Cumulative spectral conductance (native)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig_cum_native = os.path.join(PATH_ROOT, f"{OUT_PREFIX}_Gomega_cumulative_native.png")
    plt.savefig(fig_cum_native, dpi=300)
    print(f"Saved: {fig_cum_native}")

    plt.figure(figsize=(10, 6))
    plt.plot(freq_plot[mask_w], cum_G_paper[mask_w], color="C0", linewidth=2)
    plt.xlabel(xlab)
    plt.ylabel(r"Cumulative $\int g(\omega)\,d\omega$ (MW m$^{-2}$ K$^{-1}$)")
    plt.title("Cumulative spectral conductance (paper unit)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig_cum_paper = os.path.join(PATH_ROOT, f"{OUT_PREFIX}_Gomega_cumulative_paper.png")
    plt.savefig(fig_cum_paper, dpi=300)
    print(f"Saved: {fig_cum_paper}")

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            _log_fh.close()
        except Exception:
            pass