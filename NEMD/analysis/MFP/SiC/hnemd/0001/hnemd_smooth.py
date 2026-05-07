#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime

# =========================
# USER SETTINGS
# =========================
path = "<PATH_TO_DATA>/qe_jobs/nep_training/NEMD/0217/post/MFP/SiC/0001"
INPUT_GLOB = path + "/job_*"

KAPPA_NAME = "kappa.out"
SHC_NAME   = "shc.out"

# choose one driven direction: "x", "y", or "z"
DIRECTION = "z"
SMOOTH_W = 21   # points for moving average of spectral kappa(omega), odd integer, 1 = off


# HNEMD settings used in run.in
DT_FS = 1.0
HNEMD_OUTPUT_INTERVAL = 1000   # from compute_hnemd
NUM_OMEGA = 1000               # from compute_shc
VOLUME_ANG3 = 105650.23        # simulation cell volume in Å^3
TEMPERATURE = 300.0            # K
FE = 2e-5                      # HNEMD driving force

# averaging window
USE_PLATEAU = False            # True: auto plateau; False: use last LAST_WINDOW_PS
LAST_WINDOW_PS = 100.0

# spectral axis / validation
PLOT_W_OVER_2PI = True     # True: plot/store frequency as omega/2pi (THz)
VALIDATE_SPECTRUM = True   # integrate spectral kappa and compare with kappa.out mean

# plotting
MAX_T_PLOT = None              # ps, or None
MAX_W_PLOT = None              # THz, or None

OUT_PREFIX = "hnemd_sic_z"

# unit conversion:
# shc.out J(omega): Å*eV/ps/THz
# kappa(omega) = J(omega)/(V*T*Fe)
# 1 eV/(ps*Å) = 1.602176634e3 W/m
EV_PER_PS_PER_ANG_TO_W_PER_M = 1.602176634e3


# =========================
# LOGGING
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = f"{path}/postprocess_{timestamp}.log"


def log(msg: str):
    print(msg)
    with open(logfile, "a") as f:
        f.write(msg + "\n")


# =========================
# HELPERS
# =========================
def collect_existing_files(folder_glob: str, filename: str) -> list[str]:
    folders = sorted(glob.glob(folder_glob))
    return [os.path.join(d, filename) for d in folders if os.path.exists(os.path.join(d, filename))]


def plateau_average(arr: np.ndarray, i_start: int, i_end: int) -> float:
    return float(np.mean(arr[i_start:i_end]))


def find_plateau_indices(t, y,
                         min_fraction=0.3,
                         window_ps=20.0,
                         slope_threshold=5e-4):
    """
    Heuristic plateau finder on y(t).
    Returns slice indices [i_start:i_end].
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if len(t) < 5:
        raise RuntimeError("Time array too short to find plateau.")

    dt = np.median(np.diff(t))
    nwin = max(int(window_ps / dt), 1)

    kernel = np.ones(nwin) / nwin
    y_smooth = np.convolve(y, kernel, mode="same")
    dy_dt = np.gradient(y_smooth, t)

    t_min = t[0] + min_fraction * (t[-1] - t[0])
    mask = (np.abs(dy_dt) < slope_threshold) & (t >= t_min)

    best_len = 0
    best_start = None
    cur_start = None

    for i, m in enumerate(mask):
        if m:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                cur_len = i - cur_start
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
                cur_start = None

    if cur_start is not None:
        cur_len = len(mask) - cur_start
        if cur_len > best_len:
            best_len = cur_len
            best_start = cur_start

    if best_len == 0 or best_start is None:
        raise RuntimeError("No plateau region found; try relaxing thresholds.")

    return best_start, best_start + best_len


def window_stats(arr, i_start, i_end):
    """
    arr shape: (n_runs, n_t)
    First average each run over the window, then compute run-to-run stats.
    """
    vals_per_run = arr[:, i_start:i_end].mean(axis=1)
    mean = vals_per_run.mean()
    std = vals_per_run.std(ddof=1) if arr.shape[0] > 1 else 0.0
    stderr = std / np.sqrt(arr.shape[0]) if arr.shape[0] > 0 else np.nan
    return mean, std, stderr


def read_kappa_running(kappa_file: str, direction: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read GPUMD kappa.out for one run.

    GPUMD kappa.out format (5 columns):
      col1 = k_{mu x}^{in}
      col2 = k_{mu x}^{out}
      col3 = k_{mu y}^{in}
      col4 = k_{mu y}^{out}
      col5 = k_{mu z}^{tot}

    For 3D materials:
      k_{mu x}^{tot} = col1 + col2
      k_{mu y}^{tot} = col3 + col4
      k_{mu z}^{tot} = col5
    """
    data = np.loadtxt(kappa_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 5:
        raise ValueError(f"{kappa_file} has only {data.shape[1]} columns; expected 5.")

    nrow = data.shape[0]
    t_ps = (np.arange(nrow) + 1) * HNEMD_OUTPUT_INTERVAL * DT_FS / 1000.0

    direction = direction.lower()

    if direction == "x":
        k_running = data[:, 0] + data[:, 1]
    elif direction == "y":
        k_running = data[:, 2] + data[:, 3]
    elif direction == "z":
        k_running = data[:, 4]
    else:
        raise ValueError(f"Unknown direction: {direction}. Use x, y, or z.")

    return t_ps, k_running
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

def read_spectral_kappa(shc_file: str,
                        num_omega: int,
                        volume_ang3: float,
                        temperature_K: float,
                        Fe_inv_ang: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Read GPUMD shc.out and convert the frequency block to spectral kappa(omega).

    Assumes last num_omega rows are frequency block:
      col0 = omega (THz)
      col1 = J_in(omega)
      col2 = J_out(omega)

    Returns:
      omega_thz, kappa_omega [W/m/K/THz]
    """
    data = np.loadtxt(shc_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[0] < num_omega:
        raise ValueError(f"{shc_file} has fewer than num_omega={num_omega} rows.")

    freq_block = data[-num_omega:, :]
    if freq_block.shape[1] < 3:
        raise ValueError(f"{shc_file} frequency block has fewer than 3 columns.")

    omega_raw = freq_block[:, 0]
    Jw = freq_block[:, 1] + freq_block[:, 2]

    # spectral kappa on the raw omega grid from shc.out
    kappa_omega_raw = (
        Jw / (volume_ang3 * temperature_K * Fe_inv_ang)
    ) * EV_PER_PS_PER_ANG_TO_W_PER_M

    return omega_raw, kappa_omega_raw


# =========================
# COLLECT FILES
# =========================
kappa_files = collect_existing_files(INPUT_GLOB, KAPPA_NAME)
shc_files   = collect_existing_files(INPUT_GLOB, SHC_NAME)

if len(kappa_files) == 0:
    raise SystemExit(f"No {KAPPA_NAME} files found under {INPUT_GLOB}!")
if len(shc_files) == 0:
    raise SystemExit(f"No {SHC_NAME} files found under {INPUT_GLOB}!")

log(f"Found {len(kappa_files)} {KAPPA_NAME} files under {INPUT_GLOB}.")
log(f"Found {len(shc_files)} {SHC_NAME} files under {INPUT_GLOB}.")

if len(kappa_files) != len(shc_files):
    log("Warning: number of kappa.out and shc.out files differs. Proceeding separately.")


# =========================
# PROCESS kappa.out
# =========================
k_list = []
t_ref = None

for kappa in kappa_files:
    t_ps, k_running = read_kappa_running(kappa, DIRECTION)
    if t_ref is None:
        t_ref = t_ps
    else:
        if len(t_ps) != len(t_ref):
            raise ValueError(f"Time array length mismatch in {kappa}")

    k_list.append(k_running)

k_arr = np.vstack(k_list)
k_avg = k_arr.mean(axis=0)
n_runs_k = k_arr.shape[0]

log(f"Averaged running kappa over {n_runs_k} runs.")


# =========================
# PROCESS shc.out
# =========================
spec_list = []
omega_ref_raw = None

for shc in shc_files:
    omega_raw, kappa_omega_raw = read_spectral_kappa(
        shc_file=shc,
        num_omega=NUM_OMEGA,
        volume_ang3=VOLUME_ANG3,
        temperature_K=TEMPERATURE,
        Fe_inv_ang=FE,
    )

    if omega_ref_raw is None:
        omega_ref_raw = omega_raw
    else:
        if len(omega_raw) != len(omega_ref_raw) or not np.allclose(omega_raw, omega_ref_raw):
            raise ValueError(f"Frequency grid mismatch in {shc}")

    spec_list.append(kappa_omega_raw)

spec_arr = np.vstack(spec_list)
spec_avg = spec_arr.mean(axis=0)
n_runs_spec = spec_arr.shape[0]

spec_smooth = moving_average(spec_avg, SMOOTH_W)

# convert axis for plotting / comparison with PDOS
if PLOT_W_OVER_2PI:
    freq_ref = omega_ref_raw / (2.0 * np.pi)
    x_label_spec = r"Frequency $\omega/2\pi$ (THz)"
else:
    freq_ref = omega_ref_raw
    x_label_spec = r"Angular frequency $\omega$ (THz)"

log(f"Averaged spectral kappa over {n_runs_spec} runs.")


# =========================
# PLOT MASKS
# =========================
if MAX_T_PLOT is not None:
    plot_mask_t = t_ref <= MAX_T_PLOT
else:
    plot_mask_t = np.ones_like(t_ref, dtype=bool)

if MAX_W_PLOT is not None:
    plot_mask_w = freq_ref <= MAX_W_PLOT
else:
    plot_mask_w = np.ones_like(freq_ref, dtype=bool)


# =========================
# AVERAGING WINDOW
# =========================
if USE_PLATEAU:
    i0, i1 = find_plateau_indices(
        t_ref, k_avg,
        min_fraction=0.3,
        window_ps=20.0,
        slope_threshold=5e-4
    )
    method_str = f"plateau (from kappa_{DIRECTION} avg)"
else:
    t_end_target = t_ref[-1]
    t_start_target = max(t_ref[0], t_end_target - LAST_WINDOW_PS)
    i0 = int(np.searchsorted(t_ref, t_start_target, side="left"))
    i1 = len(t_ref)
    method_str = f"last {LAST_WINDOW_PS:.2f} ps"

t_start, t_end = t_ref[i0], t_ref[i1 - 1]
log(f"Averaging window [{method_str}]: {t_start:.2f} – {t_end:.2f} ps ({i1 - i0} points)")


# =========================
# RUN-TO-RUN STATS
# =========================
results = {}
mean, std, stderr = window_stats(k_arr, i0, i1)
results[f"k_{DIRECTION}"] = (mean, std, stderr)
log(f"k_{DIRECTION}: mean = {mean:8.3f} W/mK, std = {std:8.3f}, stderr = {stderr:8.3f}")


# =========================
# SAVE STATS
# =========================
stats_file = f"{path}/kappa_plateau_stats.txt"
with open(stats_file, "w") as f:
    f.write("# Averaging window (ps): "
            f"{t_start:.6f}  {t_end:.6f}\n")
    f.write("# name    mean(W/mK)      std(W/mK)       stderr(W/mK)\n")
    for name, (m, s, se) in results.items():
        f.write(f"{name:8s}  {m:13.6f}  {s:13.6f}  {se:13.6f}\n")

log(f"Saved plateau statistics to: {stats_file}")


# =========================
# SAVE AVG KAPPA VS TIME
# =========================
out_time_file = f"{path}/kappa_avg_vs_time.txt"
out_data = np.column_stack([t_ref, k_avg])

header = (
    f"t(ps)   kappa_{DIRECTION}_avg(W/mK)\n"
    f"Averaged over {n_runs_k} runs"
)

np.savetxt(out_time_file, out_data, header=header, fmt="%.6f")
log(f"Saved averaged kappa curve to: {out_time_file}")


# =========================
# SAVE FINAL VALUE
# =========================
final_value_file = f"{path}/kappa_final_values.txt"
np.savetxt(
    final_value_file,
    np.array([k_avg[-1]]),
    header=f"kappa_{DIRECTION}_avg_final\n",
    fmt="%.6f"
)
log(f"Saved final averaged kappa value to: {final_value_file}")


# =========================
# SAVE SPECTRAL KAPPA
# =========================
out_spec_file = f"{path}/kappa_spectrum_avg.txt"
out_spec = np.column_stack([freq_ref, spec_avg])
freq_header = "freq_w_over_2pi_THz" if PLOT_W_OVER_2PI else "omega_raw_THz"
np.savetxt(
    out_spec_file,
    out_spec,
    header=f"{freq_header}  kappa_{DIRECTION}_avg_Wm-1K-1THz-1\nAveraged over {n_runs_spec} runs",
    fmt="%.8e"
)
log(f"Saved averaged spectral kappa to: {out_spec_file}")


# =========================
# SAVE PER-RUN SPECTRAL STATS
# =========================
spec_std = spec_arr.std(axis=0, ddof=1) if n_runs_spec > 1 else np.zeros_like(spec_avg)
spec_stderr = spec_std / np.sqrt(n_runs_spec) if n_runs_spec > 0 else np.full_like(spec_avg, np.nan)

out_spec_stat_file = f"{path}/kappa_spectrum_avg_with_error.txt"
out_spec_stat = np.column_stack([freq_ref, spec_avg, spec_std, spec_stderr])
freq_header = "freq_w_over_2pi_THz" if PLOT_W_OVER_2PI else "omega_raw_THz"
np.savetxt(
    out_spec_stat_file,
    out_spec_stat,
    header=(
        f"{freq_header}  kappa_{DIRECTION}_avg_Wm-1K-1THz-1  "
        f"std_Wm-1K-1THz-1  stderr_Wm-1K-1THz-1"
    ),
    fmt="%.8e"
)
log(f"Saved spectral kappa with error to: {out_spec_stat_file}")

# =========================
# VALIDATE SPECTRAL INTEGRAL
# =========================
if VALIDATE_SPECTRUM:
    # integrate on raw omega grid: kappa = ∫ [kappa(omega)/(2*pi)] domega
    kappa_spec_per_run = np.trapz(spec_arr / (2.0 * np.pi), x=omega_ref_raw, axis=1)
    kappa_spec_mean = kappa_spec_per_run.mean()
    kappa_spec_std = kappa_spec_per_run.std(ddof=1) if len(kappa_spec_per_run) > 1 else 0.0
    kappa_spec_stderr = kappa_spec_std / np.sqrt(len(kappa_spec_per_run))

    log(
        f"Spectral integral check: "
        f"kappa_spec = {kappa_spec_mean:.3f} ± {kappa_spec_stderr:.3f} W/mK"
    )
    log(
        f"Running-kappa window mean: "
        f"kappa_run  = {mean:.3f} ± {stderr:.3f} W/mK"
    )
    log(
        f"Difference (spec - run) = {kappa_spec_mean - mean:.3f} W/mK"
    )

# =========================
# PLOT RUNNING KAPPA
# =========================
plt.rcParams["font.size"] = 16
plt.figure(figsize=(10, 6))

for i in range(n_runs_k):
    plt.plot(t_ref[plot_mask_t], k_arr[i, plot_mask_t], color="gray", alpha=0.25, linewidth=1)

plt.plot(t_ref[plot_mask_t], k_avg[plot_mask_t], label=f"kappa_{DIRECTION} (avg)", linewidth=3)
plt.axvspan(t_start, t_end, color="orange", alpha=0.15, label="avg window")

plt.xlabel("Time (ps)")
plt.ylabel("Running κ(t) [W/mK]")
plt.title(f"HNEMD running thermal conductivity ({DIRECTION})")
plt.legend()

plt.text(
    0.5, 1.02,
    f"mean κ = {mean:.2f} ± {stderr:.2f} W/mK",
    transform=plt.gca().transAxes,
    ha="center",
    va="bottom",
    fontsize=16,
)

plt.tight_layout()
fig_running = f"{path}/hnemd_running_kappa_avg.png"
plt.savefig(fig_running, dpi=300)
log(f"Saved figure: {fig_running}")


# =========================
# PLOT SPECTRAL KAPPA
# =========================
plt.figure(figsize=(10, 6))

for i in range(n_runs_spec):
    plt.plot(freq_ref[plot_mask_w], spec_arr[i, plot_mask_w], color="gray", alpha=0.2, linewidth=1)

plt.plot(freq_ref[plot_mask_w], spec_avg[plot_mask_w], color="C0", linewidth=2, label="avg raw")
plt.plot(freq_ref[plot_mask_w], spec_smooth[plot_mask_w], color="C1", linewidth=2, label=f"avg smooth (w={SMOOTH_W})")

if n_runs_spec > 1:
    plt.fill_between(
        freq_ref[plot_mask_w],
        (spec_avg - spec_stderr)[plot_mask_w],
        (spec_avg + spec_stderr)[plot_mask_w],
        color="C0",
        alpha=0.25,
        label="± stderr"
    )

plt.xlabel(x_label_spec)
plt.ylabel(r"Spectral $\kappa(\omega)$ [W m$^{-1}$ K$^{-1}$ THz$^{-1}$]")
axis_tag = r"$\omega/2\pi$" if PLOT_W_OVER_2PI else r"$\omega$"
plt.title(f"HNEMD spectral thermal conductivity ({DIRECTION}, axis={axis_tag})")
plt.legend()
plt.tight_layout()

fig_spec = f"{path}/hnemd_spectral_kappa_smooth.png"
plt.savefig(fig_spec, dpi=300)
log(f"Saved figure: {fig_spec}")


# =========================
# OPTIONAL: CUMULATIVE AVG OF RUNNING KAPPA
# =========================
k_cum = np.cumsum(k_avg) / np.arange(1, len(k_avg) + 1)

plt.figure(figsize=(10, 6))
plt.plot(t_ref[plot_mask_t], k_cum[plot_mask_t], linewidth=2.5)
plt.axhline(mean, linestyle="--", linewidth=1.5, label=f"window mean = {mean:.2f}")
plt.xlabel("Time (ps)")
plt.ylabel("Cumulative average κ [W/mK]")
plt.title(f"Cumulative average of HNEMD κ ({DIRECTION})")
plt.legend()
plt.tight_layout()

fig_cum = f"{path}/hnemd_cumulative_kappa_avg.png"
plt.savefig(fig_cum, dpi=300)
log(f"Saved figure: {fig_cum}")


log("Done.")