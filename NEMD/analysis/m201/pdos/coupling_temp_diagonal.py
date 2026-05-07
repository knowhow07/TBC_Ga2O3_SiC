#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, savgol_filter
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif"]
plt.rcParams.update({"mathtext.default": "regular"})
mpl.rcParams["font.size"] = 16

# =========================
# USER SETTINGS
# =========================
TEMP_INPUTS = {
    "300K": {
        "model_xyz": "./300k/model.xyz",
        "velocity_npy": "./300k/velocity_g2.npy",
    },
    "600K": {
        "model_xyz": "./600k/model.xyz",
        "velocity_npy": "./600k/velocity_g2.npy",
    },
    "900K": {
        "model_xyz": "./900k/model.xyz",
        "velocity_npy": "./900k/velocity_g2.npy",
    },
}

GROUP_ID = 2
X_INTERFACE = 105.0
WINDOW_W = 5.0

DT_FS = 1.0
DUMP_EVERY = 10

FREQ_MAX_THz = 35.0
NPERSEG = 4096
NOOVERLAP = None
DETREND = "constant"

MIN_ATOMS_PER_REGION = 20

# Frequency bins for matrix
FREQ_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]

# Pick a few off-diagonal elements to track vs temperature
# format: (left_bin_index, right_bin_index)
SELECTED_OFFDIAG = [
    (1, 2),  # 5-10 -> 10-15
    (2, 3),  # 10-15 -> 15-20
    (3, 4),  # 15-20 -> 20-25
    (4, 5),  # 20-25 -> 25-30
    (4, 1),  # 20-25 -> 5-10
    (5, 4),  # 25-30 -> 20-25
]

# band-energy smoothing for covariance analysis
ENERGY_SMOOTH_WINDOW = 21
ENERGY_SMOOTH_POLY = 3

# coherence smoothing for plotting
COH_SMOOTH_WINDOW = 21
COH_SMOOTH_POLY = 3

RESULTS_DIR = "results_matrix_compare"
os.makedirs(RESULTS_DIR, exist_ok=True)
# =========================


def read_snapshot_xyz(fname):
    with open(fname, "r") as f:
        n = int(f.readline().strip())
        header2 = f.readline().strip()
        lines = [f.readline().strip() for _ in range(n)]

    x = np.zeros(n, float)
    g = np.full(n, -1, int)

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 4:
            raise RuntimeError(f"{fname}: bad atom line: {line}")

        try:
            int(parts[0])  # id sp x y z [group]
            x[i] = float(parts[2])
            if len(parts) >= 6:
                g[i] = int(parts[5])
        except ValueError:
            x[i] = float(parts[1])
            if len(parts) == 5:
                g[i] = int(parts[4])

    lx = None
    m = re.search(r'Lattice="([^"]+)"', header2)
    if m:
        lat = list(map(float, m.group(1).split()))
        if len(lat) >= 1:
            lx = float(lat[0])

    if np.all(g < 0):
        raise RuntimeError(f"{fname}: no group column detected.")

    return x, g, lx


def reshape_velocity_npy(npy_file, n_atoms):
    data = np.load(npy_file, mmap_mode="r")
    if data.ndim != 2 or data.shape[1] != 3:
        raise RuntimeError(f"{npy_file}: expected shape (total_rows, 3), got {data.shape}")

    total_rows = data.shape[0]
    if total_rows % n_atoms != 0:
        raise RuntimeError(
            f"{npy_file}: row count mismatch.\n"
            f"  total_rows = {total_rows}\n"
            f"  n_atoms(group {GROUP_ID}) = {n_atoms}\n"
            f"  total_rows % n_atoms = {total_rows % n_atoms}"
        )

    n_frames = total_rows // n_atoms
    if n_frames < 10:
        raise RuntimeError(f"{npy_file}: too few frames ({n_frames})")

    return data.reshape(n_frames, n_atoms, 3)


def make_region_indices(x_g, x_interface, window_w):
    left_lo = x_interface - window_w
    left_hi = x_interface
    right_lo = x_interface
    right_hi = x_interface + window_w

    mask_left = (x_g >= left_lo) & (x_g < left_hi)
    mask_right = (x_g > right_lo) & (x_g <= right_hi)

    sel_left = np.where(mask_left)[0]
    sel_right = np.where(mask_right)[0]

    return sel_left, sel_right, (left_lo, left_hi, right_lo, right_hi)


def smooth_curve(y, window=21, poly=3):
    y = np.asarray(y, dtype=float)
    if window is None or window <= 1 or len(y) < 7:
        return y.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window < poly + 2:
        return y.copy()
    return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")


def build_signals(v_region):
    vx = np.asarray(v_region[:, :, 0], dtype=np.float64)
    vx_com = vx.mean(axis=1, keepdims=True)
    vx_fluc = vx - vx_com
    return vx_fluc


def compute_avg_coherence(vxL, vxR, fs_hz, nperseg, noverlap, detrend):
    nL = vxL.shape[1]
    nR = vxR.shape[1]
    npair = min(nL, nR)
    if npair < 2:
        raise RuntimeError(f"Too few matched pairs: nL={nL}, nR={nR}")

    coh_list = []
    freq_ref = None
    for i in range(npair):
        f_hz, cxy = coherence(
            vxL[:, i], vxR[:, i],
            fs=fs_hz,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend
        )
        if freq_ref is None:
            freq_ref = f_hz
        coh_list.append(cxy)

    coh_avg = np.mean(np.vstack(coh_list), axis=0)
    return freq_ref, coh_avg


def band_average(freq, y, f1, f2):
    m = (freq >= f1) & (freq < f2)
    if np.sum(m) < 2:
        return np.nan
    return np.trapz(y[m], freq[m]) / (f2 - f1)


def bandpass_fft_time_series(vx_atoms, dt_s, f1_THz, f2_THz):
    tsteps = vx_atoms.shape[0]
    freqs_hz = np.fft.rfftfreq(tsteps, d=dt_s)
    freqs_thz = freqs_hz * 1e-12

    mask = (freqs_thz >= f1_THz) & (freqs_thz < f2_THz)

    V = np.fft.rfft(vx_atoms, axis=0)
    V_filt = np.zeros_like(V)
    V_filt[mask, :] = V[mask, :]
    vx_band = np.fft.irfft(V_filt, n=tsteps, axis=0)
    return vx_band


def compute_band_energy_series(vx_atoms, dt_s, bins):
    energies = []
    for (f1, f2) in bins:
        vx_bin = bandpass_fft_time_series(vx_atoms, dt_s, f1, f2)
        e_t = np.mean(vx_bin ** 2, axis=1)
        e_t = smooth_curve(e_t, ENERGY_SMOOTH_WINDOW, ENERGY_SMOOTH_POLY)
        energies.append(e_t)
    return energies


def make_coupling_matrices(left_energy_series, right_energy_series):
    nb = len(left_energy_series)
    corr_mat = np.zeros((nb, nb), float)
    cov_mat = np.zeros((nb, nb), float)
    abs_cov_mat = np.zeros((nb, nb), float)

    for i in range(nb):
        x = np.asarray(left_energy_series[i], dtype=float)
        for j in range(nb):
            y = np.asarray(right_energy_series[j], dtype=float)

            n = min(len(x), len(y))
            xx = x[:n]
            yy = y[:n]

            xx0 = xx - xx.mean()
            yy0 = yy - yy.mean()

            cov = np.mean(xx0 * yy0)
            sx = xx.std()
            sy = yy.std()
            corr = cov / (sx * sy + 1e-30)

            corr_mat[i, j] = corr
            cov_mat[i, j] = cov
            abs_cov_mat[i, j] = abs(cov)

    return corr_mat, cov_mat, abs_cov_mat


def plot_heatmap(mat, bins, title, out_png, cmap="viridis", vmin=None, vmax=None, annotate_fmt=".2e"):
    labels = [f"{a}-{b}" for a, b in bins]
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(mat, origin="lower", cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Right-side frequency bin (THz)")
    ax.set_ylabel("Left-side frequency bin (THz)")
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            txt = format(mat[i, j], annotate_fmt)
            ax.text(
                j, i, txt, ha="center", va="center",
                color="white" if (vmax is not None and mat[i, j] > 0.6 * vmax) else "black",
                fontsize=9
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.ax.set_ylabel("Value", rotation=270, labelpad=15)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def save_matrix_txt(fname, bins, mat, header_name):
    labels = [f"{a}-{b}" for a, b in bins]
    with open(fname, "w") as f:
        f.write(f"# {header_name}\n")
        f.write("# rows = left bins, cols = right bins\n")
        f.write("# bins: " + "  ".join(labels) + "\n")
        for i, lab in enumerate(labels):
            row = " ".join(f"{mat[i, j]:.8e}" for j in range(len(labels)))
            f.write(f"{lab:>8s}  {row}\n")


def plot_selected_elements_vs_temperature(temp_order, results, selected_pairs, out_png):
    x = np.arange(len(temp_order))
    fig, ax = plt.subplots(figsize=(9, 6))

    for (i, j) in selected_pairs:
        y = [results[t]["abs_cov_mat"][i, j] for t in temp_order]
        lab = f"L {FREQ_BINS[i][0]}-{FREQ_BINS[i][1]} → R {FREQ_BINS[j][0]}-{FREQ_BINS[j][1]}"
        ax.plot(x, y, marker="o", linewidth=2, label=lab)

    ax.set_xticks(x)
    ax.set_xticklabels(temp_order)
    ax.set_ylabel("Unnormalized |covariance|")
    ax.set_xlabel("Temperature")
    ax.set_title("Selected off-diagonal coupling elements vs temperature")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_diag_offdiag_abs_cov(temp_order, results, out_png):
    labels = [f"{a}-{b}" for a, b in FREQ_BINS]
    nb = len(FREQ_BINS)
    fig, axes = plt.subplots(1, len(temp_order), figsize=(5.0 * len(temp_order), 4.8), sharey=True)

    if len(temp_order) == 1:
        axes = [axes]

    for ax, temp in zip(axes, temp_order):
        mat = results[temp]["abs_cov_mat"]
        diag = np.diag(mat)
        off = []
        for i in range(nb):
            vals = np.delete(mat[i, :], i)
            off.append(np.mean(vals))
        off = np.array(off)

        xx = np.arange(nb)
        w = 0.38
        ax.bar(xx - w/2, diag, width=w, label="Diagonal")
        ax.bar(xx + w/2, off, width=w, label="Mean off-diagonal")
        ax.set_xticks(xx)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(temp)
        ax.set_xlabel("Frequency bin (THz)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Unnormalized |covariance|")
    axes[0].legend()
    fig.suptitle("Diagonal vs off-diagonal coupling from unnormalized |covariance|", y=0.98)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def analyze_one_temperature(temp_label, model_xyz, velocity_npy):
    x_all, g_all, lx = read_snapshot_xyz(model_xyz)
    idx_g = np.where(g_all == GROUP_ID)[0]
    if idx_g.size == 0:
        raise RuntimeError(f"[{temp_label}] No atoms with group == {GROUP_ID}")

    x_g = x_all[idx_g]
    sel_left, sel_right, bounds = make_region_indices(x_g, X_INTERFACE, WINDOW_W)
    left_lo, left_hi, right_lo, right_hi = bounds

    if sel_left.size < MIN_ATOMS_PER_REGION or sel_right.size < MIN_ATOMS_PER_REGION:
        raise RuntimeError(
            f"[{temp_label}] Too few atoms in selected slab(s): left={sel_left.size}, right={sel_right.size}"
        )

    sel_left = sel_left[np.argsort(x_g[sel_left])]
    sel_right = sel_right[np.argsort(x_g[sel_right])]

    v = reshape_velocity_npy(velocity_npy, n_atoms=idx_g.size)
    n_frames = v.shape[0]
    dt_s = DT_FS * DUMP_EVERY * 1e-15
    fs_hz = 1.0 / dt_s

    if NPERSEG > n_frames:
        nperseg = max(256, 2 ** int(np.floor(np.log2(n_frames))))
    else:
        nperseg = NPERSEG
    noverlap = NOOVERLAP if NOOVERLAP is not None else nperseg // 2

    vL = np.asarray(v[:, sel_left, :], dtype=np.float64)
    vR = np.asarray(v[:, sel_right, :], dtype=np.float64)

    vxL = build_signals(vL)
    vxR = build_signals(vR)

    # coherence: keep as main dephasing evidence
    f_hz, coh = compute_avg_coherence(vxL, vxR, fs_hz, nperseg, noverlap, DETREND)
    f_thz = f_hz * 1e-12
    m = f_thz <= FREQ_MAX_THz
    f_thz = f_thz[m]
    coh = coh[m]
    coh_smooth = smooth_curve(coh, COH_SMOOTH_WINDOW, COH_SMOOTH_POLY)
    coh_bins = [band_average(f_thz, coh_smooth, f1, f2) for f1, f2 in FREQ_BINS]

    # covariance matrices from band-energy fluctuations
    left_energy = compute_band_energy_series(vxL, dt_s, FREQ_BINS)
    right_energy = compute_band_energy_series(vxR, dt_s, FREQ_BINS)
    corr_mat, cov_mat, abs_cov_mat = make_coupling_matrices(left_energy, right_energy)

    return {
        "temp": temp_label,
        "Lx": lx,
        "n_frames": n_frames,
        "left_n": sel_left.size,
        "right_n": sel_right.size,
        "left_bounds": (left_lo, left_hi),
        "right_bounds": (right_lo, right_hi),
        "freq": f_thz,
        "coh": coh,
        "coh_smooth": coh_smooth,
        "coh_bins": np.array(coh_bins),
        "corr_mat": corr_mat,
        "cov_mat": cov_mat,
        "abs_cov_mat": abs_cov_mat,
    }


def main():
    results = {}
    for temp, paths in TEMP_INPUTS.items():
        print(f"[INFO] analyzing {temp}")
        results[temp] = analyze_one_temperature(
            temp,
            paths["model_xyz"],
            paths["velocity_npy"],
        )

    temp_order = list(TEMP_INPUTS.keys())

    # ---------- coherence kept as main evidence ----------
    plt.figure(figsize=(10, 6.5))
    for temp in temp_order:
        r = results[temp]
        plt.plot(r["freq"], r["coh_smooth"], lw=2.8, label=temp)
    for f1, f2 in FREQ_BINS:
        plt.axvline(f1, ls="--", lw=0.8, alpha=0.18, color="gray")
    plt.axvline(FREQ_BINS[-1][1], ls="--", lw=0.8, alpha=0.18, color="gray")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Coherence")
    plt.xlim(0, FREQ_MAX_THz)
    plt.grid(alpha=0.18, linestyle="--")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "coherence_spectrum_compare.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(9, 6))
    bin_labels = [f"{f1}-{f2}" for f1, f2 in FREQ_BINS]
    x = np.arange(len(bin_labels))
    width = 0.8 / len(temp_order)
    for j, temp in enumerate(temp_order):
        plt.bar(
            x + (j - (len(temp_order)-1)/2) * width,
            results[temp]["coh_bins"],
            width=width,
            label=temp
        )
    plt.xticks(x, bin_labels)
    plt.xlabel("Frequency bin (THz)")
    plt.ylabel("Average coherence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "coherence_bins_compare.png"), dpi=300)
    plt.close()

    # ---------- global color scales for covariance heatmaps ----------
    cov_all = np.concatenate([results[t]["cov_mat"].ravel() for t in temp_order])
    abs_cov_all = np.concatenate([results[t]["abs_cov_mat"].ravel() for t in temp_order])

    cov_vmin = float(np.min(cov_all))
    cov_vmax = float(np.max(cov_all))
    abs_cov_vmin = 0.0
    abs_cov_vmax = float(np.max(abs_cov_all))

    # ---------- save and plot same-scale heatmaps ----------
    for temp in temp_order:
        r = results[temp]

        save_matrix_txt(
            os.path.join(RESULTS_DIR, f"{temp}_cov_matrix.txt"),
            FREQ_BINS, r["cov_mat"], f"{temp} covariance matrix"
        )
        save_matrix_txt(
            os.path.join(RESULTS_DIR, f"{temp}_abs_cov_matrix.txt"),
            FREQ_BINS, r["abs_cov_mat"], f"{temp} abs covariance matrix"
        )
        save_matrix_txt(
            os.path.join(RESULTS_DIR, f"{temp}_corr_matrix.txt"),
            FREQ_BINS, r["corr_mat"], f"{temp} correlation matrix"
        )

        plot_heatmap(
            r["cov_mat"],
            FREQ_BINS,
            title=f"{temp}: cross-interface coupling matrix\n(covariance of band-energy fluctuations)",
            out_png=os.path.join(RESULTS_DIR, f"{temp}_cov_matrix.png"),
            cmap="coolwarm",
            vmin=cov_vmin,
            vmax=cov_vmax,
            annotate_fmt=".1e",
        )

        plot_heatmap(
            r["abs_cov_mat"],
            FREQ_BINS,
            title=f"{temp}: cross-interface coupling matrix\n(|covariance| of band-energy fluctuations)",
            out_png=os.path.join(RESULTS_DIR, f"{temp}_abs_cov_matrix.png"),
            cmap="viridis",
            vmin=abs_cov_vmin,
            vmax=abs_cov_vmax,
            annotate_fmt=".1e",
        )

    # ---------- selected off-diagonal terms vs temperature ----------
    plot_selected_elements_vs_temperature(
        temp_order,
        results,
        SELECTED_OFFDIAG,
        os.path.join(RESULTS_DIR, "selected_offdiag_vs_temperature.png"),
    )

    # ---------- diagonal vs off-diagonal from unnormalized abs covariance ----------
    plot_diag_offdiag_abs_cov(
        temp_order,
        results,
        os.path.join(RESULTS_DIR, "diag_vs_offdiag_abs_cov_compare.png"),
    )

    # ---------- save summary table ----------
    with open(os.path.join(RESULTS_DIR, "selected_offdiag_vs_temperature.txt"), "w") as f:
        f.write("# selected off-diagonal unnormalized |covariance| vs temperature\n")
        f.write("# pair_label")
        for t in temp_order:
            f.write(f"  {t}")
        f.write("\n")
        for (i, j) in SELECTED_OFFDIAG:
            lab = f"L{FREQ_BINS[i][0]}-{FREQ_BINS[i][1]}_R{FREQ_BINS[j][0]}-{FREQ_BINS[j][1]}"
            f.write(lab)
            for t in temp_order:
                f.write(f"  {results[t]['abs_cov_mat'][i, j]:.8e}")
            f.write("\n")

    with open(os.path.join(RESULTS_DIR, "meta.txt"), "w") as f:
        f.write(f"GROUP_ID = {GROUP_ID}\n")
        f.write(f"X_INTERFACE = {X_INTERFACE}\n")
        f.write(f"WINDOW_W = {WINDOW_W}\n")
        f.write(f"DT_FS = {DT_FS}\n")
        f.write(f"DUMP_EVERY = {DUMP_EVERY}\n")
        f.write(f"NPERSEG = {NPERSEG}\n")
        f.write(f"FREQ_BINS = {FREQ_BINS}\n")
        f.write(f"SELECTED_OFFDIAG = {SELECTED_OFFDIAG}\n")
        f.write(f"cov_vmin = {cov_vmin:.8e}\n")
        f.write(f"cov_vmax = {cov_vmax:.8e}\n")
        f.write(f"abs_cov_vmax = {abs_cov_vmax:.8e}\n")
        for t in temp_order:
            r = results[t]
            f.write(
                f"{t}: left_n={r['left_n']}, right_n={r['right_n']}, "
                f"left_bounds={r['left_bounds']}, right_bounds={r['right_bounds']}\n"
            )

    print("[OK] saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()