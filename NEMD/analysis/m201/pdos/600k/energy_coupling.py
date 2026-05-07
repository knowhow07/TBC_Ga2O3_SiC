#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif"]
plt.rcParams.update({"mathtext.default": "regular"})
mpl.rcParams["font.size"] = 15

# =========================
# USER SETTINGS
# =========================
SNAPSHOT_XYZ = "model.xyz"
VELOCITY_OUT = "velocity.out"

GROUP_ID = 2
X_INTERFACE = 105.848  # Å
WINDOW_W = 5.0         # left: [x_int-w, x_int), right: (x_int, x_int+w]

DT_FS = 1.0
DUMP_EVERY = 10
FREQ_MAX_THz = 35.0

MIN_ATOMS_PER_REGION = 50

# IO / MODE
IO_MODE = "npy"  # "out" or "npy"
CACHE_NPY = f"velocity_g{GROUP_ID}.npy"
STRICT = True

# Frequency bins for coupling matrix
FREQ_BINS = [(1, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]

# Optional smoothing of band-energy time series before correlation
ENERGY_SMOOTH_POINTS = 21  # odd integer, 1 = off

# Output names
os.makedirs("couple", exist_ok=True)
OUT_PREFIX = f"couple/bin_coupling_g2_near_interface"
# =========================


def read_snapshot_xyz(fname):
    """Read grouped XYZ and return x positions, group ids, Lx."""
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
            int(parts[0])  # id exists
            x[i] = float(parts[2])
            if len(parts) >= 6:
                g[i] = int(parts[5])
        except ValueError:
            x[i] = float(parts[1])
            if len(parts) == 5:
                g[i] = int(parts[4])

    if np.all(g < 0):
        raise RuntimeError(
            f"{fname}: no group column detected. Use grouped model.xyz."
        )

    lx = None
    m = re.search(r'Lattice="([^"]+)"', header2)
    if m:
        lat = list(map(float, m.group(1).split()))
        if len(lat) >= 1:
            lx = float(lat[0])

    return x, g, lx


def _load_vel_rows_from_out(fname):
    data = np.loadtxt(fname, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 3:
        raise RuntimeError(f"{fname}: expected 3 numeric columns.")
    return data


def _load_vel_rows_from_npy(npy):
    if not os.path.exists(npy):
        raise RuntimeError(f"Missing {npy}. Set IO_MODE='out' once to create it.")
    return np.load(npy, mmap_mode="r")


def read_velocity_rows(mode, out_fname, cache_npy):
    if mode == "out":
        if os.path.exists(cache_npy):
            return np.load(cache_npy, mmap_mode="r")
        data = _load_vel_rows_from_out(out_fname)
        np.save(cache_npy, data)
        return np.load(cache_npy, mmap_mode="r")
    elif mode == "npy":
        return _load_vel_rows_from_npy(cache_npy)
    else:
        raise ValueError("IO_MODE must be 'out' or 'npy'")


def reshape_velocity_rows(data, n_atoms, fname_for_msg, strict=True):
    total_rows = int(data.shape[0])

    if total_rows % n_atoms != 0:
        if strict:
            raise RuntimeError(
                f"{fname_for_msg}: row count mismatch.\n"
                f"  total_rows = {total_rows}\n"
                f"  n_atoms(group {GROUP_ID}) = {n_atoms}\n"
                f"  remainder = {total_rows % n_atoms}"
            )
        rem = total_rows % n_atoms
        print(f"[WARN] dropping last {rem} rows from {fname_for_msg}")
        total_rows -= rem
        data = data[:total_rows]

    n_frames = total_rows // n_atoms
    if n_frames < 2:
        raise RuntimeError(f"{fname_for_msg}: too few frames ({n_frames})")

    return data.reshape(n_frames, n_atoms, 3)


def smooth_1d(y, points):
    if points <= 1:
        return y
    if points % 2 == 0:
        points += 1
    pad = points // 2
    k = np.ones(points, dtype=float) / points
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, k, mode="valid")


def zscore(y):
    y = np.asarray(y, dtype=float)
    mu = y.mean()
    sig = y.std()
    if sig < 1e-30:
        return np.zeros_like(y)
    return (y - mu) / sig


def bandpass_fft_time_series(vx_atoms, dt_s, f1_THz, f2_THz):
    """
    Bandpass-filter atomwise vx time series using FFT mask.

    vx_atoms shape: (T, N)
    returns shape: (T, N)
    """
    tsteps = vx_atoms.shape[0]
    freqs_hz = np.fft.rfftfreq(tsteps, d=dt_s)
    freqs_thz = freqs_hz * 1e-12

    mask = (freqs_thz >= f1_THz) & (freqs_thz < f2_THz)

    V = np.fft.rfft(vx_atoms, axis=0)
    V_filt = np.zeros_like(V)
    V_filt[mask, :] = V[mask, :]
    vx_band = np.fft.irfft(V_filt, n=tsteps, axis=0)

    return vx_band


def compute_band_energy_series(vx_atoms, dt_s, bins, smooth_points=1):
    """
    For each frequency bin:
      1) bandpass filter each atom vx(t)
      2) compute instantaneous band energy proxy = mean(v_bin^2 over atoms)
      3) optionally smooth in time

    returns:
      energies: list of 1D arrays, one per bin
    """
    energies = []
    for (f1, f2) in bins:
        vx_bin = bandpass_fft_time_series(vx_atoms, dt_s, f1, f2)
        e_t = np.mean(vx_bin ** 2, axis=1)
        if smooth_points > 1:
            e_t = smooth_1d(e_t, smooth_points)
        energies.append(e_t)
    return energies


def make_coupling_matrices(left_energy_series, right_energy_series):
    """
    Build:
      corr_mat[i,j] = Pearson corr between left-bin-i energy and right-bin-j energy
      cov_mat[i,j]  = covariance between left-bin-i and right-bin-j
      abs_cov_mat   = abs(cov_mat)
    """
    nb = len(left_energy_series)
    corr_mat = np.zeros((nb, nb), float)
    cov_mat = np.zeros((nb, nb), float)

    for i in range(nb):
        x = np.asarray(left_energy_series[i], dtype=float)
        for j in range(nb):
            y = np.asarray(right_energy_series[j], dtype=float)

            n = min(len(x), len(y))
            xx = x[:n]
            yy = y[:n]

            xz = zscore(xx)
            yz = zscore(yy)

            corr = np.mean(xz * yz)
            cov = np.mean((xx - xx.mean()) * (yy - yy.mean()))

            corr_mat[i, j] = corr
            cov_mat[i, j] = cov

    abs_cov_mat = np.abs(cov_mat)
    return corr_mat, cov_mat, abs_cov_mat


def normalize_matrix(mat):
    m = np.asarray(mat, dtype=float)
    vmax = np.max(np.abs(m))
    if vmax < 1e-30:
        return np.zeros_like(m)
    return m / vmax


def save_matrix_txt(fname, bins, mat, header_name):
    labels = [f"{a}-{b}" for a, b in bins]
    with open(fname, "w") as f:
        f.write(f"# {header_name}\n")
        f.write("# rows = left bins, cols = right bins\n")
        f.write("# bins: " + "  ".join(labels) + "\n")
        for i, lab in enumerate(labels):
            row = " ".join(f"{mat[i, j]:.8e}" for j in range(len(labels)))
            f.write(f"{lab:>8s}  {row}\n")


def plot_heatmap(mat, bins, title, out_png, cmap="viridis", vmin=None, vmax=None):
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
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white" if abs(mat[i, j]) > 0.5 else "black", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.ax.set_ylabel("Value", rotation=270, labelpad=15)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_band_energy_series(left_series, right_series, bins, out_png):
    labels = [f"{a}-{b}" for a, b in bins]
    nb = len(bins)

    fig, axes = plt.subplots(nb, 1, figsize=(8.0, 2.2 * nb), sharex=True)
    if nb == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        ax.plot(left_series[k], label=f"Left {labels[k]}", linewidth=1.5)
        ax.plot(right_series[k], label=f"Right {labels[k]}", linewidth=1.5)
        ax.legend(loc="upper right", fontsize=12)
        ax.set_ylabel("Band energy")
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Frame index")
    fig.suptitle("Band-energy time series used for coupling matrix", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_diag_offdiag(mat, bins, out_png):
    labels = [f"{a}-{b}" for a, b in bins]
    diag = np.diag(mat)
    off = []
    for i in range(mat.shape[0]):
        vals = np.delete(mat[i, :], i)
        off.append(np.mean(vals))
    off = np.array(off)

    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, diag, width=w, label="Diagonal (same-bin)")
    ax.bar(x + w/2, off, width=w, label="Mean off-diagonal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Frequency bin (THz)")
    ax.set_ylabel("Coupling metric")
    ax.set_title("Diagonal vs off-diagonal bin coupling")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    x_all, g_all, lx = read_snapshot_xyz(SNAPSHOT_XYZ)
    idx_g = np.where(g_all == GROUP_ID)[0]
    if idx_g.size == 0:
        raise RuntimeError(f"No atoms with group == {GROUP_ID} in {SNAPSHOT_XYZ}")

    x_g = x_all[idx_g]

    left_lo = X_INTERFACE - WINDOW_W
    left_hi = X_INTERFACE
    right_lo = X_INTERFACE
    right_hi = X_INTERFACE + WINDOW_W

    mask_left = (x_g >= left_lo) & (x_g < left_hi)
    mask_right = (x_g > right_lo) & (x_g <= right_hi)

    sel_left = np.where(mask_left)[0]
    sel_right = np.where(mask_right)[0]

    print(f"Lx = {lx}")
    print(f"group {GROUP_ID} atom count = {idx_g.size}")
    print(f"Left slab  [{left_lo:.3f}, {left_hi:.3f}) Å: n={sel_left.size}")
    print(f"Right slab ({right_lo:.3f}, {right_hi:.3f}] Å: n={sel_right.size}")

    if sel_left.size < MIN_ATOMS_PER_REGION or sel_right.size < MIN_ATOMS_PER_REGION:
        raise RuntimeError(
            f"Too few atoms in slab(s): left={sel_left.size}, right={sel_right.size}"
        )

    rows = read_velocity_rows(IO_MODE, VELOCITY_OUT, CACHE_NPY)
    v = reshape_velocity_rows(
        rows,
        n_atoms=idx_g.size,
        fname_for_msg=(VELOCITY_OUT if IO_MODE == "out" else CACHE_NPY),
        strict=STRICT,
    )

    dt_s = DT_FS * DUMP_EVERY * 1e-15
    print(f"Velocity frames = {v.shape[0]}, dt_sample = {dt_s * 1e15:.1f} fs")

    # Use vx only
    vx_left = np.asarray(v[:, sel_left, 0], dtype=np.float64)   # (T, Nl)
    vx_right = np.asarray(v[:, sel_right, 0], dtype=np.float64) # (T, Nr)

    # Subtract slab COM vx at each frame
    vx_left -= vx_left.mean(axis=1, keepdims=True)
    vx_right -= vx_right.mean(axis=1, keepdims=True)

    # Remove any residual time mean per atom
    vx_left -= vx_left.mean(axis=0, keepdims=True)
    vx_right -= vx_right.mean(axis=0, keepdims=True)

    # Band-energy time series per bin
    left_energy = compute_band_energy_series(
        vx_left, dt_s, FREQ_BINS, smooth_points=ENERGY_SMOOTH_POINTS
    )
    right_energy = compute_band_energy_series(
        vx_right, dt_s, FREQ_BINS, smooth_points=ENERGY_SMOOTH_POINTS
    )

    # Coupling matrices
    corr_mat, cov_mat, abs_cov_mat = make_coupling_matrices(left_energy, right_energy)

    # Normalized abs-cov matrix for plotting/comparison
    norm_abs_cov = normalize_matrix(abs_cov_mat)

    # Save txt
    save_matrix_txt(f"{OUT_PREFIX}_corr_matrix.txt", FREQ_BINS, corr_mat, "corr_matrix")
    save_matrix_txt(f"{OUT_PREFIX}_cov_matrix.txt", FREQ_BINS, cov_mat, "cov_matrix")
    save_matrix_txt(f"{OUT_PREFIX}_abs_cov_matrix.txt", FREQ_BINS, abs_cov_mat, "abs_cov_matrix")
    save_matrix_txt(f"{OUT_PREFIX}_norm_abs_cov_matrix.txt", FREQ_BINS, norm_abs_cov, "norm_abs_cov_matrix")

    # Meta
    with open(f"{OUT_PREFIX}_meta.txt", "w") as f:
        f.write(f"SNAPSHOT_XYZ = {SNAPSHOT_XYZ}\n")
        f.write(f"VELOCITY_OUT = {VELOCITY_OUT}\n")
        f.write(f"IO_MODE = {IO_MODE}\n")
        f.write(f"CACHE_NPY = {CACHE_NPY}\n")
        f.write(f"GROUP_ID = {GROUP_ID}\n")
        f.write(f"X_INTERFACE = {X_INTERFACE}\n")
        f.write(f"WINDOW_W = {WINDOW_W}\n")
        f.write(f"left slab = [{left_lo:.3f}, {left_hi:.3f}) Å, n={sel_left.size}\n")
        f.write(f"right slab = ({right_lo:.3f}, {right_hi:.3f}] Å, n={sel_right.size}\n")
        f.write(f"DT_FS = {DT_FS}\n")
        f.write(f"DUMP_EVERY = {DUMP_EVERY}\n")
        f.write(f"dt_sample_fs = {dt_s * 1e15:.3f}\n")
        f.write(f"n_frames = {v.shape[0]}\n")
        f.write(f"ENERGY_SMOOTH_POINTS = {ENERGY_SMOOTH_POINTS}\n")
        f.write("FREQ_BINS = " + str(FREQ_BINS) + "\n")

    # Plots
    plot_heatmap(
        corr_mat,
        FREQ_BINS,
        title="Bin-resolved cross-interface coupling matrix\n(correlation of band-energy fluctuations)",
        out_png=f"{OUT_PREFIX}_corr_matrix.png",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )

    plot_heatmap(
        norm_abs_cov,
        FREQ_BINS,
        title="Bin-resolved cross-interface coupling matrix\n(normalized |covariance| of band-energy fluctuations)",
        out_png=f"{OUT_PREFIX}_norm_abs_cov_matrix.png",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    plot_diag_offdiag(
        norm_abs_cov,
        FREQ_BINS,
        out_png=f"{OUT_PREFIX}_diag_vs_offdiag.png",
    )

    plot_band_energy_series(
        left_energy,
        right_energy,
        FREQ_BINS,
        out_png=f"{OUT_PREFIX}_band_energy_series.png",
    )

    print("[OK] Saved:")
    print(f"  {OUT_PREFIX}_corr_matrix.txt")
    print(f"  {OUT_PREFIX}_cov_matrix.txt")
    print(f"  {OUT_PREFIX}_abs_cov_matrix.txt")
    print(f"  {OUT_PREFIX}_norm_abs_cov_matrix.txt")
    print(f"  {OUT_PREFIX}_meta.txt")
    print(f"  {OUT_PREFIX}_corr_matrix.png")
    print(f"  {OUT_PREFIX}_norm_abs_cov_matrix.png")
    print(f"  {OUT_PREFIX}_diag_vs_offdiag.png")
    print(f"  {OUT_PREFIX}_band_energy_series.png")


if __name__ == "__main__":
    main()