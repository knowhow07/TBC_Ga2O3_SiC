#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, csd, welch

# =========================
# USER SETTINGS
# =========================
SNAPSHOT_XYZ = "model.xyz"
CACHE_NPY    = "velocity_g2.npy"   # dumb velocity npy for dumped group
GROUP_ID     = 2

# interface definition
X_INTERFACE = 105.0    # Å
WINDOW_W    = 5.0      # Å

# region definition:
# left  = [X_INTERFACE-WINDOW_W, X_INTERFACE)
# right = (X_INTERFACE, X_INTERFACE+WINDOW_W]

# time sampling
DT_FS      = 1.0       # MD timestep in fs
DUMP_EVERY = 10        # dump_velocity interval in MD steps

# spectral settings
FREQ_MAX_THz = 50.0
NPERSEG      = 4096    # reduce if too long for your data
NOOVERLAP    = None    # default = nperseg//2
DETREND      = "constant"

# frequency bins for comparison
FREQ_BINS = [(0,5), (5,10), (10,15), (15,20), (20,25), (25,30)]

# atom threshold
MIN_ATOMS_PER_REGION = 20

# how to build slab signal:
# "mean_xyz"  : mean over atoms, then coherence for x/y/z separately and average
# "mean_vx"   : only slab-averaged vx
# "sum_ke"    : summed v^2 signal per slab (less standard for coherence)
SIGNAL_MODE = "mean_xyz"

# output prefix
OUT_PREFIX = "coherence_g2_near_interface"
# =========================


def read_snapshot_xyz(fname):
    """Read extended XYZ and return x, group(if present else None), Lx."""
    with open(fname, "r") as f:
        N = int(f.readline().strip())
        header2 = f.readline().strip()
        lines = [f.readline().strip() for _ in range(N)]

    x = np.zeros(N, float)
    g = np.full(N, -1, int)

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 4:
            raise RuntimeError(f"{fname}: bad atom line: {line}")

        try:
            int(parts[0])  # format: id sp x y z [group]
            x[i] = float(parts[2])
            if len(parts) >= 6:
                g[i] = int(parts[5])
        except ValueError:
            # format: sp x y z [group]
            x[i] = float(parts[1])
            if len(parts) == 5:
                g[i] = int(parts[4])

    Lx = None
    m = re.search(r'Lattice="([^"]+)"', header2)
    if m:
        lat = list(map(float, m.group(1).split()))
        if len(lat) >= 1:
            Lx = float(lat[0])

    if np.all(g < 0):
        raise RuntimeError(
            f"{fname}: no group column detected. Need grouped model.xyz."
        )

    return x, g, Lx


def reshape_velocity_npy(npy_file, n_atoms):
    data = np.load(npy_file, mmap_mode="r")
    if data.ndim != 2 or data.shape[1] != 3:
        raise RuntimeError(
            f"{npy_file}: expected shape (total_rows, 3), got {data.shape}"
        )

    total_rows = data.shape[0]
    if total_rows % n_atoms != 0:
        raise RuntimeError(
            f"{npy_file}: row count mismatch.\n"
            f"  total_rows = {total_rows}\n"
            f"  n_atoms(group {GROUP_ID}) = {n_atoms}\n"
            f"  total_rows % n_atoms = {total_rows % n_atoms}\n"
            f"Likely atom-count/order mismatch between model.xyz and velocity_g2.npy"
        )

    n_frames = total_rows // n_atoms
    if n_frames < 10:
        raise RuntimeError(f"{npy_file}: too few frames ({n_frames})")

    return data.reshape(n_frames, n_atoms, 3)


def make_region_indices(x_g, x_interface, window_w):
    left_lo  = x_interface - window_w
    left_hi  = x_interface
    right_lo = x_interface
    right_hi = x_interface + window_w

    mask_left  = (x_g >= left_lo)  & (x_g < left_hi)
    mask_right = (x_g >  right_lo) & (x_g <= right_hi)

    sel_left  = np.where(mask_left)[0]
    sel_right = np.where(mask_right)[0]

    return sel_left, sel_right, (left_lo, left_hi, right_lo, right_hi)


def build_signals(v_region, mode="mean_xyz"):
    """
    v_region shape: (n_frames, n_atoms, 3)

    Returns:
      if mode == mean_xyz:
          dict with keys x,y,z and values 1D time series
      if mode == mean_vx:
          dict with key vx
      if mode == sum_ke:
          dict with key ke
    """
    v_region = np.asarray(v_region)

    if mode == "mean_xyz":
        sig = {
            "x": v_region[:, :, 0].mean(axis=1),
            "y": v_region[:, :, 1].mean(axis=1),
            "z": v_region[:, :, 2].mean(axis=1),
        }
    elif mode == "mean_vx":
        sig = {"vx": v_region[:, :, 0].mean(axis=1)}
    elif mode == "sum_ke":
        sig = {"ke": (v_region**2).sum(axis=(1,2))}
    else:
        raise ValueError("Unknown SIGNAL_MODE")
    return sig


def compute_avg_coherence(sigL_dict, sigR_dict, fs_hz, nperseg, noverlap, detrend):
    """
    Compute coherence for each matching channel and average them.
    """
    keys = [k for k in sigL_dict if k in sigR_dict]
    if not keys:
        raise RuntimeError("No common channels between left/right signals.")

    coh_list = []
    freq_ref = None

    for k in keys:
        f_hz, cxy = coherence(
            sigL_dict[k], sigR_dict[k],
            fs=fs_hz,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend
        )
        if freq_ref is None:
            freq_ref = f_hz
        coh_list.append(cxy)

    coh_avg = np.mean(np.vstack(coh_list), axis=0)
    return freq_ref, coh_avg, keys


def compute_cross_spectrum(sigL_dict, sigR_dict, fs_hz, nperseg, noverlap, detrend):
    """
    Compute average |cross spectrum| and average auto spectra over channels.
    """
    keys = [k for k in sigL_dict if k in sigR_dict]
    csd_list = []
    sll_list = []
    srr_list = []
    freq_ref = None

    for k in keys:
        f_hz, Pxy = csd(
            sigL_dict[k], sigR_dict[k],
            fs=fs_hz,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            scaling="density"
        )
        _, Pxx = welch(
            sigL_dict[k],
            fs=fs_hz,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            scaling="density"
        )
        _, Pyy = welch(
            sigR_dict[k],
            fs=fs_hz,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            scaling="density"
        )

        if freq_ref is None:
            freq_ref = f_hz

        csd_list.append(np.abs(Pxy))
        sll_list.append(Pxx)
        srr_list.append(Pyy)

    return (
        freq_ref,
        np.mean(np.vstack(csd_list), axis=0),
        np.mean(np.vstack(sll_list), axis=0),
        np.mean(np.vstack(srr_list), axis=0),
    )


def band_average(freq, y, f1, f2):
    m = (freq >= f1) & (freq < f2)
    if np.sum(m) < 2:
        return np.nan
    return np.trapz(y[m], freq[m]) / (f2 - f1)


def save_bin_table(txtfile, bins, coh_vals, csd_vals):
    with open(txtfile, "w") as f:
        f.write("# f_low_THz f_high_THz avg_coherence avg_abs_cross_spectrum\n")
        for (f1, f2), c1, c2 in zip(bins, coh_vals, csd_vals):
            f.write(f"{f1:8.3f} {f2:8.3f} {c1:16.8e} {c2:16.8e}\n")


def save_meta(txtfile, meta_lines):
    with open(txtfile, "w") as f:
        for line in meta_lines:
            f.write(line.rstrip() + "\n")


def main():
    if not os.path.exists(SNAPSHOT_XYZ):
        raise FileNotFoundError(SNAPSHOT_XYZ)
    if not os.path.exists(CACHE_NPY):
        raise FileNotFoundError(CACHE_NPY)

    x_all, g_all, Lx = read_snapshot_xyz(SNAPSHOT_XYZ)
    idx_g = np.where(g_all == GROUP_ID)[0]
    if idx_g.size == 0:
        raise RuntimeError(f"No atoms with group == {GROUP_ID} in {SNAPSHOT_XYZ}")

    # x positions of dumped group atoms, in grouped snapshot order
    x_g = x_all[idx_g]

    sel_left, sel_right, bounds = make_region_indices(x_g, X_INTERFACE, WINDOW_W)
    left_lo, left_hi, right_lo, right_hi = bounds

    print(f"Lx = {Lx}")
    print(f"group {GROUP_ID} atom count = {idx_g.size}")
    print(f"Left near-interface slab  [{left_lo:.2f}, {left_hi:.2f}) Å: n={sel_left.size}")
    print(f"Right near-interface slab ({right_lo:.2f}, {right_hi:.2f}] Å: n={sel_right.size}")

    if sel_left.size < MIN_ATOMS_PER_REGION or sel_right.size < MIN_ATOMS_PER_REGION:
        raise RuntimeError(
            f"Too few atoms in selected slab(s): left={sel_left.size}, right={sel_right.size}"
        )

    v = reshape_velocity_npy(CACHE_NPY, n_atoms=idx_g.size)
    n_frames = v.shape[0]
    dt_s = DT_FS * DUMP_EVERY * 1e-15
    fs_hz = 1.0 / dt_s

    if NPERSEG > n_frames:
        nperseg = max(256, 2 ** int(np.floor(np.log2(n_frames))))
        print(f"[WARN] NPERSEG reduced from {NPERSEG} to {nperseg}")
    else:
        nperseg = NPERSEG

    noverlap = NOOVERLAP if NOOVERLAP is not None else nperseg // 2

    print(f"frames      = {n_frames}")
    print(f"dt_sample   = {dt_s*1e15:.1f} fs")
    print(f"nperseg     = {nperseg}")
    print(f"noverlap    = {noverlap}")
    print(f"signal mode = {SIGNAL_MODE}")

    # build left/right slab signals
    vL = np.asarray(v[:, sel_left, :], dtype=np.float64)
    vR = np.asarray(v[:, sel_right, :], dtype=np.float64)

    sigL = build_signals(vL, SIGNAL_MODE)
    sigR = build_signals(vR, SIGNAL_MODE)

    # coherence
    f_hz, coh, used_keys = compute_avg_coherence(
        sigL, sigR, fs_hz, nperseg, noverlap, DETREND
    )
    f_thz = f_hz * 1e-12

    # cross spectrum
    f2_hz, abs_csd, sll, srr = compute_cross_spectrum(
        sigL, sigR, fs_hz, nperseg, noverlap, DETREND
    )
    f2_thz = f2_hz * 1e-12

    # limit frequency range
    m1 = f_thz <= FREQ_MAX_THz
    f_thz = f_thz[m1]
    coh = coh[m1]

    m2 = f2_thz <= FREQ_MAX_THz
    f2_thz = f2_thz[m2]
    abs_csd = abs_csd[m2]
    sll = sll[m2]
    srr = srr[m2]

    # bin averages
    coh_bins = [band_average(f_thz, coh, f1, f2) for f1, f2 in FREQ_BINS]
    csd_bins = [band_average(f2_thz, abs_csd, f1, f2) for f1, f2 in FREQ_BINS]

    # save txt outputs
    save_bin_table(
        f"{OUT_PREFIX}_bin_table.txt",
        FREQ_BINS, coh_bins, csd_bins
    )

    meta = [
        f"SNAPSHOT_XYZ = {SNAPSHOT_XYZ}",
        f"CACHE_NPY    = {CACHE_NPY}",
        f"GROUP_ID     = {GROUP_ID}",
        f"X_INTERFACE  = {X_INTERFACE}",
        f"WINDOW_W     = {WINDOW_W}",
        f"left slab    = [{left_lo:.3f}, {left_hi:.3f}) Å, n={sel_left.size}",
        f"right slab   = ({right_lo:.3f}, {right_hi:.3f}] Å, n={sel_right.size}",
        f"DT_FS        = {DT_FS}",
        f"DUMP_EVERY   = {DUMP_EVERY}",
        f"dt_sample_fs = {dt_s*1e15:.3f}",
        f"n_frames     = {n_frames}",
        f"nperseg      = {nperseg}",
        f"noverlap     = {noverlap}",
        f"SIGNAL_MODE  = {SIGNAL_MODE}",
        f"used_keys    = {used_keys}",
    ]
    save_meta(f"{OUT_PREFIX}_meta.txt", meta)

    # save raw spectrum txt
    raw = np.column_stack([f_thz, coh])
    np.savetxt(
        f"{OUT_PREFIX}_coherence_spectrum.txt",
        raw,
        header="freq_THz coherence_avg"
    )

    raw2 = np.column_stack([f2_thz, abs_csd, sll, srr])
    np.savetxt(
        f"{OUT_PREFIX}_cross_spectrum.txt",
        raw2,
        header="freq_THz abs_cross_spectrum auto_left auto_right"
    )

    # ---- figure 1: coherence spectrum ----
    plt.figure(figsize=(8, 6))
    plt.plot(f_thz, coh, lw=2, label="Left-right coherence")
    for f1, f2 in FREQ_BINS:
        plt.axvline(f1, ls="--", lw=0.8, alpha=0.4, color="gray")
    plt.axvline(FREQ_BINS[-1][1], ls="--", lw=0.8, alpha=0.4, color="gray")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Coherence")
    plt.title(
        f"Cross-interface coherence (group {GROUP_ID})\n"
        f"Left [{left_lo:.1f},{left_hi:.1f}) Å vs Right ({right_lo:.1f},{right_hi:.1f}] Å"
    )
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_coherence.png", dpi=300)
    plt.close()

    # ---- figure 2: abs cross spectrum ----
    plt.figure(figsize=(8, 6))
    plt.plot(f2_thz, abs_csd, lw=2, label="|Cross spectrum|")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Average |cross spectrum|")
    plt.title(
        f"Cross-interface spectral correlation (group {GROUP_ID})\n"
        f"Left [{left_lo:.1f},{left_hi:.1f}) Å vs Right ({right_lo:.1f},{right_hi:.1f}] Å"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_cross_spectrum.png", dpi=300)
    plt.close()

    # ---- figure 3: bin-averaged coherence ----
    labels = [f"{f1}-{f2}" for f1, f2 in FREQ_BINS]
    x = np.arange(len(labels))

    plt.figure(figsize=(8, 6))
    plt.bar(x, coh_bins)
    plt.xticks(x, labels)
    plt.xlabel("Frequency bin (THz)")
    plt.ylabel("Average coherence")
    plt.ylim(0, 0.8)
    plt.title("Bin-averaged cross-interface coherence")
    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_coherence_bins.png", dpi=300)
    plt.close()

    print(f"[OK] Saved:")
    print(f"  {OUT_PREFIX}_coherence.png")
    print(f"  {OUT_PREFIX}_cross_spectrum.png")
    print(f"  {OUT_PREFIX}_coherence_bins.png")
    print(f"  {OUT_PREFIX}_bin_table.txt")
    print(f"  {OUT_PREFIX}_meta.txt")
    print(f"  {OUT_PREFIX}_coherence_spectrum.txt")
    print(f"  {OUT_PREFIX}_cross_spectrum.txt")


if __name__ == "__main__":
    main()