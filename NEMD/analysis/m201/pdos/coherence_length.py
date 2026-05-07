#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence, csd, welch, savgol_filter

# =========================
# MULTI-LENGTH INPUTS
# =========================
LENGTH_INPUTS = {
    "L=200A": {
        "model_xyz": "./300k/model.xyz",
        "velocity_npy": "./300k/velocity_g2.npy",
    },
    "L=600A": {
        "model_xyz": "./60+30/model.xyz",
        "velocity_npy": "./60+30/velocity_g2.npy",
    },
    # add more lengths here if needed
    # "L=400A": {
    #     "model_xyz": "./18/model.xyz",
    #     "velocity_npy": "./18/velocity_g2.npy",
    # },
}

GROUP_ID = 2

# interface definition
WINDOW_W = 5.0
INTERFACE_SPECIES_LEFT = "Si"
INTERFACE_SPECIES_RIGHT = "O"
EPS = 1e-9

# time sampling
DT_FS = 1.0
DUMP_EVERY = 10

# spectral settings
FREQ_MAX_THz = 50.0
NPERSEG = 4096
NOOVERLAP = None
DETREND = "constant"

# compare in bins
FREQ_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]

MIN_ATOMS_PER_REGION = 20

# use only vx and subtract slab COM vx before building signal
SIGNAL_MODE = "mean_vx_demean"

RESULTS_DIR = "results_length"
os.makedirs(RESULTS_DIR, exist_ok=True)

# plotting / smoothing controls
SMOOTH_WINDOW = 21      # odd integer; set <= 1 to disable
SMOOTH_POLY = 3
NORMALIZE_CSD_FOR_PLOT = True
USE_LOG_CSD_PLOT = True
# =========================

plt.rcParams["font.size"] = 14


def read_snapshot_xyz(fname):
    """Read extended XYZ and return x, species, group(if present else None), Lx."""
    with open(fname, "r") as f:
        N = int(f.readline().strip())
        header2 = f.readline().strip()
        lines = [f.readline().strip() for _ in range(N)]

    x = np.zeros(N, float)
    g = np.full(N, -1, int)
    species = []

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 4:
            raise RuntimeError(f"{fname}: bad atom line: {line}")

        try:
            int(parts[0])  # format: id sp x y z [group]
            species.append(parts[1])
            x[i] = float(parts[2])
            if len(parts) >= 6:
                g[i] = int(parts[5])
        except ValueError:
            # format: sp x y z [group]
            species.append(parts[0])
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

    return x, np.array(species), g, Lx


def detect_interface_from_species_change(x_all, species_all, Lx=None,
                                         left_species="Si", right_species="O"):
    """
    Detect interface position from species change along x:
    interface = midpoint between the last left_species atom and
    the first right_species atom to its right.
    """
    order = np.argsort(x_all)
    xs = np.asarray(x_all)[order]
    ss = np.asarray(species_all)[order]

    mask_left = (ss == left_species)
    mask_right = (ss == right_species)

    if not np.any(mask_left):
        raise RuntimeError(f"No {left_species} atoms found in model.xyz")
    if not np.any(mask_right):
        raise RuntimeError(f"No {right_species} atoms found in model.xyz")

    x_last_left = np.max(xs[mask_left])

    x_right_candidates = xs[mask_right & (xs >= x_last_left - EPS)]
    if x_right_candidates.size == 0:
        raise RuntimeError(
            f"Could not find any {right_species} atoms to the right of the last {left_species} atom"
        )

    x_first_right = np.min(x_right_candidates)
    x_interface = 0.5 * (x_last_left + x_first_right)

    if Lx is not None and not (-1e-6 <= x_interface <= Lx + 1e-6):
        raise RuntimeError(
            f"Detected interface x={x_interface:.6f} Å is outside box [0, {Lx:.6f}]"
        )

    return x_interface, x_last_left, x_first_right


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
    left_lo = x_interface - window_w
    left_hi = x_interface
    right_lo = x_interface
    right_hi = x_interface + window_w

    mask_left = (x_g >= left_lo) & (x_g < left_hi)
    mask_right = (x_g > right_lo) & (x_g <= right_hi)

    sel_left = np.where(mask_left)[0]
    sel_right = np.where(mask_right)[0]

    return sel_left, sel_right, (left_lo, left_hi, right_lo, right_hi)


def build_signals(v_region, mode="mean_vx_demean"):
    """
    v_region shape: (n_frames, n_atoms, 3)

    mean_vx_demean:
      use atomwise vx fluctuations after subtracting slab COM vx at each frame
      then average atomwise coherence later, not slab-RMS coherence
    """
    v_region = np.asarray(v_region, dtype=np.float64)

    if mode == "mean_vx_demean":
        vx = v_region[:, :, 0]
        vx_com = vx.mean(axis=1, keepdims=True)
        vx_fluc = vx - vx_com
        return {"vx_atoms": vx_fluc}

    elif mode == "mean_vx":
        return {"vx_atoms": v_region[:, :, 0]}

    else:
        raise ValueError("Unknown SIGNAL_MODE")


def compute_avg_coherence(sigL_dict, sigR_dict, fs_hz, nperseg, noverlap, detrend):
    """
    Compute coherence from atomwise vx signals and average over matched left-right pairs.
    To avoid O(N_left*N_right), pair atoms by sorted x order and use min(nL, nR) pairs.
    """
    vxL = sigL_dict["vx_atoms"]
    vxR = sigR_dict["vx_atoms"]

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
    return freq_ref, coh_avg, [f"vx_pair_avg_npairs={npair}"]


def compute_cross_spectrum(sigL_dict, sigR_dict, fs_hz, nperseg, noverlap, detrend):
    vxL = sigL_dict["vx_atoms"]
    vxR = sigR_dict["vx_atoms"]

    nL = vxL.shape[1]
    nR = vxR.shape[1]
    npair = min(nL, nR)
    if npair < 2:
        raise RuntimeError(f"Too few matched pairs: nL={nL}, nR={nR}")

    csd_list = []
    sll_list = []
    srr_list = []
    freq_ref = None

    for i in range(npair):
        f_hz, Pxy = csd(
            vxL[:, i], vxR[:, i],
            fs=fs_hz,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            scaling="density"
        )
        _, Pxx = welch(
            vxL[:, i],
            fs=fs_hz,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            scaling="density"
        )
        _, Pyy = welch(
            vxR[:, i],
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


def analyze_one_length(length_label, model_xyz, velocity_npy):
    x_all, species_all, g_all, Lx = read_snapshot_xyz(model_xyz)
    idx_g = np.where(g_all == GROUP_ID)[0]
    if idx_g.size == 0:
        raise RuntimeError(f"[{length_label}] No atoms with group == {GROUP_ID} in {model_xyz}")

    x_interface, x_last_si, x_first_o = detect_interface_from_species_change(
        x_all, species_all, Lx=Lx,
        left_species=INTERFACE_SPECIES_LEFT,
        right_species=INTERFACE_SPECIES_RIGHT
    )

    x_g = x_all[idx_g]
    sel_left, sel_right, bounds = make_region_indices(x_g, x_interface, WINDOW_W)
    left_lo, left_hi, right_lo, right_hi = bounds

    if sel_left.size < MIN_ATOMS_PER_REGION or sel_right.size < MIN_ATOMS_PER_REGION:
        raise RuntimeError(
            f"[{length_label}] Too few atoms in selected slab(s): left={sel_left.size}, right={sel_right.size}"
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

    sigL = build_signals(vL, SIGNAL_MODE)
    sigR = build_signals(vR, SIGNAL_MODE)

    f_hz, coh, used_keys = compute_avg_coherence(
        sigL, sigR, fs_hz, nperseg, noverlap, DETREND
    )
    f2_hz, abs_csd, sll, srr = compute_cross_spectrum(
        sigL, sigR, fs_hz, nperseg, noverlap, DETREND
    )

    f_thz = f_hz * 1e-12
    f2_thz = f2_hz * 1e-12

    m1 = f_thz <= FREQ_MAX_THz
    f_thz = f_thz[m1]
    coh = coh[m1]

    m2 = f2_thz <= FREQ_MAX_THz
    f2_thz = f2_thz[m2]
    abs_csd = abs_csd[m2]
    sll = sll[m2]
    srr = srr[m2]

    coh_smooth = smooth_curve(coh, SMOOTH_WINDOW, SMOOTH_POLY)
    abs_csd_smooth = smooth_curve(abs_csd, SMOOTH_WINDOW, SMOOTH_POLY)
    sll_smooth = smooth_curve(sll, SMOOTH_WINDOW, SMOOTH_POLY)
    srr_smooth = smooth_curve(srr, SMOOTH_WINDOW, SMOOTH_POLY)

    coh_bins = [band_average(f_thz, coh_smooth, f1, f2) for f1, f2 in FREQ_BINS]
    csd_bins = [band_average(f2_thz, abs_csd_smooth, f1, f2) for f1, f2 in FREQ_BINS]
    sll_bins = [band_average(f2_thz, sll_smooth, f1, f2) for f1, f2 in FREQ_BINS]
    srr_bins = [band_average(f2_thz, srr_smooth, f1, f2) for f1, f2 in FREQ_BINS]

    return {
        "label": length_label,
        "Lx": Lx,
        "x_interface": x_interface,
        "x_last_left_species": x_last_si,
        "x_first_right_species": x_first_o,
        "n_frames": n_frames,
        "left_n": sel_left.size,
        "right_n": sel_right.size,
        "left_bounds": (left_lo, left_hi),
        "right_bounds": (right_lo, right_hi),
        "freq": f_thz,
        "coh": coh,
        "coh_smooth": coh_smooth,
        "freq2": f2_thz,
        "abs_csd": abs_csd,
        "abs_csd_smooth": abs_csd_smooth,
        "sll": sll,
        "sll_smooth": sll_smooth,
        "srr": srr,
        "srr_smooth": srr_smooth,
        "coh_bins": np.array(coh_bins),
        "csd_bins": np.array(csd_bins),
        "sll_bins": np.array(sll_bins),
        "srr_bins": np.array(srr_bins),
        "used_keys": used_keys,
    }


def main():
    results = {}
    for length_label, paths in LENGTH_INPUTS.items():
        print(f"[INFO] analyzing {length_label}")
        results[length_label] = analyze_one_length(
            length_label,
            paths["model_xyz"],
            paths["velocity_npy"]
        )
        r = results[length_label]
        print(
            f"  Lx={r['Lx']:.3f} Å, "
            f"x_interface={r['x_interface']:.3f} Å, "
            f"{INTERFACE_SPECIES_LEFT}_last={r['x_last_left_species']:.3f} Å, "
            f"{INTERFACE_SPECIES_RIGHT}_first={r['x_first_right_species']:.3f} Å, "
            f"left_n={r['left_n']}, right_n={r['right_n']}"
        )

    labels = list(LENGTH_INPUTS.keys())
    bin_labels = [f"{f1}-{f2}" for f1, f2 in FREQ_BINS]
    x = np.arange(len(bin_labels))
    width = 0.8 / len(labels)

    out_txt = os.path.join(RESULTS_DIR, "coherence_compare_all_columns.txt")
    with open(out_txt, "w") as f:
        header = ["bin"]
        for label in labels:
            header += [
                f"{label}_coh",
                f"{label}_abs_csd",
                f"{label}_auto_left",
                f"{label}_auto_right",
            ]
        f.write("# " + "  ".join(header) + "\n")

        for i, lab in enumerate(bin_labels):
            row = [lab]
            for label in labels:
                r = results[label]
                row += [
                    f"{r['coh_bins'][i]:.8e}",
                    f"{r['csd_bins'][i]:.8e}",
                    f"{r['sll_bins'][i]:.8e}",
                    f"{r['srr_bins'][i]:.8e}",
                ]
            f.write("  ".join(row) + "\n")

    plt.figure(figsize=(10, 6.5))
    for label in labels:
        r = results[label]
        plt.plot(r["freq"], r["coh_smooth"], lw=2.8, label=label)
    for f1, f2 in FREQ_BINS:
        plt.axvline(f1, ls="--", lw=0.8, alpha=0.18, color="gray")
    plt.axvline(FREQ_BINS[-1][1], ls="--", lw=0.8, alpha=0.18, color="gray")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Coherence")
    plt.title("Cross-interface coherence vs bar length")
    plt.xlim(0, FREQ_MAX_THz)
    plt.grid(alpha=0.18, linestyle="--")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "coherence_spectrum_compare.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6.5))
    for label in labels:
        r = results[label]
        y = r["abs_csd_smooth"].copy()
        if NORMALIZE_CSD_FOR_PLOT:
            ymax = np.max(y)
            if ymax > 0:
                y = y / ymax
        plt.plot(r["freq2"], y, lw=2.8, label=label)

    plt.xlabel("Frequency (THz)")
    plt.ylabel("Normalized |cross spectrum|" if NORMALIZE_CSD_FOR_PLOT else "Average |cross spectrum|")
    plt.title("Cross-interface spectral correlation vs bar length")
    plt.xlim(0, FREQ_MAX_THz)
    if USE_LOG_CSD_PLOT and not NORMALIZE_CSD_FOR_PLOT:
        plt.yscale("log")
    plt.grid(alpha=0.18, linestyle="--")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cross_spectrum_compare.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6.5))
    for label in labels:
        r = results[label]
        plt.plot(r["freq2"], r["sll_smooth"], lw=2.6, label=label)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Average auto spectrum (left)")
    plt.title("Left near-interface auto spectrum vs bar length")
    plt.xlim(0, FREQ_MAX_THz)
    plt.grid(alpha=0.18, linestyle="--")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "auto_left_spectrum_compare.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6.5))
    for label in labels:
        r = results[label]
        plt.plot(r["freq2"], r["srr_smooth"], lw=2.6, label=label)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Average auto spectrum (right)")
    plt.title("Right near-interface auto spectrum vs bar length")
    plt.xlim(0, FREQ_MAX_THz)
    plt.grid(alpha=0.18, linestyle="--")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "auto_right_spectrum_compare.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6.5))
    for j, label in enumerate(labels):
        plt.bar(
            x + (j - (len(labels) - 1) / 2) * width,
            results[label]["coh_bins"],
            width=width * 0.92,
            alpha=0.88,
            label=label
        )
    plt.xticks(x, bin_labels)
    plt.xlabel("Frequency bin (THz)")
    plt.ylabel("Average coherence")
    plt.title("Bin-averaged cross-interface coherence vs bar length")
    plt.grid(axis="y", alpha=0.18, linestyle="--")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "coherence_bins_compare.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6.5))
    for j, label in enumerate(labels):
        vals = results[label]["csd_bins"].copy()
        if NORMALIZE_CSD_FOR_PLOT:
            vmax = np.nanmax(vals)
            if vmax > 0:
                vals = vals / vmax
        plt.bar(
            x + (j - (len(labels) - 1) / 2) * width,
            vals,
            width=width * 0.92,
            alpha=0.88,
            label=label
        )
    plt.xticks(x, bin_labels)
    plt.xlabel("Frequency bin (THz)")
    plt.ylabel("Normalized average |cross spectrum|" if NORMALIZE_CSD_FOR_PLOT else "Average |cross spectrum|")
    plt.title("Bin-averaged cross-interface spectral correlation vs bar length")
    plt.grid(axis="y", alpha=0.18, linestyle="--")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cross_spectrum_bins_compare.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6.5))
    for j, label in enumerate(labels):
        plt.bar(
            x + (j - (len(labels) - 1) / 2) * width,
            results[label]["sll_bins"],
            width=width * 0.92,
            alpha=0.88,
            label=label
        )
    plt.xticks(x, bin_labels)
    plt.xlabel("Frequency bin (THz)")
    plt.ylabel("Average auto spectrum (left)")
    plt.title("Bin-averaged left auto spectrum vs bar length")
    plt.grid(axis="y", alpha=0.18, linestyle="--")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "auto_left_bins_compare.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6.5))
    for j, label in enumerate(labels):
        plt.bar(
            x + (j - (len(labels) - 1) / 2) * width,
            results[label]["srr_bins"],
            width=width * 0.92,
            alpha=0.88,
            label=label
        )
    plt.xticks(x, bin_labels)
    plt.xlabel("Frequency bin (THz)")
    plt.ylabel("Average auto spectrum (right)")
    plt.title("Bin-averaged right auto spectrum vs bar length")
    plt.grid(axis="y", alpha=0.18, linestyle="--")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "auto_right_bins_compare.png"), dpi=300)
    plt.close()

    for label in labels:
        r = results[label]
        safe_label = label.replace(" ", "_")
        np.savetxt(
            os.path.join(RESULTS_DIR, f"{safe_label}_coherence_spectrum.txt"),
            np.column_stack([r["freq"], r["coh"]]),
            header="freq_THz coherence"
        )
        np.savetxt(
            os.path.join(RESULTS_DIR, f"{safe_label}_cross_auto_spectra.txt"),
            np.column_stack([r["freq2"], r["abs_csd"], r["sll"], r["srr"]]),
            header="freq_THz abs_cross_spectrum auto_left auto_right"
        )

    print("[OK] saved comparison figures and txt files to:", RESULTS_DIR)


if __name__ == "__main__":
    main()