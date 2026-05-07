#!/usr/bin/env python3
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif"]

plt.rcParams.update({'mathtext.default': 'regular'})
mpl.rcParams["font.size"] = 18

# ============================================================
# User settings
# ============================================================
GROUP_ID = 4   # change to 3 or 5 if needed

TEMP_PATHS = {
    300: "./300k/results",
    600: "./600k-fix/results",
    900: "./900k/results",
}

# add near the top, after FIG_DPI = 300
EV_PER_PS_A2_TO_MW_PER_M2 = 1.602176634e7

# Frequency bins in THz
BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]
# BINS = [(0, 10), (10, 20), (20, 30)]

# Use smoothed G_like per area from previous results
# columns in group_<gid>_spectral_data.txt:
# revised group_<gid>_spectral_data.txt columns:
# 0 omega_THz
# 1 Jin_native
# 2 Jout_native
# 3 Jsum_native
# 4 G_native
# 5 G_native_smoothed
# 6 G_paper_MW
# 7 G_paper_smoothed_MW
# 8 cumulative_integral_native
# USE_COLUMN = 7
USE_COLUMN = 5

# Clip negative values to zero before integration
CLIP_NEGATIVE = True

OUT_DIR = "results_freq_bin_contrib"
FIG_DPI = 300


# ============================================================
# Helpers
# ============================================================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_spectral_data(file_path):
    arr = np.loadtxt(file_path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] <= USE_COLUMN:
        raise RuntimeError(f"Unexpected format in {file_path}; only {arr.shape[1]} columns found")
    omega = arr[:, 0]
    G = arr[:, USE_COLUMN]
    return omega, G


def interp_to_ref_grid(ref_x, x, y):
    return np.interp(ref_x, x, y)


def integrate_bin(x, y, lo, hi):
    mask = (x >= lo) & (x <= hi)
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(np.trapz(y[mask], x[mask]))


def bin_label(lo, hi):
    return f"{lo}-{hi}"


# ============================================================
# Main
# ============================================================
def main():
    out_dir = Path(OUT_DIR)
    ensure_dir(out_dir)

    loaded = {}
    for T, folder in TEMP_PATHS.items():
        fp = Path(folder) / f"group_{GROUP_ID}_spectral_data.txt"
        if not fp.is_file():
            warnings.warn(f"Missing file for {T} K: {fp}")
            continue
        omega, G = load_spectral_data(fp)
        G = G * EV_PER_PS_A2_TO_MW_PER_M2
        loaded[T] = {"omega": omega, "G": G, "file": str(fp)}
        print(f"[OK] Loaded {T} K from {fp}")

    if len(loaded) == 0:
        raise RuntimeError("No valid spectral data files found.")

    temps = sorted(loaded.keys())
    ref_T = temps[0]
    omega_ref = loaded[ref_T]["omega"]

    # interpolate all temperatures to common grid
    contrib = {}   # contrib[T] = [bin integrals...]
    total = {}

    for T in temps:
        y = interp_to_ref_grid(omega_ref, loaded[T]["omega"], loaded[T]["G"])
        if CLIP_NEGATIVE:
            y = np.maximum(y, 0.0)

        vals = []
        for lo, hi in BINS:
            vals.append(integrate_bin(omega_ref, y, lo, hi))
        contrib[T] = vals
        total[T] = sum(vals)

    # ============================================================
    # Save txt summary
    # ============================================================
    txt_out = out_dir / f"group_{GROUP_ID}_freq_bin_contributions.txt"
    with open(txt_out, "w") as f:
        f.write("# Frequency-bin contributions of g(omega) in paper unit\n")
        
        f.write(f"# group_id = {GROUP_ID}\n")
        f.write(f"# clip_negative = {CLIP_NEGATIVE}\n")
        f.write(f"# source_column = {USE_COLUMN} (G_like_per_A2_smoothed, converted to MW m^-2 K^-1 THz^-1)\n")
        f.write("#\n")
        f.write("# Bin definitions (THz):\n")
        for lo, hi in BINS:
            f.write(f"#   {lo:>5.1f}-{hi:<5.1f}\n")
        f.write("#\n")
        f.write("# T_K  total_integral  " + "  ".join([f"bin_{bin_label(lo,hi)}" for lo, hi in BINS]) + "\n")

        for T in temps:
            vals = contrib[T]
            f.write(
                f"{T:6.1f}  {total[T]:.10e}  " +
                "  ".join([f"{v:.10e}" for v in vals]) + "\n"
            )

        f.write("\n# Fraction by bin\n")
        f.write("# T_K  " + "  ".join([f"frac_{bin_label(lo,hi)}" for lo, hi in BINS]) + "\n")
        for T in temps:
            vals = contrib[T]
            if total[T] > 0:
                frac = [v / total[T] for v in vals]
            else:
                frac = [np.nan] * len(vals)
            f.write(
                f"{T:6.1f}  " +
                "  ".join([f"{v:.10e}" for v in frac]) + "\n"
            )

    # ============================================================
    # Figure 1: stacked absolute contributions by bin
    # ============================================================
    plt.figure(figsize=(8, 6), dpi=FIG_DPI)

    bottoms = np.zeros(len(temps))
    x = np.array(temps, dtype=float)
    width = 80

    for i, (lo, hi) in enumerate(BINS):
        vals = np.array([contrib[T][i] for T in temps], dtype=float)
        plt.bar(x, vals, width=width, bottom=bottoms, label=f"{lo}-{hi} THz")
        bottoms += vals

    plt.xlabel("Temperature (K)")
    plt.ylabel(r"Integrated $g(\omega)$ contribution (MW m$^{-2}$ K$^{-1}$)")
    plt.title(f"Group {GROUP_ID}: G contribution from each frequency bin")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend(title="Frequency bin", loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / f"group_{GROUP_ID}_freq_bin_stacked_absolute.png")
    plt.savefig(out_dir / f"group_{GROUP_ID}_freq_bin_stacked_absolute.pdf", bbox_inches="tight")
    plt.close()

    # ============================================================
    # Figure 2: stacked fractional contributions by bin
    # ============================================================
    plt.figure(figsize=(8, 6), dpi=FIG_DPI)

    bottoms = np.zeros(len(temps))
    x = np.array(temps, dtype=float)

    for i, (lo, hi) in enumerate(BINS):
        vals = np.array([contrib[T][i] for T in temps], dtype=float)
        frac = np.array([vals[j] / total[temps[j]] if total[temps[j]] > 0 else np.nan
                         for j in range(len(temps))], dtype=float)
        plt.bar(x, frac, width=width, bottom=bottoms, label=f"{lo}-{hi} THz")
        bottoms += frac

    plt.xlabel("Temperature (K)")
    plt.ylabel("Fraction of total integrated G contribution")
    plt.title(f"Group {GROUP_ID}: fractional G contribution by frequency bin")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend(title="Frequency bin", loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / f"group_{GROUP_ID}_freq_bin_stacked_fraction.png")
    plt.close()

    # ============================================================
    # Figure 3: grouped bar chart for direct bin-by-bin comparison
    # ============================================================
    plt.figure(figsize=(9, 6), dpi=FIG_DPI)

    labels = [bin_label(lo, hi) for lo, hi in BINS]
    xpos = np.arange(len(BINS))
    nT = len(temps)
    bar_w = 0.22 if nT >= 3 else 0.3

    for k, T in enumerate(temps):
        vals = np.array(contrib[T], dtype=float)
        shift = (k - (nT - 1) / 2.0) * bar_w
        plt.bar(xpos + shift, vals, width=bar_w, label=f"{T} K")

    plt.xticks(xpos, labels)
    plt.xlabel("Frequency bin (THz)")
    plt.ylabel(r"Integrated $g(\omega)$ (MW m$^{-2}$ K$^{-1}$)",fontsize=16)
    # plt.title(f"Group {GROUP_ID}: bin-wise G contribution vs temperature")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"group_{GROUP_ID}_freq_bin_grouped_bar.png")
    plt.savefig(out_dir / f"group_{GROUP_ID}_freq_bin_grouped_bar.pdf", bbox_inches="tight")
    plt.close()

    print("\nSaved:")
    print(f"  {txt_out}")
    print(f"  {out_dir / f'group_{GROUP_ID}_freq_bin_stacked_absolute.png'}")
    print(f"  {out_dir / f'group_{GROUP_ID}_freq_bin_stacked_fraction.png'}")
    print(f"  {out_dir / f'group_{GROUP_ID}_freq_bin_grouped_bar.png'}")


if __name__ == "__main__":
    main()