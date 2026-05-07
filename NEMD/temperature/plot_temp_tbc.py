#!/usr/bin/env python3
import os
import re
import glob
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# User settings
# =========================================================
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif"]

plt.rcParams.update({'mathtext.default': 'regular'})

# label=r'$Ga_2O_3(-201)/SiC(0001)$'

PATHS = {
    r"$Ga_2O_3(100)/SiC(0001)$": {
        300: "./100/300K",
        600: "./100/600K",
        900: "./100/900K",
    },
    r"$Ga_2O_3(\bar{2}01)/SiC(0001)$": {
        300: "./m201/300K",
        600: "./m201/600K",
        900: "./m201/900K",
    },
}

OUT_PNG = "TBC_vs_temperature_100_vs_minus201.png"
OUT_TXT = "TBC_vs_temperature_100_vs_minus201.txt"
FONT_SIZE = 20


# =========================================================
# Helpers
# =========================================================
def find_latest_tbc_result_file(folder):
    """
    Find the latest tbc_results-like file in a folder.
    Priority:
      1) files containing 'tbc_results' in the name
      2) any .txt/.log/.out file that contains 'Average TBC G'
    """
    folder = Path(folder)
    if not folder.exists():
        warnings.warn(f"Missing path: {folder}")
        return None

    candidates = []
    for pat in ["*tbc_results*", "*.txt", "*.log", "*.out"]:
        candidates.extend(folder.glob(pat))

    candidates = sorted(set(candidates))
    if not candidates:
        warnings.warn(f"No candidate result files found in: {folder}")
        return None

    valid = []
    for f in candidates:
        if not f.is_file():
            continue
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "Average TBC G" in txt:
            valid.append(f)

    if not valid:
        warnings.warn(f"No valid result file containing 'Average TBC G' found in: {folder}")
        return None

    def extract_timestamp_from_name(path):
        m = re.search(r"(\d{8}_\d{6})", path.name)
        return m.group(1) if m else ""

    valid_with_timestamp = [f for f in valid if extract_timestamp_from_name(f)]

    if valid_with_timestamp:
        latest = max(valid_with_timestamp, key=extract_timestamp_from_name)
    else:
        latest = max(valid, key=lambda x: x.stat().st_mtime)

    return latest


def parse_tbc_summary(filepath):
    """
    Parse lines like:
    Average TBC G   = 2.217e+04 ± 1.376e+04 MW/(m²·K)
    """
    txt = Path(filepath).read_text(encoding="utf-8", errors="ignore")

    pat = re.compile(
        r"Average\s+TBC\s+G\s*=\s*([0-9.eE+\-]+)\s*±\s*([0-9.eE+\-]+)\s*MW/\(m²·K\)"
    )
    m = pat.search(txt)
    if not m:
        raise RuntimeError(f"Could not parse Average TBC G from {filepath}")

    G = float(m.group(1))
    Gerr = float(m.group(2))
    return G, Gerr


def collect_orientation_data(orientation, temp_map):
    """
    Returns sorted arrays of T, G, Gerr and metadata rows.
    Missing paths/files are skipped with warning.
    """
    rows = []
    for T, folder in temp_map.items():
        if not os.path.isdir(folder):
            warnings.warn(f"[{orientation}] Missing folder for {T} K: {folder}")
            continue

        latest_file = find_latest_tbc_result_file(folder)
        if latest_file is None:
            warnings.warn(f"[{orientation}] No readable TBC result in {folder}")
            continue

        try:
            G, Gerr = parse_tbc_summary(latest_file)
        except Exception as e:
            warnings.warn(f"[{orientation}] Failed parsing {latest_file}: {e}")
            continue

        rows.append((T, G, Gerr, str(latest_file)))

    rows.sort(key=lambda x: x[0])
    return rows


def save_txt(rows_by_orientation, outfile):
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("# TBC vs Temperature comparison\n")
        f.write("# Columns: orientation  T_K  G_MW_m^-2_K^-1  Gerr_MW_m^-2_K^-1  source_file\n")
        for ori, rows in rows_by_orientation.items():
            for T, G, Gerr, src in rows:
                f.write(f"{ori:>5s}  {T:6.1f}  {G:16.8e}  {Gerr:16.8e}  {src}\n")


def plot_tbc_vs_temperature(rows_by_orientation, outpng):
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 1,
    })

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    plotted_any = False
    for ori, rows in rows_by_orientation.items():
        if not rows:
            warnings.warn(f"No valid data for orientation {ori}, skipped in plot.")
            continue

        T = np.array([r[0] for r in rows], dtype=float)
        G = np.array([r[1] for r in rows], dtype=float)
        Gerr = np.array([r[2] for r in rows], dtype=float)

        marker = "o-" if ori == r"$Ga_2O_3(100)/SiC(0001)$" else "s-"
        ax.errorbar(
            T, G, yerr=Gerr,
            fmt=marker,
            linewidth=2,
            markersize=7,
            capsize=4,
            label=ori
        )
        plotted_any = True

    if not plotted_any:
        raise RuntimeError("No valid data found for either orientation. Nothing to plot.")

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("TBC (MW m$^{-2}$ K$^{-1}$)")
    # ax.set_title("TBC vs Temperature at Two Orientations")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpng, bbox_inches="tight")
    plt.savefig(f"{outpng}.pdf", bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    rows_by_orientation = {}

    for ori, temp_map in PATHS.items():
        rows = collect_orientation_data(ori, temp_map)
        rows_by_orientation[ori] = rows

        print(f"\nOrientation {ori}:")
        if not rows:
            print("  No valid data found.")
        else:
            for T, G, Gerr, src in rows:
                print(f"  T={T:>5.1f} K   G={G:.6g} ± {Gerr:.6g} MW/m^2K   file={src}")

    save_txt(rows_by_orientation, OUT_TXT)
    plot_tbc_vs_temperature(rows_by_orientation, OUT_PNG)

    print(f"\nSaved: {OUT_TXT}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()