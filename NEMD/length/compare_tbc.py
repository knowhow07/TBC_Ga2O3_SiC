#!/usr/bin/env python3
import re
import glob
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================
DIR_100 = "100"
DIR_201 = "m201"

OUT_PNG = "TBC_compare_100_vs_201.png"
OUT_TXT = "TBC_compare_100_vs_201.txt"

import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif"]
plt.rcParams.update({'mathtext.default': 'regular'})

FONT = 20


# =========================
# HELPERS
# =========================
def newest_log(folder):
    files = sorted(glob.glob(os.path.join(folder, "TBC_thickness_analysis_*.log")))
    if not files:
        raise FileNotFoundError(f"No TBC_thickness_analysis_*.log found in {folder}")
    newest = max(files, key=os.path.getmtime)
    return newest


def parse_tbc_log(logfile):
    """
    Parse lines like:
    14: Lx=316.754 Å, 1/Lx=..., G=610.50±39.78 MW/m²K, Rk=...
    18/fix: Lx=404.748 Å, 1/Lx=..., G=590.00±32.83 MW/m²K, Rk=...
    """
    pattern = re.compile(
        r"""^\s*
        (?P<label>[^:]+):\s*
        Lx=(?P<Lx>[0-9.]+)\s*Å,\s*
        1/Lx=(?P<invLx>[0-9.eE+\-]+)\s*1/Å,\s*
        G=(?P<G>[0-9.]+)±(?P<Gerr>[0-9.]+)\s*MW/m²K
        """,
        re.VERBOSE,
    )

    data = []
    with open(logfile, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                data.append({
                    "label": m.group("label").strip(),
                    "Lx": float(m.group("Lx")),
                    "invLx": float(m.group("invLx")),
                    "G": float(m.group("G")),
                    "Gerr": float(m.group("Gerr")),
                })

    if not data:
        raise RuntimeError(f"No valid TBC entries parsed from {logfile}")

    data = sorted(data, key=lambda d: d["Lx"])
    return data


def save_combined_txt(data100, data201, file100, file201, outfile):
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("# TBC comparison: orientation 100 vs 201\n")
        f.write(f"# newest_100_log = {file100}\n")
        f.write(f"# newest_201_log = {file201}\n")
        f.write("#\n")
        f.write("# orientation  label  Lx_A  invLx_1_per_A  G_MW_m^-2_K^-1  Gerr_MW_m^-2_K^-1\n")
        for d in data100:
            f.write(f"100  {d['label']}  {d['Lx']:.6f}  {d['invLx']:.9e}  {d['G']:.6f}  {d['Gerr']:.6f}\n")
        for d in data201:
            f.write(f"201  {d['label']}  {d['Lx']:.6f}  {d['invLx']:.9e}  {d['G']:.6f}  {d['Gerr']:.6f}\n")


def make_plot(data100, data201, outpng):
    lx100 = np.array([d["Lx"] for d in data100])
    g100 = np.array([d["G"] for d in data100])
    e100 = np.array([d["Gerr"] for d in data100])

    lx201 = np.array([d["Lx"] for d in data201])
    g201 = np.array([d["G"] for d in data201])
    e201 = np.array([d["Gerr"] for d in data201])

    plt.rcParams.update({
        "font.size": FONT,
        "axes.labelsize": FONT,
        "axes.titlesize": FONT,
        "xtick.labelsize": FONT,
        "ytick.labelsize": FONT,
        "legend.fontsize": FONT - 1,
    })

    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

    ax.errorbar(
        lx100, g100, yerr=e100,
        fmt='o-', capsize=4, linewidth=2, markersize=6,
        label=r'$Ga_2O_3(100)/SiC(0001)$'
    )
    ax.errorbar(
        lx201, g201, yerr=e201,
        fmt='s-', capsize=4, linewidth=2, markersize=6,
        label=r'$Ga_2O_3(\bar{2}01)/SiC(0001)$'
    )

    ax.set_xlabel("Length Lx (Å)")
    ax.set_ylabel("TBC (MW m$^{-2}$ K$^{-1}$)")
    # ax.set_title("TBC vs Length at Two Orientations")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    fig.savefig(outpng, bbox_inches="tight")
    fig.savefig(f"{outpng}.pdf", bbox_inches="tight")
    plt.close(fig)


# =========================
# MAIN
# =========================
def main():
    log100 = newest_log(DIR_100)
    log201 = newest_log(DIR_201)

    data100 = parse_tbc_log(log100)
    data201 = parse_tbc_log(log201)

    save_combined_txt(data100, data201, log100, log201, OUT_TXT)
    make_plot(data100, data201, OUT_PNG)

    print("Newest log for 100:", log100)
    print("Newest log for 201:", log201)
    print("Saved:", OUT_TXT)
    print("Saved:", OUT_PNG)


if __name__ == "__main__":
    main()