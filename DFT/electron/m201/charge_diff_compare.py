#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import Chgcar
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif"]
plt.rcParams.update({"mathtext.default": "regular"})
mpl.rcParams["font.size"] = 20

# ============================================================
# User knobs
# ============================================================
CHGDIFF_100  = "<PATH_TO_DATA>/qe_jobs_pre/pre/interface/100/Bei-106/Si-O/electron/CHGDIFF.vasp"
CHGDIFF_M201 = "<PATH_TO_DATA>/qe_jobs_pre/pre/interface/201/106/electron/CHGDIFF.vasp"

LABEL_100  = r"Ga$_2$O$_3$(100)/SiC(0001)"
LABEL_M201 = r"Ga$_2$O$_3$($\bar{2}01$)/SiC(0001)"

AXIS = 2   # 0=x, 1=y, 2=z

OUT_DIFF = "charge_density_difference_compare.png"
OUT_INT  = "integrated_charge_density_compare.png"

TXT_100_DIFF  = "rho_ave_100.txt"
TXT_100_INT   = "rho_int_100.txt"
TXT_M201_DIFF = "rho_ave_m201.txt"
TXT_M201_INT  = "rho_int_m201.txt"

FIG_DPI = 300
# ============================================================


def get_grid_axis_length(chg, axis):
    if axis == 0:
        return chg.structure.lattice.a
    elif axis == 1:
        return chg.structure.lattice.b
    elif axis == 2:
        return chg.structure.lattice.c
    raise ValueError("AXIS must be 0, 1, or 2")


def load_profile(chgdiff_path, axis):
    print(f"Loading {chgdiff_path}")
    chg = Chgcar.from_file(chgdiff_path)

    rho_avg = chg.get_average_along_axis(axis)

    L = get_grid_axis_length(chg, axis)
    ngrid = len(rho_avg)
    coord = np.linspace(0.0, L, ngrid, endpoint=False)
    dcoord = L / ngrid

    # integral along chosen axis only, unit ~ e/Å^2 if rho_avg is e/Å^3
    rho_int = np.cumsum(rho_avg) * dcoord

    return coord, rho_avg, rho_int


def save_txt(fname, x, y, header):
    np.savetxt(fname, np.column_stack([x, y]), header=header)
    print(f"Saved {fname}")


def main():
    coord_100, rho_avg_100, rho_int_100 = load_profile(CHGDIFF_100, AXIS)
    coord_m201, rho_avg_m201, rho_int_m201 = load_profile(CHGDIFF_M201, AXIS)

    save_txt(TXT_100_DIFF, coord_100, rho_avg_100, "coord_A   rho_avg_e_per_A3")
    save_txt(TXT_100_INT,  coord_100, rho_int_100, "coord_A   integrated_rho_e_per_A2")
    save_txt(TXT_M201_DIFF, coord_m201, rho_avg_m201, "coord_A   rho_avg_e_per_A3")
    save_txt(TXT_M201_INT,  coord_m201, rho_int_m201, "coord_A   integrated_rho_e_per_A2")

    axis_label = ["x", "y", "z"][AXIS]

    # planar-averaged charge density difference
    plt.figure(figsize=(8, 6))
    plt.plot(coord_100,  rho_avg_100,  lw=2.0, label=LABEL_100)
    plt.plot(coord_m201, rho_avg_m201, lw=2.0, label=LABEL_M201)
    plt.legend(loc='upper left', fontsize=18)
    plt.axhline(0.0, color="k", lw=0.8, alpha=0.6)
    plt.xlabel(f"{axis_label} (Å)")
    plt.ylabel(r"$\Delta \rho$ (e/Å$^3$)")
    # plt.title("Planar-Averaged Charge Density Difference")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIFF, dpi=FIG_DPI, bbox_inches="tight")
    plt.savefig(f"{OUT_DIFF}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIFF}")

    # integrated charge density difference
    plt.figure(figsize=(8, 6))
    plt.plot(coord_100,  rho_int_100,  lw=2.0, label=LABEL_100)
    plt.plot(coord_m201, rho_int_m201, lw=2.0, label=LABEL_M201)
    plt.legend(loc='upper left', fontsize=18)
    plt.axhline(0.0, color="k", lw=0.8, alpha=0.6)
    plt.xlabel(f"{axis_label} (Å)")
    plt.ylabel(r"$\int \Delta \rho \, d%s$ (e/Å$^2$)" % axis_label)
    plt.title("Integrated Charge Density Difference")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_INT, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_INT}")

    print("Done.")


if __name__ == "__main__":
    main()