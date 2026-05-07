#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.io.vasp.outputs import Chgcar
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif"]

plt.rcParams.update({'mathtext.default': 'regular'})
mpl.rcParams["font.size"] = 16

# ============================================================
# User knobs
# ============================================================
# CHGCAR_INTERFACE = "inter_dipole/CHGCAR_sum"
# CHGCAR_SIC       = "SiCH_dipole/CHGCAR_sum"
# CHGCAR_GAO       = "GaO_dipole/CHGCAR_sum"
CHGCAR_INTERFACE = "inter_dipole/CHGCAR"
CHGCAR_SIC       = "SiCH_dipole_4H/CHGCAR"
CHGCAR_GAO       = "GaO_dipole/CHGCAR"

OUT_CHGDIFF      = "CHGDIFF.vasp"

AXIS             = 2   # 0=x, 1=y, 2=z ; interface normal usually z
MAKE_DIFF        = False

PLOT_DIFF        = "charge_density_difference_new.png"
PLOT_INT         = "integrated_charge_density_new.png"

TXT_DIFF         = "rho_ave_z.txt"
TXT_INT          = "rho_integrated_z.txt"

FIG_DPI          = 300
# ============================================================


def check_same_grid(chg_if, chg1, chg2):
    s_if = chg_if.data["total"].shape
    s_1  = chg1.data["total"].shape
    s_2  = chg2.data["total"].shape
    print("Grid shapes:")
    print("interface :", s_if)
    print("SiC       :", s_1)
    print("Ga2O3     :", s_2)
    if not (s_if == s_1 == s_2):
        raise ValueError("CHGCAR grids do not match. Use matched files from the same interface setup.")


def get_grid_axis_length(chg, axis):
    if axis == 0:
        return chg.structure.lattice.a
    elif axis == 1:
        return chg.structure.lattice.b
    elif axis == 2:
        return chg.structure.lattice.c
    else:
        raise ValueError("AXIS must be 0, 1, or 2")


def build_chgdiff():
    print("Reading charge densities...")
    chg_if = Chgcar.from_file(CHGCAR_INTERFACE)
    chg1   = Chgcar.from_file(CHGCAR_SIC)
    chg2   = Chgcar.from_file(CHGCAR_GAO)
    print(chg_if.data["total"].shape)
    print(chg1.data["total"].shape)
    print(chg2.data["total"].shape)

    check_same_grid(chg_if, chg1, chg2)

    print("Building charge density difference:")
    print("rho_diff = rho_interface - rho_SiC - rho_Ga2O3")
    chg_diff = chg_if - (chg1 + chg2)

    print(f"Writing {OUT_CHGDIFF}")
    chg_diff.write_file(OUT_CHGDIFF)
    return chg_diff


def load_chgdiff():
    print(f"Loading {OUT_CHGDIFF}")
    return Chgcar.from_file(OUT_CHGDIFF)


def main():
    if MAKE_DIFF:
        chg_diff = build_chgdiff()
    else:
        chg_diff = load_chgdiff()

    # planar average along chosen axis
    rho_avg = chg_diff.get_average_along_axis(AXIS)

    L = get_grid_axis_length(chg_diff, AXIS)
    ngrid = len(rho_avg)
    coord = np.linspace(0.0, L, ngrid, endpoint=False)
    dcoord = L / ngrid

    # integrated profile
    rho_int = np.cumsum(rho_avg) * dcoord

    # save txt
    np.savetxt(
        TXT_DIFF,
        np.column_stack([coord, rho_avg]),
        header="coord_A   rho_avg_e_per_A3"
    )
    np.savetxt(
        TXT_INT,
        np.column_stack([coord, rho_int]),
        header="coord_A   integrated_rho"
    )

    # plot planar-averaged charge density difference
    plt.figure(figsize=(7, 5))
    plt.plot(coord, rho_avg, lw=1.8)
    plt.axhline(0.0, color="k", lw=0.8, alpha=0.6)
    plt.xlabel(["x", "y", "z"][AXIS] + " (Å)")
    plt.ylabel(r"$\int \Delta \rho \, dz$ (e/Å$^2$)")
    plt.title("Planar-Averaged Charge Density Difference")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIFF, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

    # plot integrated charge profile
    plt.figure(figsize=(7, 5))
    plt.plot(coord, rho_int, lw=1.8)
    plt.axhline(0.0, color="k", lw=0.8, alpha=0.6)
    plt.xlabel(["x", "y", "z"][AXIS] + " (Å)")
    plt.ylabel(r"$\int \Delta \rho \, d%s$" % ["x", "y", "z"][AXIS])
    plt.title("Integrated Charge Density Difference")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_INT, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

    print("Done.")
    print(f"Saved: {OUT_CHGDIFF}")
    print(f"Saved: {TXT_DIFF}")
    print(f"Saved: {TXT_INT}")
    print(f"Saved: {PLOT_DIFF}")
    print(f"Saved: {PLOT_INT}")


if __name__ == "__main__":
    main()