
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif"]
mpl.rcParams["font.size"] = 20
plt.rcParams.update({'mathtext.default': 'regular'})

# =========================
# USER SETTINGS
# =========================

# -------- Ga2O3 (-201) --------
KAPPA_FILE_GA = "<PATH_TO_DATA>/qe_jobs/nep_training/NEMD/0217/post/MFP/Ga2O3/201/hnemd/kappa_spectrum_avg_with_error.txt"
G_FILE_GA     = "<PATH_TO_DATA>/qe_jobs/nep_training/NEMD/0217/post/MFP/Ga2O3/201/shortbar/bulk_ga2o3_shortbar_Gomega_avg_native.txt"
MODEL_FILE_GA = "<PATH_TO_DATA>/qe_jobs/nep_training/NEMD/0217/post/MFP/Ga2O3/201/shortbar/job_01/model.xyz"
SHC_GROUP_ID_GA = 3

# -------- SiC (0001) --------
KAPPA_FILE_SIC = "<PATH_TO_DATA>/qe_jobs/nep_training/NEMD/0217/post/MFP/SiC/hnemd/0001/kappa_spectrum_avg_with_error.txt"
G_FILE_SIC     = "<PATH_TO_DATA>/qe_jobs/nep_training/NEMD/0217/post/MFP/SiC/shortbar/bulk_SiC_shortbar_Gomega_avg_native.txt"
MODEL_FILE_SIC = "<PATH_TO_DATA>/qe_jobs/nep_training/NEMD/0217/post/MFP/SiC/shortbar/job_01/model.xyz"
SHC_GROUP_ID_SIC = 3

OUT_PREFIX = "mfp_ga2o3_sic_compare_largefont"

# 1 eV / (ps * Å^2) = 1.602176634e13 W / m^2
EV_PER_PS_A2_TO_W_M2 = 1.602176634e13

# columns (0-based, after skipping # comments)
KAPPA_FREQ_COL = 0
KAPPA_VAL_COL  = 1
KAPPA_ERR_COL  = 3   # stderr

G_FREQ_COL     = 0
G_VAL_COL      = 1
G_ERR_COL      = 2   # stderr

# plotting / filtering
FREQ_MIN_THz = 0.0
FREQ_MAX_THz = None
XMAX_THz     = 15.0   # x-axis cutoff for plot
SMOOTH_POINTS = 1     # odd integer, 1 = off

# output MFP unit
OUTPUT_UNIT = "nm"    # "m", "um", "nm", "A"

# G conversion
# Both files are assumed in native unit:
# eV ps^-1 THz^-1 Å^-2 K^-1
# converted to:
# W m^-2 K^-1 THz^-1
# use model-dependent SHC length
def read_group_x_length(model_xyz: str, group_id: int):
    """
    Read grouped model.xyz and return:
      xmin, xmax, Lx_group = xmax - xmin   (Å)

    Supports:
      species x y z group
      id species x y z group
    """
    with open(model_xyz, "r") as f:
        n = int(f.readline().strip())
        _ = f.readline()
        xs = []

        for _ in range(n):
            parts = f.readline().split()
            if not parts:
                continue

            if len(parts) == 5:
                x = float(parts[1])
                g = int(parts[4])
            elif len(parts) >= 6:
                x = float(parts[2])
                g = int(parts[-1])
            else:
                continue

            if g == group_id:
                xs.append(x)

    if len(xs) == 0:
        raise RuntimeError(f"{model_xyz}: no atoms found for group {group_id}")

    xs = np.asarray(xs, dtype=float)
    xmin = float(xs.min())
    xmax = float(xs.max())
    return xmin, xmax, xmax - xmin
_, _, L_SHC_A_GA  = read_group_x_length(MODEL_FILE_GA, SHC_GROUP_ID_GA)
_, _, L_SHC_A_SIC = read_group_x_length(MODEL_FILE_SIC, SHC_GROUP_ID_SIC)

# if your G files are native SHC output and need length normalization, use these
# otherwise keep the old conversion if already calibrated
# G_CONV_TO_SI_GA  = 2.0 * EV_PER_PS_A2_TO_W_M2 / L_SHC_A_GA
# G_CONV_TO_SI_SIC = 2.0 * EV_PER_PS_A2_TO_W_M2 / L_SHC_A_SIC

G_CONV_TO_SI_GA  = EV_PER_PS_A2_TO_W_M2
G_CONV_TO_SI_SIC = EV_PER_PS_A2_TO_W_M2

print(f"L_SHC_A_GA  = {L_SHC_A_GA:.6f} Å")
print(f"L_SHC_A_SIC = {L_SHC_A_SIC:.6f} Å")
print(f"G_CONV_TO_SI_GA  = {G_CONV_TO_SI_GA:.6e}")
print(f"G_CONV_TO_SI_SIC = {G_CONV_TO_SI_SIC:.6e}")

# =========================
# HELPERS
# =========================



def smooth_1d(y, points):
    if points <= 1:
        return y.copy()
    if points % 2 == 0:
        points += 1
    pad = points // 2
    k = np.ones(points, dtype=float) / points
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, k, mode="valid")

def unit_scale_and_label(unit):
    if unit == "m":
        return 1.0, "m"
    elif unit == "um":
        return 1e6, r"$\mu$m"
    elif unit == "nm":
        return 1e9, "nm"
    elif unit == "A":
        return 1e10, r"$\AA$"
    else:
        raise ValueError("OUTPUT_UNIT must be one of: m, um, nm, A")

def load_two_cols_with_err(fname, fcol, vcol, ecol):
    data = np.loadtxt(fname, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    f = data[:, fcol]
    v = data[:, vcol]
    e = data[:, ecol]
    return f, v, e

def compute_mfp_curve(kappa_file, g_file, g_conv_to_si,
                      kappa_freq_col=0, kappa_val_col=1, kappa_err_col=3,
                      g_freq_col=0, g_val_col=1, g_err_col=2,
                      freq_min=0.0, freq_max=None,
                      smooth_points=1, output_unit="nm"):
    fk, kappa, kappa_err = load_two_cols_with_err(
        kappa_file, kappa_freq_col, kappa_val_col, kappa_err_col
    )
    fg, Graw, Graw_err = load_two_cols_with_err(
        g_file, g_freq_col, g_val_col, g_err_col
    )

    G = Graw * g_conv_to_si
    G_err = Graw_err * g_conv_to_si

    fmin = max(np.min(fk), np.min(fg), freq_min)
    fmax = min(np.max(fk), np.max(fg))
    if freq_max is not None:
        fmax = min(fmax, freq_max)

    mask_k = (fk >= fmin) & (fk <= fmax)
    fk_use = fk[mask_k]
    kappa_use = kappa[mask_k]
    kappa_err_use = kappa_err[mask_k]

    G_use = np.interp(fk_use, fg, G)
    G_err_use = np.interp(fk_use, fg, G_err)

    valid = (
        np.isfinite(kappa_use) &
        np.isfinite(G_use) &
        (G_use > 0.0)
    )

    f_valid = fk_use[valid]
    k_valid = kappa_use[valid]
    kerr_valid = kappa_err_use[valid]
    G_valid = G_use[valid]
    Gerr_valid = G_err_use[valid]

    lambda_m = k_valid / G_valid

    rel_k = np.zeros_like(kerr_valid)
    rel_G = np.zeros_like(Gerr_valid)

    nz_k = np.abs(k_valid) > 0
    nz_G = np.abs(G_valid) > 0
    rel_k[nz_k] = kerr_valid[nz_k] / np.abs(k_valid[nz_k])
    rel_G[nz_G] = Gerr_valid[nz_G] / np.abs(G_valid[nz_G])

    lambda_err_m = lambda_m * np.sqrt(rel_k**2 + rel_G**2)

    scale, unit_label = unit_scale_and_label(output_unit)
    lambda_plot = lambda_m * scale
    lambda_err_plot = lambda_err_m * scale

    lambda_plot = smooth_1d(lambda_plot, smooth_points)
    lambda_err_plot = smooth_1d(lambda_err_plot, smooth_points)

    out = np.column_stack([
        f_valid,
        k_valid,
        kerr_valid,
        G_valid,
        Gerr_valid,
        lambda_m,
        lambda_err_m
    ])

    return {
        "f": f_valid,
        "kappa": k_valid,
        "kappa_err": kerr_valid,
        "G": G_valid,
        "G_err": Gerr_valid,
        "lambda_m": lambda_m,
        "lambda_err_m": lambda_err_m,
        "lambda_plot": lambda_plot,
        "lambda_err_plot": lambda_err_plot,
        "unit_label": unit_label,
        "out_table": out,
    }

# =========================
# COMPUTE BOTH CURVES
# =========================
ga = compute_mfp_curve(
    KAPPA_FILE_GA, G_FILE_GA, G_CONV_TO_SI_GA,
    kappa_freq_col=KAPPA_FREQ_COL,
    kappa_val_col=KAPPA_VAL_COL,
    kappa_err_col=KAPPA_ERR_COL,
    g_freq_col=G_FREQ_COL,
    g_val_col=G_VAL_COL,
    g_err_col=G_ERR_COL,
    freq_min=FREQ_MIN_THz,
    freq_max=FREQ_MAX_THz,
    smooth_points=SMOOTH_POINTS,
    output_unit=OUTPUT_UNIT
)

sic = compute_mfp_curve(
    KAPPA_FILE_SIC, G_FILE_SIC, G_CONV_TO_SI_SIC,
    kappa_freq_col=KAPPA_FREQ_COL,
    kappa_val_col=KAPPA_VAL_COL,
    kappa_err_col=KAPPA_ERR_COL,
    g_freq_col=G_FREQ_COL,
    g_val_col=G_VAL_COL,
    g_err_col=G_ERR_COL,
    freq_min=FREQ_MIN_THz,
    freq_max=FREQ_MAX_THz,
    smooth_points=SMOOTH_POINTS,
    output_unit=OUTPUT_UNIT
)

# =========================
# SAVE DATA
# =========================
np.savetxt(
    f"{OUT_PREFIX}_ga2o3.txt",
    ga["out_table"],
    header=(
        "freq_THz  "
        "kappa_Wm-1K-1THz-1  kappa_err  "
        "G_Wm-2K-1THz-1  G_err  "
        "lambda_m  lambda_err_m"
    ),
    fmt="%.10e"
)

np.savetxt(
    f"{OUT_PREFIX}_sic.txt",
    sic["out_table"],
    header=(
        "freq_THz  "
        "kappa_Wm-1K-1THz-1  kappa_err  "
        "G_Wm-2K-1THz-1  G_err  "
        "lambda_m  lambda_err_m"
    ),
    fmt="%.10e"
)

# =========================
# PLOT
# =========================
plt.figure(figsize=(8, 6))

plt.plot(
    sic["f"], sic["lambda_plot"],
    lw=2,
    color="tab:green",
    label="SiC(0001)"
)

plt.plot(
    ga["f"], ga["lambda_plot"],
    lw=2,
    color="tab:orange",
    label=r"$Ga_2O_3$($\bar{2}$01)"
)



# optional error band
# plt.fill_between(
#     ga["f"],
#     np.maximum(ga["lambda_plot"] - ga["lambda_err_plot"], 0.0),
#     ga["lambda_plot"] + ga["lambda_err_plot"],
#     alpha=0.20
# )
# plt.fill_between(
#     sic["f"],
#     np.maximum(sic["lambda_plot"] - sic["lambda_err_plot"], 0.0),
#     sic["lambda_plot"] + sic["lambda_err_plot"],
#     alpha=0.20
# )

plt.xlabel(r"Frequency $\omega/2\pi$ (THz)")
plt.ylabel(f"MFP ({ga['unit_label']})")
plt.xlim(0.0, XMAX_THz)
plt.yscale("log")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}.png", dpi=600)
plt.savefig(f"{OUT_PREFIX}.pdf", bbox_inches="tight")

print(f"Saved: {OUT_PREFIX}_ga2o3.txt")
print(f"Saved: {OUT_PREFIX}_sic.txt")
print(f"Saved: {OUT_PREFIX}.png")
print(f"Saved: {OUT_PREFIX}.pdf")
print(f"G_CONV_TO_SI_GA  = {G_CONV_TO_SI_GA:.6e}")
print(f"G_CONV_TO_SI_SIC = {G_CONV_TO_SI_SIC:.6e}")