#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================
KAPPA_FILE = "<PATH_TO_DATA>/qe_jobs/nep_training/NEMD/0217/post/MFP/SiC/hnemd/0001/kappa_spectrum_avg_with_error.txt"
G_FILE     = "<PATH_TO_DATA>/qe_jobs/nep_training/NEMD/0217/post/MFP/SiC/shortbar/bulk_sic_shortbar_Gomega_avg.txt"

OUT_PREFIX = "mfp_from_kappa_over_G"

# --- geometry source for SHC-region length ---
MODEL_FILE = "<PATH_TO_DATA>/qe_jobs/nep_training/NEMD/0217/post/MFP/SiC/shortbar/job_01/model.xyz"
SHC_GROUP_ID = 3

# 1 eV / (ps * Å^2) = 1.602176634e13 W / m^2
EV_PER_PS_A2_TO_W_M2 = 1.602176634e13

# columns (0-based, after skipping # comments)
KAPPA_FREQ_COL   = 0
KAPPA_VAL_COL    = 1
KAPPA_ERR_COL    = 3   # use stderr

G_FREQ_COL       = 0
G_VAL_COL        = 3
G_ERR_COL        = 4   # use stderr

plt.rcParams["font.size"] = 14



# If G(ω) is already in W m^-2 K^-1 THz^-1, keep = 1.0
# If not, set the correct conversion factor here.
# Current G file header says "native units", so this likely needs calibration.
# G_CONV_TO_SI = 1.0
# L_SHC_A = 28.324 - 20.141   # = 8.183 Å
# G_CONV_TO_SI = 2.0 * 1.602176634e13 / L_SHC_A

# plotting / filtering
FREQ_MIN_THz = 0.0
FREQ_MAX_THz = None
SMOOTH_POINTS = 9   # odd integer, 1 = off
MIN_G_POSITIVE = 1e-30

# output MFP unit
OUTPUT_UNIT = "nm"   # "m", "um", "nm", "A"

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
                # species x y z group
                x = float(parts[1])
                g = int(parts[4])
            elif len(parts) >= 6:
                # id species x y z group
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

xmin_shc, xmax_shc, L_SHC_A = read_group_x_length(MODEL_FILE, SHC_GROUP_ID)
print(f"SHC group {SHC_GROUP_ID} x-range: {xmin_shc:.6f} -> {xmax_shc:.6f} Å")
print(f"L_SHC_A = {L_SHC_A:.6f} Å")

G_CONV_TO_SI = 2.0 * EV_PER_PS_A2_TO_W_M2 / L_SHC_A
print(f"G_CONV_TO_SI = {G_CONV_TO_SI:.6e}")

# =========================
# LOAD DATA
# =========================
fk, kappa, kappa_err = load_two_cols_with_err(
    KAPPA_FILE, KAPPA_FREQ_COL, KAPPA_VAL_COL, KAPPA_ERR_COL
)

fg, Graw, Graw_err = load_two_cols_with_err(
    G_FILE, G_FREQ_COL, G_VAL_COL, G_ERR_COL
)


G = Graw * G_CONV_TO_SI
G_err = Graw_err * G_CONV_TO_SI

xmin_shc, xmax_shc, L_SHC_A = read_group_x_length(MODEL_FILE, SHC_GROUP_ID)
print(f"SHC group {SHC_GROUP_ID} x-range: {xmin_shc:.6f} -> {xmax_shc:.6f} Å")
print(f"L_SHC_A = {L_SHC_A:.6f} Å")

G_CONV_TO_SI = 2.0 * EV_PER_PS_A2_TO_W_M2 / L_SHC_A
print(f"G_CONV_TO_SI = {G_CONV_TO_SI:.6e}")

# =========================
# MATCH FREQUENCY GRID
# =========================
fmin = max(np.min(fk), np.min(fg), FREQ_MIN_THz)
fmax = min(np.max(fk), np.max(fg))
if FREQ_MAX_THz is not None:
    fmax = min(fmax, FREQ_MAX_THz)

mask_k = (fk >= fmin) & (fk <= fmax)
fk_use = fk[mask_k]
kappa_use = kappa[mask_k]
kappa_err_use = kappa_err[mask_k]

# interpolate G onto kappa frequency grid
G_use = np.interp(fk_use, fg, G)
G_err_use = np.interp(fk_use, fg, G_err)



# =========================
# COMPUTE MFP
# =========================
# For physical MFP, G must be in W m^-2 K^-1 THz^-1
valid = (
    np.isfinite(kappa_use) &
    np.isfinite(G_use) &
    (G_use > 0.0) &
    (G_use > 0.0 * G_err_use)   # or 2x, 5x depending on how strict you want
)

# valid = (
#     np.isfinite(kappa_use) &
#     np.isfinite(G_use) &
#     (np.abs(G_use) > MIN_G_POSITIVE)
# )

f_valid = fk_use[valid]
k_valid = kappa_use[valid]
kerr_valid = kappa_err_use[valid]
G_valid = G_use[valid]
Gerr_valid = G_err_use[valid]

lambda_m = k_valid / G_valid

# propagated relative uncertainty
rel_k = np.zeros_like(kerr_valid)
rel_G = np.zeros_like(Gerr_valid)

nz_k = np.abs(k_valid) > 0
nz_G = np.abs(G_valid) > 0
rel_k[nz_k] = kerr_valid[nz_k] / np.abs(k_valid[nz_k])
rel_G[nz_G] = Gerr_valid[nz_G] / np.abs(G_valid[nz_G])

lambda_err_m = lambda_m * np.sqrt(rel_k**2 + rel_G**2)

# convert unit
scale, unit_label = unit_scale_and_label(OUTPUT_UNIT)
lambda_plot = lambda_m * scale
lambda_err_plot = lambda_err_m * scale

# optional smoothing for plotting only
lambda_plot = smooth_1d(lambda_plot, SMOOTH_POINTS)
lambda_err_plot = smooth_1d(lambda_err_plot, SMOOTH_POINTS)
# =========================
# SAVE DATA
# =========================
out = np.column_stack([
    f_valid,
    k_valid,
    kerr_valid,
    G_valid,
    Gerr_valid,
    lambda_m,
    lambda_err_m
])

np.savetxt(
    f"{OUT_PREFIX}.txt",
    out,
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
plt.plot(f_valid, lambda_plot, lw=2, label=r"$\lambda(\omega)=\kappa(\omega)/G(\omega)$")
plt.fill_between(
    f_valid,
    np.maximum(lambda_plot - lambda_err_plot, 0.0),
    lambda_plot + lambda_err_plot,
    alpha=0.25,
    label="± propagated stderr"
)

plt.xlabel(r"Frequency $\omega/2\pi$ (THz)")
plt.ylabel(f"MFP ({unit_label})")
plt.yscale('log')
plt.title("MFP vs frequency")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}.png", dpi=300)

print(f"Saved: {OUT_PREFIX}.txt")
print(f"Saved: {OUT_PREFIX}.png")
print("\nNOTE:")
print("  lambda = kappa / G is only a physical MFP if G_FILE has been converted")
print("  to SI units of W m^-2 K^-1 THz^-1.")
print(f"L_SHC_A read from model = {L_SHC_A:.6f} Å")
print(f"  Current G_CONV_TO_SI = {G_CONV_TO_SI}")