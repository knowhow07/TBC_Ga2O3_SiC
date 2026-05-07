#!/usr/bin/env python3
import os
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# User settings
# ============================================================
GROUP_ID = 4   # use 4 for interface; can change to 3 or 5

TEMP_PATHS = {
    300: "./300k/results",
    600: "./600k-fix/results",
    900: "./900k/results",
}

# Threshold from PDOS-overlap discussion
# Frequencies <= cutoff are treated as elastic-like overlap region
# Frequencies > cutoff are treated as inelastic-sensitive region
FREQ_CUT_THz = 18.0

# If True, clip negative G_like to zero before integration
# Recommended for "contribution" analysis to suppress noisy negative spikes
CLIP_NEGATIVE = True

OUT_DIR = "results_elastic_inelastic"
FIG_DPI = 300


# ============================================================
# Helpers
# ============================================================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_spectral_data(file_path):
    """
    Reads previous output:
      group_<gid>_spectral_data.txt

    Expected columns:
      omega_THz
      Jin_native
      Jout_native
      Jsum_native
      G_like_native
      G_like_per_A2_native
      G_like_per_A2_smoothed
      cumulative_integral_native
    """
    arr = np.loadtxt(file_path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 7:
        raise RuntimeError(f"Unexpected format in {file_path}; got {arr.shape[1]} columns")

    return {
        "omega": arr[:, 0],
        "Jin": arr[:, 1],
        "Jout": arr[:, 2],
        "Jsum": arr[:, 3],
        "G_like": arr[:, 4],
        "G_like_A": arr[:, 5],
        "G_like_A_smooth": arr[:, 6],
    }


def integrate_window(x, y, mask):
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(np.trapz(y[mask], x[mask]))


def nearest_common_grid(ref_x, x, y):
    """
    Interpolate y(x) onto ref_x so all temperatures use same grid if needed.
    """
    return np.interp(ref_x, x, y)


# ============================================================
# Main analysis
# ============================================================
def main():
    out_dir = Path(OUT_DIR)
    ensure_dir(out_dir)

    loaded = {}
    missing_any = False

    # ---- load all temperatures ----
    for T, folder in TEMP_PATHS.items():
        file_path = Path(folder) / f"group_{GROUP_ID}_spectral_data.txt"
        if not file_path.is_file():
            warnings.warn(f"Missing file for {T} K: {file_path}")
            missing_any = True
            continue

        d = load_spectral_data(file_path)
        loaded[T] = d
        print(f"[OK] Loaded {T} K: {file_path}")

    if len(loaded) == 0:
        raise RuntimeError("No valid spectral data files found.")

    # ---- choose common frequency grid from first available temperature ----
    Ts_sorted = sorted(loaded.keys())
    ref_T = Ts_sorted[0]
    omega_ref = loaded[ref_T]["omega"]

    results = []

    for T in Ts_sorted:
        d = loaded[T]

        # use smoothed G_like per area for integration
        y = nearest_common_grid(omega_ref, d["omega"], d["G_like_A_smooth"])

        if CLIP_NEGATIVE:
            y_use = np.maximum(y, 0.0)
        else:
            y_use = y.copy()

        mask_el = omega_ref <= FREQ_CUT_THz
        mask_inel = omega_ref > FREQ_CUT_THz

        I_total = float(np.trapz(y_use, omega_ref))
        I_el = integrate_window(omega_ref, y_use, mask_el)
        I_inel = integrate_window(omega_ref, y_use, mask_inel)

        frac_el = I_el / I_total if I_total > 0 else np.nan
        frac_inel = I_inel / I_total if I_total > 0 else np.nan

        results.append({
            "T": T,
            "omega": omega_ref,
            "G_plot": y,
            "G_use": y_use,
            "I_total": I_total,
            "I_el": I_el,
            "I_inel": I_inel,
            "frac_el": frac_el,
            "frac_inel": frac_inel,
        })

    # ============================================================
    # Save txt summary
    # ============================================================
    summary_txt = out_dir / f"group_{GROUP_ID}_elastic_inelastic_summary.txt"
    with open(summary_txt, "w") as f:
        f.write("# Elastic / inelastic-sensitive integration summary\n")
        f.write(f"# group_id = {GROUP_ID}\n")
        f.write(f"# freq_cut_THz = {FREQ_CUT_THz:.6f}\n")
        f.write(f"# clip_negative = {CLIP_NEGATIVE}\n")
        f.write("#\n")
        f.write("# Definition:\n")
        f.write("#   elastic_like         : integral of G_like_smoothed over omega <= cutoff\n")
        f.write("#   inelastic_sensitive  : integral of G_like_smoothed over omega > cutoff\n")
        f.write("#\n")
        f.write("# T_K  I_total  I_elastic_like  I_inelastic_sensitive  frac_elastic_like  frac_inelastic_sensitive\n")
        for r in results:
            f.write(
                f"{r['T']:6.1f}  "
                f"{r['I_total']:.10e}  "
                f"{r['I_el']:.10e}  "
                f"{r['I_inel']:.10e}  "
                f"{r['frac_el']:.10e}  "
                f"{r['frac_inel']:.10e}\n"
            )

    # ============================================================
    # Figure 1: spectral curves with threshold region
    # ============================================================
    plt.figure(figsize=(8, 6), dpi=FIG_DPI)

    ymax = 0.0
    for r in results:
        ymax = max(ymax, np.nanmax(r["G_plot"]))
        plt.plot(r["omega"], r["G_plot"], label=f"{int(r['T'])} K", linewidth=2)

    plt.axvspan(0.0, FREQ_CUT_THz, alpha=0.12, label="elastic-like window")
    plt.axvline(FREQ_CUT_THz, linestyle="--", linewidth=1.5)

    plt.xlabel("Frequency (THz)")
    plt.ylabel("G_like(ω) / Å$^2$ (smoothed)")
    plt.title(f"Group {GROUP_ID}: elastic-like vs inelastic-sensitive window")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"group_{GROUP_ID}_spectral_elastic_inelastic_window.png")
    plt.close()

    # ============================================================
    # Figure 2: stacked bar of integrated parts
    # ============================================================
    temps = [r["T"] for r in results]
    I_el = [r["I_el"] for r in results]
    I_inel = [r["I_inel"] for r in results]

    plt.figure(figsize=(7, 6), dpi=FIG_DPI)
    plt.bar(temps, I_el, width=80, label="elastic-like")
    plt.bar(temps, I_inel, width=80, bottom=I_el, label="inelastic-sensitive")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Integrated G_like (native units)")
    plt.title(f"Group {GROUP_ID}: integrated elastic-like / inelastic-sensitive parts")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / f"group_{GROUP_ID}_elastic_inelastic_stacked_bar.png")
    plt.close()

    # ============================================================
    # Figure 3: fractions vs temperature (dual y-axis)
    # ============================================================
    frac_el = [r["frac_el"] for r in results]
    frac_inel = [r["frac_inel"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 6), dpi=FIG_DPI)
    ax2 = ax1.twinx()

    l1 = ax1.plot(
        temps, frac_el, "o-", linewidth=2, markersize=7,
        color="tab:blue",
        label="elastic-like fraction"
    )
    l2 = ax2.plot(
        temps, frac_inel, "s-", linewidth=2, markersize=7,
        color="tab:orange",
        label="inelastic-sensitive fraction"
    )

    ax1.tick_params(axis="y", colors="tab:blue")
    ax2.tick_params(axis="y", colors="tab:orange")
    ax1.yaxis.label.set_color("tab:blue")
    ax2.yaxis.label.set_color("tab:orange")

    ax1.set_xlabel("Temperature (K)")
    ax1.set_ylabel("Elastic-like fraction")
    ax2.set_ylabel("Inelastic-sensitive fraction")
    ax1.set_title(f"Group {GROUP_ID}: elastic-like / inelastic-sensitive fractions")

    ax1.grid(True, linestyle="--", alpha=0.4)

    # tighten the y-ranges so the trends are easier to see
    fe_min, fe_max = min(frac_el), max(frac_el)
    fi_min, fi_max = min(frac_inel), max(frac_inel)

    pad1 = max(0.002, 0.15 * (fe_max - fe_min if fe_max > fe_min else 0.01))
    pad2 = max(0.002, 0.15 * (fi_max - fi_min if fi_max > fi_min else 0.01))

    ax1.set_ylim(fe_min - pad1, fe_max + pad1)
    ax2.set_ylim(fi_min - pad2, fi_max + pad2)

    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="center right")

    fig.tight_layout()
    fig.savefig(out_dir / f"group_{GROUP_ID}_elastic_inelastic_fraction_vs_T.png")
    plt.close(fig)

    # ============================================================
    # Optional detailed spectral txt after clipping/interpolation
    # ============================================================
    for r in results:
        out_spec = out_dir / f"group_{GROUP_ID}_{int(r['T'])}K_elastic_inelastic_spectrum.txt"
        mask_el = r["omega"] <= FREQ_CUT_THz
        mask_inel = r["omega"] > FREQ_CUT_THz
        region = np.where(mask_el, 0, 1)  # 0 elastic-like, 1 inelastic-sensitive

        header = (
            "omega_THz  G_like_smoothed_interp  G_like_used_for_integration  region_flag\n"
            "region_flag: 0=elastic_like (omega<=cutoff), 1=inelastic_sensitive (omega>cutoff)"
        )
        np.savetxt(
            out_spec,
            np.column_stack([r["omega"], r["G_plot"], r["G_use"], region]),
            header=header
        )

    print("\nSaved:")
    print(f"  {summary_txt}")
    print(f"  {out_dir / f'group_{GROUP_ID}_spectral_elastic_inelastic_window.png'}")
    print(f"  {out_dir / f'group_{GROUP_ID}_elastic_inelastic_stacked_bar.png'}")
    print(f"  {out_dir / f'group_{GROUP_ID}_elastic_inelastic_fraction_vs_T.png'}")
    for r in results:
        print(f"  {out_dir / f'group_{GROUP_ID}_{int(r['T'])}K_elastic_inelastic_spectrum.txt'}")

    if missing_any:
        print("\nWarning: some temperature files were missing; plotted available results only.")


if __name__ == "__main__":
    main()