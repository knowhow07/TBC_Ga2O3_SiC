#!/usr/bin/env python3
"""
Plot experimental thermal conductivity vs NEP-EMD results for beta-Ga2O3.

Updates vs previous version:
1) Larger fonts controlled by a single variable FONT_SCALE
2) NEP points use error bars (stderr) parsed from the LATEST postprocess_*.log
   in each temperature folder (e.g., 300k/600k/900k).
"""

import os
import re
import glob
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif"]
plt.rcParams.update({'mathtext.default': 'regular'})

mpl.rcParams["axes.labelsize"] = 13
mpl.rcParams["axes.titlesize"] = 13
mpl.rcParams["xtick.labelsize"] = 11
mpl.rcParams["ytick.labelsize"] = 11
mpl.rcParams["legend.fontsize"] = 10

# ============================================================
# User variables (edit here)
# ============================================================

BASE_DIR = "."                       # contains experimental.csv + temp folders
EXPERIMENT_CSV = "experiment.csv"  # as you showed

# NEP folders:
NEP_FOLDERS = ["300k", "450k", "600k", "750k", "900k"]    # explicit list
NEP_FOLDER_GLOB = None                    # if not None, overrides NEP_FOLDERS (e.g. "*k")
NEP_FOLDER_REGEX = r"^\d+k$"              # used when NEP_FOLDER_GLOB is not None

# Latest log file pattern inside each NEP folder
NEP_LOG_GLOB = "postprocess_*.log"        # e.g. postprocess_20260213_195459.log

# Temperature range to plot
T_MIN = 280.0
T_MAX = 920.0

# Directions to plot
# --- REVISE: plot channels you want to show ---
# Experiment only has: in-plane (kr) and cross-plane (kz)
# NEP has: kx, ky, kz  (show all three)

EXP_DIRECTIONS = ["in-plane (kr)", "cross-plane (kz)"]
NEP_DIRECTIONS = ["kx_tot", "ky_tot", "kz_tot"]
# --- REVISE: shorten experiment legend labels ---
EXP_LABEL_SHORT = {
    "in-plane (kr)": "4H(UID-SI) Exp in-plane",
    "cross-plane (kz)": "4H(UID-SI) Exp cross-plane",
}
EXP_COLOR = {
    "in-plane (kr)": "tab:orange",
    "cross-plane (kz)": "tab:green",
}

# how to label NEP curves in the plot
NEP_PRETTY = {
    "kx_tot": r"NEP $k_x$ [$\bar{1}\bar{1}20$]",
    "ky_tot": r"NEP $k_y$ [$\bar{1}100$]",
    "kz_tot": r"NEP $k_z$ [$0001$]",
}

# how to label experiment curves
EXP_PRETTY = {
    "in-plane (kr)": "Exp in-plane (kr)",
    "cross-plane (kz)": "Exp cross-plane (kz)",
}



# Mapping from NEP log keys -> direction label
# Your log uses kx_tot, ky_tot, kz_tot
# NEP_LOG_KEY_MAP = {
#     "[100]": "kx_tot",
#     "[010]": "ky_tot",
#     "[001]": "kz_tot",
# }


# Figure output
FIGSIZE = (8.5, 6.5)
SAVE_FIG = True
OUTFIG = "kappa_exp_vs_nep.png"
DPI = 300

# ============================================================
# Font control (single variable)
# ============================================================
FONT_SCALE = 1.4   # <-- increase this to make everything larger (e.g. 1.6, 1.8)

BASE_FONT = 14
plt.rcParams.update({
    "font.size": BASE_FONT * FONT_SCALE,
    "axes.labelsize": (BASE_FONT + 2) * FONT_SCALE,
    "axes.titlesize": (BASE_FONT + 1) * FONT_SCALE,
    "xtick.labelsize": (BASE_FONT - 1) * FONT_SCALE,
    "ytick.labelsize": (BASE_FONT - 1) * FONT_SCALE,
    "legend.fontsize": (BASE_FONT - 2) * FONT_SCALE,
    "lines.linewidth": 2.0,
    "errorbar.capsize": 4,
})

# --- User variables (add near FONT_SCALE / plot knobs) ---
MARKER_SIZE_EXP = 70   # scatter size for experimental points (matplotlib "s" is area)
MARKER_SIZE_NEP = 7.5  # NEP marker size in errorbar (points)

# --- (Optional) also scale marker sizes with FONT_SCALE ---
MARKER_SIZE_EXP = MARKER_SIZE_EXP * (FONT_SCALE ** 2)
MARKER_SIZE_NEP = MARKER_SIZE_NEP * FONT_SCALE

DIR_PRETTY = {
    "in-plane (kr)": "in-plane (kr)",
    "cross-plane (kz)": "cross-plane (kz)",
}

# ============================================================
# Helpers
# ============================================================

def folder_to_temperature_k(folder_name: str) -> float:
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[kK]\s*$", folder_name)
    if not m:
        raise ValueError(f"Cannot parse temperature from folder name: {folder_name}")
    return float(m.group(1))


def discover_nep_folders(base_dir: str) -> List[str]:
    if NEP_FOLDER_GLOB is None:
        return NEP_FOLDERS[:]
    candidates = glob.glob(os.path.join(base_dir, NEP_FOLDER_GLOB))
    out = []
    for p in candidates:
        name = os.path.basename(os.path.normpath(p))
        if re.match(NEP_FOLDER_REGEX, name):
            out.append(name)
    return sorted(out, key=folder_to_temperature_k)


def latest_file_by_mtime(paths: List[str]) -> str:
    if not paths:
        raise FileNotFoundError("No files found to choose latest from.")
    return max(paths, key=lambda p: os.path.getmtime(p))


def read_experimental_csv(path: str) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """
    Supports your multi-block CSV with label lines like:
      Guo et al.
      Direction,Temperature_K,ThermalConductivity_W_per_mK
      ...
      Galazka et al.
      Direction,Temperature_K,ThermalConductivity_W_per_mK
      ...
    """
    data: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
    current_dataset = "Unknown"

    header_re = re.compile(r"^\s*Direction\s*,\s*Temperature_K\s*,\s*ThermalConductivity_W_per_mK\s*$", re.I)

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if header_re.match(line):
                continue
            if "," not in line:
                current_dataset = line
                data.setdefault(current_dataset, {})
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            d, t_str, k_str = parts[0], parts[1], parts[2]
            try:
                T = float(t_str)
                k = float(k_str)
            except ValueError:
                continue

            data.setdefault(current_dataset, {}).setdefault(d, []).append((T, k))

    for ds in data:
        for d in data[ds]:
            data[ds][d] = sorted(data[ds][d], key=lambda x: x[0])
    return data


def filter_by_temp(points: List[Tuple[float, float]], tmin: float, tmax: float) -> List[Tuple[float, float]]:
    return [(t, k) for (t, k) in points if (tmin <= t <= tmax)]


def parse_nep_log_for_mean_stderr(log_path: str) -> Dict[str, Tuple[float, float]]:
    """
    Parse lines like:
      kx_tot: mean =    3.276 W/mK, std =    2.433, stderr =    0.769
    Return:
      {"kx_tot": (mean, stderr), "ky_tot": (...), "kz_tot": (...)}
    """
    out: Dict[str, Tuple[float, float]] = {}
    # be tolerant of spaces
    pat = re.compile(
        r"^\s*(kx_tot|ky_tot|kz_tot|k_iso)\s*:\s*mean\s*=\s*([+-]?\d+(?:\.\d+)?)\s*W/mK\s*,\s*std\s*=\s*([+-]?\d+(?:\.\d+)?)\s*,\s*stderr\s*=\s*([+-]?\d+(?:\.\d+)?)\s*$"
    )
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                key = m.group(1)
                mean = float(m.group(2))
                stderr = float(m.group(4))
                out[key] = (mean, stderr)
    return out


def load_nep_points_with_errorbars(base_dir: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    For each temp folder, find latest postprocess_*.log, parse mean & stderr,
    and return arrays sorted by T:
      T_arr
      nep_mean[dir]
      nep_stderr[dir]
    """
    folders = discover_nep_folders(base_dir)

    Ts: List[float] = []
    # --- REVISE: in load_nep_points_with_errorbars(), initialize using NEP_DIRECTIONS ---
    means_by_dir = {k: [] for k in NEP_DIRECTIONS}
    errs_by_dir  = {k: [] for k in NEP_DIRECTIONS}

    for folder in folders:
        T = folder_to_temperature_k(folder)
        if not (T_MIN <= T <= T_MAX):
            continue

        fpath = os.path.join(base_dir, folder)
        logs = glob.glob(os.path.join(fpath, NEP_LOG_GLOB))
        if not logs:
            print(f"[warn] no log files matching {NEP_LOG_GLOB} in {fpath} (skipping)")
            continue

        latest_log = latest_file_by_mtime(logs)
        stats = parse_nep_log_for_mean_stderr(latest_log)

        # require all requested directions present
        ok = True
        for d in NEP_DIRECTIONS:
            # --- REVISE: in load_nep_points_with_errorbars(), require keys and append ---
            # require all requested NEP components present
            for key in NEP_DIRECTIONS:
                if key not in stats:
                    print(f"[warn] {latest_log} missing '{key}' line (skipping folder {folder})")
                    ok = False
                    break
            if not ok:
                continue

            Ts.append(T)
            for key in NEP_DIRECTIONS:
                mean, stderr = stats[key]
                means_by_dir[key].append(mean)
                errs_by_dir[key].append(stderr)


    # --- REVISE: in load_nep_points_with_errorbars(), return arrays for NEP_DIRECTIONS ---
    if not Ts:
        return np.array([]), {k: np.array([]) for k in NEP_DIRECTIONS}, {k: np.array([]) for k in NEP_DIRECTIONS}

    order = np.argsort(Ts)
    T_arr = np.array(Ts, dtype=float)[order]
    mean_arrs = {k: np.array(means_by_dir[k], dtype=float)[order] for k in NEP_DIRECTIONS}
    err_arrs  = {k: np.array(errs_by_dir[k], dtype=float)[order] for k in NEP_DIRECTIONS}
    return T_arr, mean_arrs, err_arrs



# ============================================================
# Main
# ============================================================

def main():
    base_dir = os.path.abspath(BASE_DIR)

    # --- experimental ---
    exp_path = os.path.join(base_dir, EXPERIMENT_CSV)
    if not os.path.isfile(exp_path):
        raise FileNotFoundError(f"Experimental CSV not found: {exp_path}")
    exp_data = read_experimental_csv(exp_path)

    # --- NEP (mean+stderr from latest log per folder) ---
    nep_T, nep_mean, nep_err = load_nep_points_with_errorbars(base_dir)
    if nep_T.size == 0:
        print("[warn] no NEP points found in range; only plotting experiment.")

    # --- plot ---
    plt.figure(figsize=FIGSIZE)

    # --- REVISE: experimental plotting loop: use EXP_DIRECTIONS (not DIRECTIONS) ---
    for ds_name, ds in exp_data.items():
        for d in EXP_DIRECTIONS:
            if d not in ds:
                continue
            pts = filter_by_temp(ds[d], T_MIN, T_MAX)
            if not pts:
                continue
            t = [p[0] for p in pts]
            k = [p[1] for p in pts]
            # --- REVISE: experimental scatter label line ---
            # --- REVISE: experimental scatter call (add color=...) ---
            plt.scatter(
                t, k,
                marker="o",
                s=MARKER_SIZE_EXP,
                color=EXP_COLOR.get(d, None),
                label=EXP_LABEL_SHORT.get(d, f"4H(SI) Exp {d}"),
            )





    # --- REVISE: NEP plotting loop: use NEP_DIRECTIONS keys directly ---
    if nep_T.size > 0:
        for key in NEP_DIRECTIONS:
            plt.errorbar(
                nep_T,
                nep_mean[key],
                yerr=nep_err[key],
                fmt="s-",
                markersize=MARKER_SIZE_NEP,
                label=NEP_PRETTY.get(key, f"NEP {key}"),
            )


    plt.xlabel("Temperature (K)")
    plt.ylabel(r"Thermal Conductivity (W m$^{-1}$ K$^{-1}$)")
    plt.xlim(T_MIN, T_MAX)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(ncol=1, frameon=True)
    plt.tight_layout()

    if SAVE_FIG:
        outpath = os.path.join(base_dir, OUTFIG)
        plt.savefig(outpath, dpi=DPI)
        plt.savefig(f"{outpath.replace('.png', '.pdf')}", bbox_inches="tight")
        print(f"[ok] saved: {outpath}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
