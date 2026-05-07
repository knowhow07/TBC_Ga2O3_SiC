#!/usr/bin/env python3
import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif"]

plt.rcParams.update({"font.size": 16})


# ============================================================
# RMSE helpers
# ============================================================

def rmse_two_col(fname):
    if not os.path.exists(fname):
        return float("nan")
    a = np.loadtxt(fname)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    diff = a[:, 0] - a[:, 1]
    return math.sqrt((diff**2).mean())


def rmse_force(fname):
    if not os.path.exists(fname):
        return float("nan")
    a = np.loadtxt(fname)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape[1] != 6:
        raise ValueError(f"Unexpected force_train.out shape: {a.shape}")
    Fx_pred, Fy_pred, Fz_pred, Fx_ref, Fy_ref, Fz_ref = a.T
    diff = np.concatenate([
        Fx_pred - Fx_ref,
        Fy_pred - Fy_ref,
        Fz_pred - Fz_ref,
    ])
    return math.sqrt((diff**2).mean())


def rmse_virial(fname):
    if not os.path.exists(fname):
        return float("nan")
    a = np.loadtxt(fname)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape[1] % 2 != 0:
        raise ValueError(f"Unexpected virial/stress shape: {a.shape}")
    ncomp = a.shape[1] // 2
    pred = a[:, :ncomp]
    ref = a[:, ncomp:]
    diff = (pred - ref).ravel()
    return math.sqrt((diff**2).mean())


# ============================================================
# Collect prediction/reference data for parity plots
# ============================================================

def collect_energy(root):
    e_pred, e_ref = [], []
    for name in os.listdir(root):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        f = os.path.join(d, "energy_train.out")
        if not os.path.exists(f):
            continue
        a = np.loadtxt(f)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.shape[1] < 2:
            continue
        e_pred.append(a[:, 0])
        e_ref.append(a[:, 1])
    if not e_pred:
        return None, None
    return np.concatenate(e_pred), np.concatenate(e_ref)


def collect_force(root):
    f_pred, f_ref = [], []
    for name in os.listdir(root):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        f = os.path.join(d, "force_train.out")
        if not os.path.exists(f):
            continue
        a = np.loadtxt(f)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.shape[1] != 6:
            continue
        Fx_p, Fy_p, Fz_p, Fx_r, Fy_r, Fz_r = a.T
        f_pred.append(np.concatenate([Fx_p, Fy_p, Fz_p]))
        f_ref.append(np.concatenate([Fx_r, Fy_r, Fz_r]))
    if not f_pred:
        return None, None
    return np.concatenate(f_pred), np.concatenate(f_ref)


def collect_virial(root):
    v_pred, v_ref = [], []
    for name in os.listdir(root):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        vf = os.path.join(d, "virial_train.out")
        sf = os.path.join(d, "stress_train.out")
        f = vf if os.path.exists(vf) else sf if os.path.exists(sf) else None
        if f is None:
            continue
        a = np.loadtxt(f)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.shape[1] % 2 != 0:
            continue
        ncomp = a.shape[1] // 2
        v_pred.append(a[:, :ncomp].ravel())
        v_ref.append(a[:, ncomp:].ravel())
    if not v_pred:
        return None, None
    return np.concatenate(v_pred), np.concatenate(v_ref)


# ============================================================
# Plot helpers
# ============================================================

def parity_plot(pred, ref, xlabel, ylabel, title, out_png, out_pdf):
    if pred is None or ref is None:
        print(f"[WARN] No data for {out_png}")
        return

    rmse = math.sqrt(np.mean((pred - ref) ** 2))
    vmin = min(pred.min(), ref.min())
    vmax = max(pred.max(), ref.max())
    margin = 0.05 * (vmax - vmin) if vmax > vmin else 1e-12
    lo, hi = vmin - margin, vmax + margin

    plt.figure(figsize=(6, 6))
    plt.scatter(ref, pred, s=5, alpha=0.4)
    plt.plot([lo, hi], [lo, hi], "k--")
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.text(
        0.05, 0.95, f"RMSE={rmse:.3e}",
        transform=plt.gca().transAxes,
        ha="left", va="top", fontsize=14
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out_png}")
    print(f"[OK] Saved {out_pdf}")


def plot_bar(df, col, ylabel, out_png, out_pdf):
    plt.figure(figsize=(14, 6))
    x = df["dataset"]
    y = df[col]
    plt.bar(x, y, edgecolor="black")
    plt.xticks(rotation=75, fontsize=14, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"{col} per dataset")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {out_png}")
    print(f"[OK] Saved {out_pdf}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="tests", help="Root folder containing per-dataset subfolders")
    parser.add_argument("--summary", default=None, help="Path to rmse_summary.csv")
    args = parser.parse_args()

    root = args.root
    summary_csv = args.summary or os.path.join(root, "rmse_summary.csv")

    # parity plots
    e_p, e_r = collect_energy(root)
    parity_plot(
        e_p, e_r,
        "DFT Energy (eV)", "NEP Energy (eV)",
        "Energy Parity (All Tests)",
        os.path.join(root, "parity_energy.png"),
        os.path.join(root, "parity_energy.pdf"),
    )

    f_p, f_r = collect_force(root)
    parity_plot(
        f_p, f_r,
        "DFT Force (eV/Å)", "NEP Force (eV/Å)",
        "Force Parity (All Tests)",
        os.path.join(root, "parity_force.png"),
        os.path.join(root, "parity_force.pdf"),
    )

    v_p, v_r = collect_virial(root)
    parity_plot(
        v_p, v_r,
        "DFT Virial/Stress", "NEP Virial/Stress",
        "Virial Parity (All Tests)",
        os.path.join(root, "parity_virial.png"),
        os.path.join(root, "parity_virial.pdf"),
    )

    # bar plots
    if not os.path.exists(summary_csv):
        print(f"[WARN] Summary file not found: {summary_csv}")
        return

    df = pd.read_csv(summary_csv)

    plot_bar(
        df, "E_RMSE", "Energy RMSE (eV)",
        os.path.join(root, "bar_E_RMSE.png"),
        os.path.join(root, "bar_E_RMSE.pdf"),
    )
    plot_bar(
        df, "F_RMSE", "Force RMSE (eV/Å)",
        os.path.join(root, "bar_F_RMSE.png"),
        os.path.join(root, "bar_F_RMSE.pdf"),
    )
    plot_bar(
        df, "V_RMSE", "Virial RMSE",
        os.path.join(root, "bar_V_RMSE.png"),
        os.path.join(root, "bar_V_RMSE.pdf"),
    )


if __name__ == "__main__":
    main()