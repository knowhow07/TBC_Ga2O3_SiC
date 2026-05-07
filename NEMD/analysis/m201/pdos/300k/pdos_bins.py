#!/usr/bin/env python3
import os, re
import numpy as np
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================
SNAPSHOT_XYZ = "model.xyz"
VELOCITY_OUT = "velocity.out"

GROUP_ID = 2
X_INTERFACE = 100.0

WINDOW_W = 5.0          # bin width (Å)
GAP = 20.0              # step between adjacent bin centers (Å), e.g. 10 Å
N_BINS_EACH_SIDE = 3    # how many bins to plot on each side (excluding interface-centered bin)

DT_FS = 1.0
DUMP_EVERY = 10
FREQ_MAX_THz = 80.0

MIN_ATOMS_PER_BIN = 80

# IO mode: "out" parses velocity.out once and caches; "npy" reads cache only
IO_MODE = "npy"   # "out" or "npy"
CACHE_NPY = f"velocity_g{GROUP_ID}.npy"
STRICT = True

SMOOTH_POINTS = 11  # odd int; 1=off
# =========================


def read_snapshot_xyz(fname):
    with open(fname, "r") as f:
        N = int(f.readline().strip())
        header2 = f.readline().strip()
        lines = [f.readline().strip() for _ in range(N)]

    x = np.zeros(N, float)
    g = np.full(N, -1, int)

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 4:
            raise RuntimeError(f"{fname}: bad atom line: {line}")

        try:
            int(parts[0])  # id exists
            x[i] = float(parts[2])
            if len(parts) >= 6:
                g[i] = int(parts[5])
        except ValueError:
            x[i] = float(parts[1])
            if len(parts) == 5:
                g[i] = int(parts[4])

    Lx = None
    m = re.search(r'Lattice="([^"]+)"', header2)
    if m:
        lat = list(map(float, m.group(1).split()))
        if len(lat) >= 1:
            Lx = float(lat[0])

    if np.all(g < 0):
        raise RuntimeError(f"{fname}: no group column detected.")

    return x, g, Lx


def _sanity_check_velocity_rows(total_rows: int, n_atoms: int, fname: str):
    if n_atoms <= 0:
        raise RuntimeError(f"{fname}: n_atoms must be > 0, got {n_atoms}")
    if total_rows % n_atoms != 0:
        gg = np.gcd(total_rows, n_atoms)
        raise RuntimeError(
            f"{fname}: row count mismatch.\n"
            f"  total_rows={total_rows}\n"
            f"  n_atoms={n_atoms}\n"
            f"  total_rows % n_atoms={total_rows % n_atoms}\n"
            f"  gcd={gg}\n"
            f"Likely mixed segments or wrong group.\n"
        )
    n_frames = total_rows // n_atoms
    if n_frames < 2:
        raise RuntimeError(f"{fname}: inferred n_frames={n_frames} (<2)")
    return n_frames


def read_velocity_rows(mode: str, out_fname: str, cache_npy: str) -> np.ndarray:
    # returns (total_rows, 3) float32; npy uses mmap
    if mode == "out":
        if os.path.exists(cache_npy):
            return np.load(cache_npy, mmap_mode="r")
        data = np.loadtxt(out_fname, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != 3:
            raise RuntimeError(f"{out_fname}: parse failed (expected 3 columns).")
        np.save(cache_npy, data)
        return np.load(cache_npy, mmap_mode="r")
    elif mode == "npy":
        if not os.path.exists(cache_npy):
            raise RuntimeError(f"Missing {cache_npy}. Run once with IO_MODE='out'.")
        return np.load(cache_npy, mmap_mode="r")
    else:
        raise ValueError("IO_MODE must be 'out' or 'npy'")


def reshape_velocity_rows(data: np.ndarray, n_atoms: int, fname_for_msg: str, strict: bool) -> np.ndarray:
    total_rows = int(data.shape[0])
    if total_rows % n_atoms != 0:
        if strict:
            _sanity_check_velocity_rows(total_rows, n_atoms, fname_for_msg)
        else:
            rem = total_rows % n_atoms
            print(f"[WARN] {fname_for_msg}: dropping last {rem} rows (incomplete frame)")
            total_rows -= rem
            data = data[:total_rows]
    n_frames = total_rows // n_atoms
    return data.reshape(n_frames, n_atoms, 3)


def smooth_1d(y: np.ndarray, points: int) -> np.ndarray:
    if points <= 1:
        return y
    if points % 2 == 0:
        points += 1
    k = np.ones(points, dtype=float) / points
    pad = points // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, k, mode="valid")


def pdos_from_v(v, dt_s, fmax_THz=None):
    # remove drift
    v = v - v.mean(axis=0, keepdims=True)

    T = v.shape[0]
    y = v.reshape(T, -1)  # (T, 3*N)

    Y = np.fft.rfft(y, axis=0)
    S = (Y * np.conj(Y)).real
    vacf = np.fft.irfft(S, n=T, axis=0).mean(axis=1)
    vacf /= (vacf[0] + 1e-30)

    G = np.fft.rfft(vacf)
    dos = G.real

    freqs_THz = np.fft.rfftfreq(T, d=dt_s) * 1e-12
    if fmax_THz is not None:
        m = freqs_THz <= fmax_THz
        freqs_THz = freqs_THz[m]
        dos = dos[m]

    area = np.trapz(dos, freqs_THz)
    if area > 0:
        dos = dos / area
    return freqs_THz, dos


def build_bin_windows(x_interface, window_w, gap, n_each_side):
    """
    Returns ordered list of (name, lo, hi, center).
    Order is left->right, but we will plot top->bottom as left->right.
    Include:
      overlap-left centered at x_interface - window_w/2
      interface centered at x_interface
      overlap-right centered at x_interface + window_w/2
    plus bins stepping outward by GAP.
    """
    bins = []

    # centers we want (left -> right)
    centers = []

    # far left to overlap-left
    # centers: x_interface - window_w/2 - k*gap ... (k=n_each_side-1 ... 1) then overlap-left
    overlap_left_c = x_interface - 0.5 * window_w
    for k in range(n_each_side-1, 0, -1):
        centers.append(overlap_left_c - k * gap)
    centers.append(overlap_left_c)

    # interface
    centers.append(x_interface)

    # overlap-right and to far right
    overlap_right_c = x_interface + 0.5 * window_w
    centers.append(overlap_right_c)
    for k in range(1, n_each_side):
        centers.append(overlap_right_c + k * gap)

    # build windows
    for c in centers:
        lo = c - 0.5 * window_w
        hi = c + 0.5 * window_w
        if abs(c - x_interface) < 1e-6:
            name = f"Interface (c={c:.1f}Å)"
        elif abs(c - overlap_left_c) < 1e-6:
            name = f"Overlap-L (c={c:.1f}Å)"
        elif abs(c - overlap_right_c) < 1e-6:
            name = f"Overlap-R (c={c:.1f}Å)"
        else:
            name = f"c={c:.1f}Å"
        bins.append((name, lo, hi, c))

    return bins


def main():
    x_all, g_all, Lx = read_snapshot_xyz(SNAPSHOT_XYZ)
    idx_g = np.where(g_all == GROUP_ID)[0]
    if idx_g.size == 0:
        raise RuntimeError(f"No atoms with group == {GROUP_ID} found in {SNAPSHOT_XYZ}")
    x_g = x_all[idx_g]

    # load velocity rows then reshape
    rows = read_velocity_rows(IO_MODE, VELOCITY_OUT, CACHE_NPY)
    v = reshape_velocity_rows(rows, n_atoms=idx_g.size,
                              fname_for_msg=(VELOCITY_OUT if IO_MODE=="out" else CACHE_NPY),
                              strict=STRICT)

    dt_s = DT_FS * 1e-15 * DUMP_EVERY
    print(f"Lx={Lx}, group{GROUP_ID} atoms={idx_g.size}, frames={v.shape[0]}, dt_sample={dt_s*1e15:.1f} fs")

    # build bin list (left->right); plot top->bottom in same order
    bins = build_bin_windows(X_INTERFACE, WINDOW_W, GAP, N_BINS_EACH_SIDE)

    # figure layout like paper: stacked axes
    n_panels = len(bins)
    fig_h = max(7, 1.3 * n_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, fig_h), sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (name, lo, hi, c) in zip(axes, bins):
        sel = np.where((x_g >= lo) & (x_g < hi))[0]
        ax.text(0.98, 0.80, f"{name}\n[{lo:.1f},{hi:.1f}) Å\nn={sel.size}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)

        if sel.size < MIN_ATOMS_PER_BIN:
            ax.plot([], [])
            ax.text(0.5, 0.5, "too few atoms", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10)
            continue

        f, dos = pdos_from_v(v[:, sel, :], dt_s, fmax_THz=FREQ_MAX_THz)
        dos = smooth_1d(dos, SMOOTH_POINTS)
        ax.plot(f, dos, linewidth=1.6)

        ax.axhline(0, linewidth=0.6)

        # mark interface panel subtly
        if "Interface" in name:
            ax.set_facecolor((0.95, 0.95, 0.95))

    axes[-1].set_xlabel("Frequency (THz)")
    axes[n_panels // 2].set_ylabel("Normalized PDOS (a.u.)")

    axes[-1].set_xlim(0, min(FREQ_MAX_THz, f.max()))
    fig.suptitle(f"Stacked PDOS (group {GROUP_ID})  window={WINDOW_W:.1f}Å  gap={GAP:.1f}Å  x_int={X_INTERFACE:.1f}Å",
                 y=0.995, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out = f"pdos_stacked_g{GROUP_ID}_w{WINDOW_W:g}_gap{GAP:g}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[OK] Saved {out}")


if __name__ == "__main__":
    main()
