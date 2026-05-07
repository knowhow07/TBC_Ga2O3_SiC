#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import re

# =========================
# USER SETTINGS
# =========================
SNAPSHOT_XYZ = "model.xyz"
VELOCITY_OUT = "velocity.out"



GROUP_ID = 2            # the group you dumped (group 2)
X_INTERFACE = 105.0     # your user-defined interface position (Å), e.g. 105

WINDOW_W = 5.0         # Å, width of each region window
# left window:  [X_INTERFACE - WINDOW_W, X_INTERFACE)
# mid window:   [X_INTERFACE - WINDOW_W/2, X_INTERFACE + WINDOW_W/2]
# right window: [X_INTERFACE, X_INTERFACE + WINDOW_W)

DT_FS = 1.0             # timestep (fs)
DUMP_EVERY = 10          # dump_velocity interval (steps)
FREQ_MAX_THz = 80.0      # plotting max frequency (THz)

MIN_ATOMS_PER_REGION = 50  # skip if too few atoms (noise)

# =========================
# IO / MODE SETTINGS
# =========================
IO_MODE = "out"   # "out" = read velocity.out (and cache), "npy" = read velocity_g2.npy only
CACHE_NPY = f"velocity_g{GROUP_ID}.npy"   # e.g. velocity_g2.npy

STRICT = True     # True: require exact divisibility; False: drop incomplete tail
SMOOTH_POINTS = 9 # odd integer, 1=off. Try 9, 15, 21 for more smoothing
# =========================


def read_snapshot_xyz(fname):
    """Read extended XYZ and return x, group(if present else None), Lx."""
    with open(fname, "r") as f:
        N = int(f.readline().strip())
        header2 = f.readline().strip()
        lines = [f.readline().strip() for _ in range(N)]

    x = np.zeros(N, float)
    g = np.full(N, -1, int)   # -1 means missing

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 4:
            raise RuntimeError(f"{fname}: bad atom line (too few columns): {line}")

        # Determine whether line starts with id or species
        # id sp x y z [group]
        # sp x y z [group]
        try:
            int(parts[0])  # id exists
            x[i] = float(parts[2])
            # group is present only if there are >= 6 columns
            if len(parts) >= 6:
                g[i] = int(parts[5])
        except ValueError:
            x[i] = float(parts[1])
            if len(parts) >= 5:
                # here 5th could be group or z depending on format
                # if there are 5 columns total: sp x y z group  (group present)
                # if there are 4 columns total: sp x y z        (no group)
                # we already checked len(parts) >=4
                if len(parts) == 5:
                    # last is group
                    g[i] = int(parts[4])
                # else len==4 -> no group

    # Parse Lx if present (optional)
    Lx = None
    m = re.search(r'Lattice="([^"]+)"', header2)
    if m:
        lat = list(map(float, m.group(1).split()))
        if len(lat) >= 1:
            Lx = float(lat[0])

    # Require group column for your workflow
    if np.all(g < 0):
        raise RuntimeError(
            f"{fname}: no group column detected in atom lines.\n"
            f"For mapping velocity.out dumped with 'group 0 2', you must read from the GROUPED model.xyz.\n"
            f"Set SNAPSHOT_XYZ = 'model.xyz' (the regrouped one)."
        )

    return x, g, Lx



def _sanity_check_velocity_rows(total_rows: int, n_atoms: int, fname: str):
    """
    Sanity checks for GPUMD velocity.out:
    - total_rows must be divisible by n_atoms
    - inferred n_frames should be >= 2 (otherwise PDOS is meaningless)
    """
    if n_atoms <= 0:
        raise RuntimeError(f"{fname}: n_atoms must be > 0, got {n_atoms}")

    if total_rows % n_atoms != 0:
        # helpful diagnostics
        g = np.gcd(total_rows, n_atoms)
        raise RuntimeError(
            f"{fname}: row count mismatch.\n"
            f"  total velocity rows = {total_rows}\n"
            f"  n_atoms (dumped group size) = {n_atoms}\n"
            f"  total_rows % n_atoms = {total_rows % n_atoms}\n"
            f"  gcd(total_rows, n_atoms) = {g}\n"
            f"Likely causes:\n"
            f"  - velocity.out corresponds to a different group than snapshot group {GROUP_ID}\n"
            f"  - velocity.out was overwritten by a later dump\n"
            f"  - velocity.out contains non-numeric/header lines not filtered out\n"
        )

    n_frames = total_rows // n_atoms
    if n_frames < 2:
        raise RuntimeError(
            f"{fname}: inferred n_frames={n_frames} (too small for PDOS). "
            f"Need >= 2 frames."
        )
    return n_frames


def read_velocity_out_fast(fname, n_atoms):
    data = np.loadtxt(fname, dtype=np.float32)  # expects exactly 3 cols
    total_rows = data.shape[0]
    rem = total_rows % n_atoms
    if rem != 0:
        print(f"[WARN] dropping last {rem} rows (incomplete frame)")
        data = data[: total_rows - rem]
        total_rows = data.shape[0]
    n_frames = total_rows // n_atoms
    return data.reshape(n_frames, n_atoms, 3)

import os

import os

import os

def _load_vel_rows_from_out(fname: str) -> np.ndarray:
    data = np.loadtxt(fname, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 3:
        raise RuntimeError(f"{fname}: parse failed (expected 3 numeric columns).")
    return data

def _load_vel_rows_from_npy(npy: str) -> np.ndarray:
    if not os.path.exists(npy):
        raise RuntimeError(f"Missing {npy}. Set IO_MODE='out' once to create it.")
    return np.load(npy, mmap_mode="r")  # (total_rows, 3)

def read_velocity_rows(mode: str, out_fname: str, cache_npy: str) -> np.ndarray:
    """
    Returns data with shape (total_rows, 3) as float32 (mmap if npy).
    """
    if mode == "out":
        # if cache exists, use it; else parse velocity.out then save
        if os.path.exists(cache_npy):
            return np.load(cache_npy, mmap_mode="r")
        data = _load_vel_rows_from_out(out_fname)
        np.save(cache_npy, data)
        return np.load(cache_npy, mmap_mode="r")
    elif mode == "npy":
        return _load_vel_rows_from_npy(cache_npy)
    else:
        raise ValueError("IO_MODE must be 'out' or 'npy'")

def reshape_velocity_rows(data: np.ndarray, n_atoms: int, fname_for_msg: str, strict: bool) -> np.ndarray:
    total_rows = int(data.shape[0])

    if total_rows % n_atoms != 0:
        if strict:
            _sanity_check_velocity_rows(total_rows, n_atoms, fname_for_msg)  # raises
        else:
            rem = total_rows % n_atoms
            print(f"[WARN] {fname_for_msg}: dropping last {rem} rows (incomplete frame)")
            total_rows -= rem
            data = data[:total_rows]

    n_frames = total_rows // n_atoms
    if n_frames < 2:
        raise RuntimeError(f"{fname_for_msg}: inferred n_frames={n_frames} (<2)")

    return data.reshape(n_frames, n_atoms, 3)
def smooth_1d(y: np.ndarray, points: int) -> np.ndarray:
    if points <= 1:
        return y
    if points % 2 == 0:
        points += 1  # force odd
    k = np.ones(points, dtype=float) / points
    # pad edges to avoid end dips
    pad = points // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, k, mode="valid")





def pdos_from_v(v, dt_s, fmax_THz=None):
    """
    Compute PDOS from velocity time series:
    - compute VACF via FFT autocorrelation (averaged over atoms + components)
    - take FFT of VACF to get PDOS-like spectrum
    """
    # remove mean drift per DOF
    v = v - v.mean(axis=0, keepdims=True)

    T = v.shape[0]
    y = v.reshape(T, -1)  # (T, 3*N)

    Y = np.fft.rfft(y, axis=0)
    S = (Y * np.conj(Y)).real
    vacf = np.fft.irfft(S, n=T, axis=0).mean(axis=1)
    vacf /= (vacf[0] + 1e-30)

    G = np.fft.rfft(vacf)
    dos = G.real

    freqs_Hz = np.fft.rfftfreq(T, d=dt_s)
    freqs_THz = freqs_Hz * 1e-12

    if fmax_THz is not None:
        m = freqs_THz <= fmax_THz
        freqs_THz = freqs_THz[m]
        dos = dos[m]

    # normalize area for comparison
    area = np.trapz(dos, freqs_THz)
    if area > 0:
        dos = dos / area

    return freqs_THz, dos


def main():


    x_all, g_all, Lx = read_snapshot_xyz(SNAPSHOT_XYZ)
    idx_g = np.where(g_all == GROUP_ID)[0]
    if idx_g.size == 0:
        raise RuntimeError(f"No atoms with group == {GROUP_ID} found in {SNAPSHOT_XYZ}")

    # x positions for the dumped group, in snapshot order
    x_g = x_all[idx_g]
    x_min = float(x_g.min())
    x_max = float(x_g.max())


    # Define your three regions inside group 2
    # Far-left / interface-mid / far-right slices within group 2
    left_lo  = x_min
    left_hi  = x_min + WINDOW_W

    mid_lo   = X_INTERFACE - 0.5 * WINDOW_W
    mid_hi   = X_INTERFACE + 0.5 * WINDOW_W

    right_lo = x_max - WINDOW_W
    right_hi = x_max


    mask_left  = (x_g >= left_lo)  & (x_g < left_hi)
    mask_mid   = (x_g >= mid_lo)   & (x_g < mid_hi)
    mask_right = (x_g >= right_lo) & (x_g <= right_hi)

    sel_left = np.where(mask_left)[0]
    sel_mid  = np.where(mask_mid)[0]
    sel_right= np.where(mask_right)[0]

    print(f"Snapshot Lx={Lx}")
    # print(f"Group {GROUP_ID} atoms = {idx_g.size}")
    # print(f"Left  window [{left_lo:.2f},{left_hi:.2f}) Å: n={sel_left.size}")
    # print(f"Mid   window [{mid_lo:.2f},{mid_hi:.2f}] Å: n={sel_mid.size}")
    # print(f"Right window ({right_lo:.2f},{right_hi:.2f}] Å: n={sel_right.size}")

    print(f"group2 x-range: [{x_min:.2f}, {x_max:.2f}] Å")
    print(f"Left  slice: [{left_lo:.2f}, {left_hi:.2f})")
    print(f"Mid   slice: [{mid_lo:.2f}, {mid_hi:.2f})")
    print(f"Right slice: [{right_lo:.2f}, {right_hi:.2f}]")


    # Read velocity.out for dumped group size
    rows = read_velocity_rows(IO_MODE, VELOCITY_OUT, CACHE_NPY)
    v = reshape_velocity_rows(rows, n_atoms=idx_g.size,
                            fname_for_msg=(VELOCITY_OUT if IO_MODE=="out" else CACHE_NPY),
                            strict=STRICT)


    dt_s = DT_FS * 1e-15 * DUMP_EVERY
    print(f"Velocity frames = {v.shape[0]}, dt_sample = {dt_s*1e15:.1f} fs")

    # Compute PDOS for each region
    plt.figure(figsize=(8, 6))

    def do_region(sel, name):
        if sel.size < MIN_ATOMS_PER_REGION:
            print(f"[WARN] {name}: too few atoms ({sel.size}) -> skip")
            return
        f, dos = pdos_from_v(v[:, sel, :], dt_s, fmax_THz=FREQ_MAX_THz)
        dos_s = smooth_1d(dos, SMOOTH_POINTS)
        plt.plot(f, dos_s, linewidth=2, label=f"{name} (n={sel.size})")

    do_region(sel_left,  f"Left {WINDOW_W:.0f}Å")
    do_region(sel_mid,   f"Mid {WINDOW_W:.0f}Å @ x={X_INTERFACE:.1f}")
    do_region(sel_right, f"Right {WINDOW_W:.0f}Å")

    plt.xlabel("Frequency (THz)")
    plt.ylabel("Normalized PDOS (a.u.)")
    plt.title(f"PDOS from group {GROUP_ID} velocity.out (x_interface={X_INTERFACE:.1f} Å, bin={WINDOW_W:.0f} Å)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"pdos_left_mid_right_{WINDOW_W}_{IO_MODE}_extreme.png", dpi=300)
    plt.close()
    print(f"[OK] Saved pdos_left_mid_right_{WINDOW_W}_{IO_MODE}_extreme.png")


if __name__ == "__main__":
    main()
