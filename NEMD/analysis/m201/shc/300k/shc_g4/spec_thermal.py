#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import re

# ---------------- user knobs ----------------
COMPUTE = "compute.out"
SKIP_FRACTION = 0.2   # use last (1-skip) as steady state
# --- ADD/REVISE in the USER KNOBS block (near COMPUTE=...) ---
SHC = "shc.out"     # <-- this was missing
MODEL = "model.xyz" # if you use it for area

M = 7                 # number of temperature groups output by compute keyword
G_LEFT  = 2           # dumbL group id (1..7)
G_RIGHT = 6           # dumbR group id (1..7)
Nc = 1000
num_omega = 4000

SHC_GROUP_ID = 4  # <-- set this to the group you computed SHC for

SMOOTH_W = 51         # moving average window (odd)
# -------------------------------------------
import sys
import time

LOG_FILE = f"spec_thermal_{time.strftime('%Y%m%d_%H%M%S')}.log"

class Tee:
    """Duplicate writes to multiple streams (e.g., terminal + log file)."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()
# --- enable tee logging for all prints ---
_log_fh = open(LOG_FILE, "w", buffering=1)   # line-buffered
sys.stdout = Tee(sys.__stdout__, _log_fh)
sys.stderr = Tee(sys.__stderr__, _log_fh)

print(f"[LOG] Writing to {LOG_FILE}")

def read_group_x_range(model_xyz: str, group_id: int):
    """
    Return (xmin, xmax, xmean) for atoms in a given group_id from model.xyz.
    Assumes last column is group id, and x is the 3rd coordinate column:
      id species x y z group
    """
    with open(model_xyz, "r") as f:
        N = int(f.readline().strip())
        _ = f.readline()
        xs = []
        for _ in range(N):
            parts = f.readline().split()
            if len(parts) < 6:
                continue
            x = float(parts[2])
            g = int(parts[-1])
            if g == group_id:
                xs.append(x)

    if len(xs) == 0:
        raise RuntimeError(f"No atoms found for group {group_id} in {model_xyz}")

    xs = np.asarray(xs, float)
    return float(xs.min()), float(xs.max()), float(xs.mean())

def read_LyLz_area_A2(model_xyz: str) -> float:
    with open(model_xyz, "r") as f:
        _ = f.readline()
        header = f.readline().strip()
    m = re.search(r'Lattice="([^"]+)"', header)
    if m is None:
        raise RuntimeError("Cannot find Lattice in model.xyz header line 2")
    lat = np.fromstring(m.group(1), sep=" ")
    if lat.size != 9:
        raise RuntimeError("Expected 9 lattice numbers")
    a1 = lat[0:3]; a2 = lat[3:6]; a3 = lat[6:9]
    Ly = np.linalg.norm(a2)
    Lz = np.linalg.norm(a3)
    return Ly * Lz  # Å^2

def moving_average(y, w):
    if w is None or w < 3:
        return y
    if w % 2 == 0:
        w += 1
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")

# --- REVISE THIS FUNCTION: thermo parsing was wrong ---
def read_group_temps_from_compute(compute_file: str, M: int) -> np.ndarray:
    """
    compute.out format for temperature:
      columns 1..M   : group temperatures (K), left->right
      last-2, last-1 : thermostat energies (source, sink)  [only if using heat_lan]
    We only need the first M columns.
    """
    data = np.loadtxt(compute_file)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < M:
        raise RuntimeError(f"compute.out has too few columns: {data.shape[1]} < M={M}")
    return data[:, :M]

def read_shc_omega_block(shc_file: str, num_omega: int):
    rows = []
    with open(shc_file) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    A = np.array(rows, float)
    if A.size == 0:
        raise RuntimeError(f"No numeric rows parsed from {shc_file}")

    # keep only rows that look like omega rows (omega >= 0)
    mask = np.isfinite(A[:, 0]) & (A[:, 0] >= 0.0)
    W = A[mask]

    # take the LAST num_omega rows (omega-block is at the end)
    if W.shape[0] < num_omega:
        raise RuntimeError(
            f"Found only {W.shape[0]} non-negative omega rows, but need num_omega={num_omega}. "
            f"Check that shc.out contains the omega-block."
        )

    Jw = W[-num_omega:, :]
    omega, Jin, Jout = Jw[:, 0]/ (2.0 * np.pi), Jw[:, 1], Jw[:, 2]

    # drop NaNs in Jin/Jout (still allow omega)
    good = np.isfinite(omega) & np.isfinite(Jin) & np.isfinite(Jout)
    omega, Jin, Jout = omega[good], Jin[good], Jout[good]

    if omega.size == 0:
        raise RuntimeError("Omega-block exists, but Jin/Jout are all NaN. Likely SHC not converged / bad run.")

    return omega, Jin, Jout
def main():
    # ---- ΔT from thermo.out ----
    Tg = read_group_temps_from_compute(COMPUTE, M)  # shape (nsteps, M)
    n = Tg.shape[0]
    start = int(np.floor(n * SKIP_FRACTION))
    Tg_ss = Tg[start:, :]

    Tavg = np.nanmean(Tg_ss, axis=0)  # g1..gM
    print("Steady-state group temperatures (K) from compute.out:")
    for i, t in enumerate(Tavg, start=1):
        print(f"  g{i}: {t:.6f}")

    T_left  = Tavg[G_LEFT - 1]
    T_right = Tavg[G_RIGHT - 1]
    dT = T_left - T_right
    print(f"\nDeltaT = <Tg{G_LEFT}> - <Tg{G_RIGHT}> = {dT:.6f} K")

    if abs(dT) < 1e-8:
        raise RuntimeError("DeltaT is ~0; cannot normalize SHC. Check compute.out and group ids.")
    # ---- SHC spectrum ----
    omega, Jin, Jout = read_shc_omega_block(SHC, num_omega)
    if omega.size == 0:
        raise RuntimeError(
            "Parsed 0 omega points from shc.out. "
            "Check Nc/num_omega match compute_shc, and confirm shc.out has the omega-block."
        )

    Jtot = Jin + Jout
    G_like = Jtot / dT

    A_A2 = read_LyLz_area_A2(MODEL)
    G_like_A = G_like / A_A2

    # ensure smoothing window not bigger than data
    w = SMOOTH_W
    if w is not None and w >= G_like_A.size:
        w = max(3, (G_like_A.size // 2) * 2 + 1)  # largest odd < size
    Gs = moving_average(G_like_A, w)

    cum = np.cumsum(0.5 * (Gs[1:] + Gs[:-1]) * (omega[1:] - omega[:-1]))
    cum = np.insert(cum, 0, 0.0)

    # ---- report spatial location (x-range) of the SHC group ----
    
    xmin, xmax, xmean = read_group_x_range(MODEL, SHC_GROUP_ID)
    print(f"\nSHC group {SHC_GROUP_ID} x-range (Å): {xmin:.3f} -> {xmax:.3f} (center ~ {xmean:.3f})")

    # ---- plots ----
    plt.figure()
    plt.plot(omega, Jtot, label="J_in + J_out (raw)")
    plt.xlabel(r"Frequency ($\omega / 2\pi$) (THz)")
    plt.ylabel("J_q(ω) (from shc.out)")
    plt.title(f"Spectral Heat Current (group {SHC_GROUP_ID}, x-range (Å): {xmin:.3f} -> {xmax:.3f})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"shc_Jsum_S{SMOOTH_W}.png", dpi=300)

    plt.figure()
    # plt.plot(omega, G_like_A, label="(J_in+J_out)/ΔT/A (raw)")
    plt.plot(omega, Gs, label=f"smoothed (w={SMOOTH_W})")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("G_like(ω) (native units) / Å²")
    plt.title("SHC normalized by ΔT and area (for comparisons)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"shc_Glike_S{SMOOTH_W}.png", dpi=300)

    plt.figure()
    plt.plot(omega, cum)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Cumulative ∫ G_like dω (native units)")
    plt.title("Cumulative spectral contribution")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"shc_Glike_cum_S{SMOOTH_W}.png", dpi=300)
    print("\nSaved: shc_Jsum.png, shc_Glike.png, shc_Glike_cum.png")

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            _log_fh.close()
        except Exception:
            pass