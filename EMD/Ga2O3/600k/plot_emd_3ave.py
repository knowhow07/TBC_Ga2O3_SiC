#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ----------------------------
# Collect all hac.out files
# ----------------------------
folders = sorted(glob.glob("job_*"))
hac_files = [f"{d}/hac.out" for d in folders if os.path.exists(f"{d}/hac.out")]

if len(hac_files) == 0:
    raise SystemExit("No hac.out files found in job_* folders!")

print(f"Found {len(hac_files)} hac.out files.")

kx_list = []
ky_list = []
kz_list = []
kiso_list = []
t_ref = None

# ----------------------------
# Process each hac.out
# ----------------------------
for hac in hac_files:
    data = np.loadtxt(hac)
    t = data[:, 0]

    # Save the reference time array once
    if t_ref is None:
        t_ref = t
    else:
        if len(t) != len(t_ref):
            raise ValueError(f"Time array length mismatch in {hac}")

    # Columns:
    # col6 = k_x_in, col7 = k_x_out, col8 = k_y_in, col9 = k_y_out, col10 = k_z_tot
    k_x_in  = data[:, 6]
    k_x_out = data[:, 7]
    k_y_in  = data[:, 8]
    k_y_out = data[:, 9]
    k_z_tot = data[:, 10]

    k_x_tot = k_x_in + k_x_out
    k_y_tot = k_y_in + k_y_out
    k_iso   = (k_x_tot + k_y_tot + k_z_tot) / 3.0

    kx_list.append(k_x_tot)
    ky_list.append(k_y_tot)
    kz_list.append(k_z_tot)
    kiso_list.append(k_iso)

# Convert lists → arrays
kx_arr   = np.vstack(kx_list)
ky_arr   = np.vstack(ky_list)
kz_arr   = np.vstack(kz_list)
kiso_arr = np.vstack(kiso_list)

# Averages across jobs
kx_avg   = kx_arr.mean(axis=0)
ky_avg   = ky_arr.mean(axis=0)
kz_avg   = kz_arr.mean(axis=0)
kiso_avg = kiso_arr.mean(axis=0)

# ----------------------------
# Save averaged kappa vs time
# ----------------------------
out_data = np.column_stack([t_ref, kx_avg, ky_avg, kz_avg, kiso_avg])

header = (
    "t(ps)   kx_avg(W/mK)   ky_avg(W/mK)   kz_avg(W/mK)   kiso_avg(W/mK)\n"
    "Averaged over {} runs".format(len(kiso_arr))
)

np.savetxt("kappa_avg_vs_time.txt", out_data, header=header, fmt="%.6f")

print("Saved averaged kappa curves to: kappa_avg_vs_time.txt")

# ----------------------------
# Also save ONLY the final values
# ----------------------------
final_values = np.array([
    kx_avg[-1],
    ky_avg[-1],
    kz_avg[-1],
    kiso_avg[-1]
])

np.savetxt(
    "kappa_final_values.txt",
    final_values,
    header="kx_avg_final   ky_avg_final   kz_avg_final   kiso_avg_final\n",
    fmt="%.6f"
)

print("Saved final averaged kappa values to: kappa_final_values.txt")


# ----------------------------
# Compute final averaged values + find the max
# ----------------------------
final_values = {
    "kx_tot (avg)": kx_avg[-1],
    "ky_tot (avg)": ky_avg[-1],
    "kz_tot (avg)": kz_avg[-1],
    "k_iso (avg)":  kiso_avg[-1],
}

max_label = max(final_values, key=final_values.get)
max_value = final_values[max_label]

# ----------------------------
# Plot
# ----------------------------
plt.rcParams['font.size'] = 16
plt.figure(figsize=(10, 6))

# Light lines: each job's k_iso
for i in range(kiso_arr.shape[0]):
    plt.plot(t_ref, kiso_arr[i], color="gray", alpha=0.25, linewidth=1)

# Dark averaged lines
plt.plot(t_ref, kx_avg,   label="kx_tot (avg)")
plt.plot(t_ref, ky_avg,   label="ky_tot (avg)")
plt.plot(t_ref, kz_avg,   label="kz_tot (avg)")
plt.plot(t_ref, kiso_avg, label="k_iso (avg)", linewidth=3)

plt.xlabel("Correlation time t (ps)")
plt.ylabel("Running κ(t) [W/mK]")
plt.legend()

# ----------------------------
# Add single annotation at top center
# ----------------------------
plt.text(
    0.5, 1.02,
    f"max_final κ = {max_label}: {max_value:.2f} W/mK",
    transform=plt.gca().transAxes,
    ha="center",
    va="bottom",
    fontsize=16,
)

plt.tight_layout()
plt.savefig("emd_running_kappa_avg.png", dpi=300)

print("Saved figure: emd_running_kappa_avg.png")
