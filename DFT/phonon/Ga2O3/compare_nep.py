#!/usr/bin/env python3
import matplotlib.pyplot as plt
import yaml
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif"]

# ----------------------------
# Global style
# ----------------------------
plt.rcParams.update({
    # "font.size": 16,          # global font size
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

# factor = 33.35641   # THz → cm^-1
factor = 1

# --- Define band.yaml paths ---
# DFT_YAML = "<PATH_TO_DATA>/md_jobs/dp_test/SiC/dft/fine_relax/2nd/displace/band.yaml"
# NEP_YAML = "<PATH_TO_DATA>/md_jobs/dp_test/SiC/nep/1202_all/band-pre_nep.yaml"


DFT_YAML = "./DFT_displace/band_nac.yaml"
NEP_YAML = "./band-pre.yaml"
yaml_files = {
    "DFT": DFT_YAML,
    "NEP": NEP_YAML
}

# --- Assign colors ---
colors = {
    "DFT": "#0073FF",   # soft blue
    "NEP": "#FA7F6F"    # soft red
}

# ----------------------------
# Read x-axis tick labels from YAML (band segments)
# Expected format:
# labels:
# - ['A','Γ']
# - ['Γ','L']
# ...
# ----------------------------
# Use the first YAML file to define the x-axis ticks/labels
with open(next(iter(yaml_files.values()))) as f:
    data0 = yaml.safe_load(f)

phonons0 = data0["phonon"]
distances0 = [p["distance"] for p in phonons0]

# Segment end positions in distance array
# (phonopy writes "segment_nqpoint" for band.yaml)
seg_nq = data0.get("segment_nqpoint", None)
if seg_nq is None:
    raise KeyError("band.yaml missing 'segment_nqpoint' needed to place band labels on x-axis")

# Build tick positions at each segment boundary
boundaries = [0]
for n in seg_nq:
    boundaries.append(boundaries[-1] + int(n))
# boundaries are indices in phonons list; convert to distances
tick_positions = [distances0[i] for i in boundaries[:-1]] + [distances0[boundaries[-1] - 1]]

# Build tick labels from labels field:
# labels: [[A, Γ], [Γ, L], [L, M], [M, Γ]]
seg_labels = data0.get("labels", None)
if seg_labels is None:
    raise KeyError("band.yaml missing 'labels' needed to label the x-axis")

# Convert segment labels to boundary labels: A, Γ, L, M, Γ
boundary_labels = [seg_labels[0][0]] + [pair[1] for pair in seg_labels]

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(7, 5))

for label, fname in yaml_files.items():
    with open(fname) as f:
        data = yaml.safe_load(f)

    phonons = data["phonon"]
    distances = [p["distance"] for p in phonons]

    nband = len(phonons[0]["band"])
    for i in range(nband):
        freqs = [p["band"][i]["frequency"] * factor for p in phonons]
        plt.plot(
            distances, freqs,
            lw=1.2,
            color=colors[label],
            label=label if i == 0 else ""
        )

# --- Formatting ---
plt.xlabel("Wave vector")
plt.ylabel("Frequency (THz)")
plt.title("Phonon Band Structure: DFT vs NEP")
plt.grid(alpha=0.3)

# X ticks as band boundary labels (A Γ L M Γ)
plt.xticks(tick_positions, boundary_labels)

# Optional: vertical lines at boundaries
for x in tick_positions:
    plt.axvline(x, lw=0.8, alpha=0.3, color="k")

plt.legend(loc="upper right", frameon=True, framealpha=1, facecolor="white")
plt.tight_layout()
plt.savefig("band_comparison_DFT_vs_NEP.png", dpi=300)
plt.savefig("band_comparison_DFT_vs_NEP.pdf", bbox_inches="tight")
plt.show()
