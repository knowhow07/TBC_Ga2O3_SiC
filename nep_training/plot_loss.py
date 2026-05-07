#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif"]

mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 12

# === Settings ===
os.chdir('./output')
fname = 'loss.out'
out_png = 'loss.png'

# === Load data ===
loss = np.loadtxt(fname, ndmin=2)
nrow, ncol = loss.shape

# Rebuild the x-axis so it always grows by 100 per row, regardless of restarts
# (row 1 -> 100, row 2 -> 200, ...)
x = np.arange(1, nrow + 1) * 100

# Column map (we will plot whatever exists)
# col 0 in file is the original step, which we IGNORE now
labels = ['Total', 'L1-reg', 'L2-reg', 'E-train', 'F-train', 'V-train', 'E-test', 'F-test', 'V-test']
# We'll auto-select up to available columns beyond the first
plot_cols = list(range(1, min(ncol, len(labels) + 1)))

plt.figure(figsize=(7, 5))

for i in plot_cols:
    lab = labels[i - 1] if (i - 1) < len(labels) else f'col{i}'
    plt.semilogy(x, loss[:, i], label=lab)

plt.xlabel('Generation')          # now literal generations
plt.ylabel('Loss / RMSE')
plt.legend(loc='best')
plt.grid(True, which='both', ls='--', lw=0.4, alpha=0.7)
plt.tight_layout()
plt.savefig(out_png, dpi=300)
plt.savefig("loss.pdf", bbox_inches="tight")
plt.close()

print(f"✅ Saved loss plot as {out_png}")
