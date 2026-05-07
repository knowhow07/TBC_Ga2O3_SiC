import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["DejaVu Serif"]

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# ============================================================
# Font size control
# ============================================================
font_size = 30
FS_LAYER   = 34   # layer titles
FS_NODE    = 36   # labels inside nodes
FS_SIDE    = font_size   # left/right labels
FS_EQ      = font_size   # equations
FS_BOX_T   = font_size   # parameter-box title
FS_BOX     = font_size   # parameter-box text
FS_CAPTION = font_size   # figure caption

# ============================================================
# Save options
# ============================================================
SAVE_FIG = True
OUT_PNG = "nep_nn_simple.png"
OUT_PDF = "nep_nn_simple.pdf"

# ============================================================
# Figure setup
# ============================================================
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(0, 13)
ax.set_ylim(0, 10)
ax.axis("off")

# ============================================================
# Layer positions
# ============================================================
x_in  = 2.0
x_hid = 6.5
x_out = 11

# input/output kept similar, hidden shifted upward a little
y_in  = [7.2, 5.4, 3.6, 1.8]
y_hid = [8.1, 6.3, 4.5, 2.7, 0.9]   # shifted upward
y_out = 4.5

# y_in  = [7.7, 5.9, 4.1, 2.3]
# y_hid = [8.0, 6.5, 5.0, 3.5, 2.0]   # shifted upward
# y_out = 4.5

r_in  = 0.8
r_hid = 0.8
r_out = 1.0

# ============================================================
# Better color set
# ============================================================
C_IN   = "#BFD3E6"   # soft blue
C_HID  = "#C7E9C0"   # soft green
C_OUT  = "#FDD49E"   # soft orange
C_EDGE = "#7A7A7A"   # gray lines

# ============================================================
# Connections
# ============================================================
for yi in y_in:
    for yh in y_hid:
        ax.plot([x_in + r_in, x_hid - r_hid], [yi, yh],
                color=C_EDGE, lw=2.0, alpha=0.75, zorder=1)

for yh in y_hid:
    ax.plot([x_hid + r_hid, x_out - r_out], [yh, y_out],
            color=C_EDGE, lw=2.0, alpha=0.8, zorder=1)
ax.annotate(
    "",
    xy=(12.4, y_out),
    xytext=(x_out + r_out, y_out)
    # arrowprops=dict(arrowstyle="->", lw=1.2, color=C_EDGE)
)

# ============================================================
# Nodes
# ============================================================
for y in y_in:
    ax.add_patch(Circle((x_in, y), r_in, facecolor=C_IN, edgecolor="black", lw=2.0  , zorder=3))

for y in y_hid:
    ax.add_patch(Circle((x_hid, y), r_hid, facecolor=C_HID, edgecolor="black", lw=2.0, zorder=3))

ax.add_patch(Circle((x_out, y_out), r_out, facecolor=C_OUT, edgecolor="black", lw=2.0, zorder=3))

# ============================================================
# Layer titles
# ============================================================
ax.text(x_in, 9.15, "Input Layer", ha="center", va="bottom",
        fontsize=FS_LAYER, fontweight="bold")
ax.text(x_hid, 9.15, "Hidden Layer", ha="center", va="bottom",
        fontsize=FS_LAYER, fontweight="bold")
ax.text(x_out, 9.15, "Output Layer", ha="center", va="bottom",
        fontsize=FS_LAYER, fontweight="bold")

# ============================================================
# Labels inside nodes
# ============================================================
in_labels = [r"$q_{i1}$", r"$q_{i2}$", r"$\cdots$", r"$q_{iN_{\mathrm{des}}}$"]
for y, lab in zip(y_in, in_labels):
    ax.text(x_in, y, lab, ha="center", va="center", fontsize=FS_NODE)

hid_labels = [r"$h_1$", r"$h_2$", r"$h_3$", r"$\cdots$", r"$h_m$"]
for y, lab in zip(y_hid, hid_labels):
    ax.text(x_hid, y, lab, ha="center", va="center", fontsize=FS_NODE)

ax.text(x_out, y_out, r"$U_i$", ha="center", va="center",
        fontsize=FS_NODE + 4, fontweight="bold")

# ============================================================
# Side labels
# ============================================================
# ax.text(1.0, 7.7, "radial", ha="left", va="center", fontsize=FS_SIDE)
# ax.text(1.0, 5.9, "angular", ha="left", va="center", fontsize=FS_SIDE)
# ax.text(1.0, 4.1, "type map", ha="left", va="center", fontsize=FS_SIDE)
# ax.text(1.0, 2.3, "other descriptors", ha="left", va="center", fontsize=FS_SIDE)

# ax.text(12.48, y_out, r"$U_i$", ha="left", va="center", fontsize=FS_SIDE)

# ============================================================
# Equations at bottom: spread out more to avoid overlap
# ============================================================
# eq_y = 0.5

# ax.text(
#     2.5, eq_y,
#     r"$\mathbf{q}_i=[q_{i1},\,q_{i2},\,\ldots,\,q_{iN_{\mathrm{des}}}]$",
#     ha="center", va="center", fontsize=FS_EQ
# )

# ax.text(
#     7.4, eq_y,
#     r"$h_{i\mu}=\tanh\!\left(\sum_{\nu} w^{(0)}_{\mu\nu} q_{i\nu}-b^{(0)}_{\mu}\right)$",
#     ha="center", va="center", fontsize=FS_EQ
# )

# ax.text(
#     11.5, eq_y,
#     r"$E_{\mathrm{total}}=\sum_i U_i$",
#     ha="center", va="center", fontsize=FS_EQ
# )

# ============================================================
# Parameter box
# ============================================================
# box_x, box_y = 12.95, 1.65
# box_w, box_h = 2.55, 7.2
# ax.add_patch(Rectangle((box_x, box_y), box_w, box_h, fill=False, lw=1.0, edgecolor="black"))

# ax.text(box_x + 0.18, box_y + box_h - 0.28, "NEP parameters",
#         ha="left", va="top", fontsize=FS_BOX_T, fontweight="bold")

# param_lines = [
#     r"type: Ga, O, Si, C, H",
#     r"generation: 500000",
#     r"batch: 1000",
#     r"$n_{\max}$: 10, 8",
#     r"basis: 12, 10",
#     r"neuron: 96",
#     r"population: 24",
#     r"cutoff: 9.0, 6.0 $\AA$",
#     r"$\lambda_1$: 0.05",
#     r"$\lambda_2$: 0.05",
#     r"$\lambda_e$: 1.0",
#     r"$\lambda_f$: 2.0",
#     r"$\lambda_v$: 0.1",
# ]

# yy = box_y + box_h - 1.02
# for line in param_lines:
#     ax.text(box_x + 0.18, yy, line, ha="left", va="top", fontsize=FS_BOX)
#     yy -= 0.47

# ============================================================
# Caption
# ============================================================
# ax.text(
#     7.9, -0.03,
#     "Schematic NEP architecture used in this work: descriptor input, one hidden layer with tanh activation, and atomic-energy output.",
#     ha="center", va="top", fontsize=FS_CAPTION
# )

plt.tight_layout()

if SAVE_FIG:
    plt.savefig(OUT_PNG, dpi=900, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")

plt.show()