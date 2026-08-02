# -*- coding: utf-8 -*-
"""
plot_ftp_matrix.py
====================
Plots a single FTP score matrix (BB, k-NN, or KDE) from its CSV file,
in the same style as FTP_score_matrices.png — fully editable.

Usage:
    python3 plot_ftp_matrix.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================
#  STYLE — edit freely
# ================================================================
FONT_FAMILY  = "DejaVu Sans"   # any installed font, e.g. "Arial", "Times New Roman"
FONT_SIZE    = 12              # base font size (titles/labels scale from this)
ANNOT_SIZE   = 10               # font size of the numbers inside cells
ANNOT_COLOR  = "black"         # colour of the numbers inside cells
BG_COLOR     = "white"         # figure/axes background ("black" for dark mode)
TEXT_COLOR   = "black"         # titles, axis labels, tick labels
CMAP         = "RdYlGn"        # colormap: "RdYlGn", "viridis", "YlOrRd", etc.
DIAG_COLOR   = "blue"          # colour of the diagonal highlight box
DPI          = 300

RES_ORDER = ["20x20", "80x80", "300x300", "60x400", "80x600", "100x1000"]

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size"  : FONT_SIZE,
})


# ================================================================
#  PLOT FUNCTION
# ================================================================
def plot_ftp_matrix(csv_path, method_name, out_path):
    
    csv_path    : "D:/UIB_onedrive/OneDrive - Universitat de les Illes Balears/CFD_Studies/FTP_Analysis/FTP_Results/FTP_matrix_BB.csv"
    method_name : "BB"
    out_path    : "D:/UIB_onedrive/OneDrive - Universitat de les Illes Balears/CFD_Studies/FTP_Analysis/FTP_Results"
    
    df = pd.read_csv(csv_path, index_col=0)
    df = df.reindex(index=RES_ORDER, columns=RES_ORDER)   # enforce consistent order
    mat = df.values.astype(float)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    im = ax.imshow(mat, cmap=CMAP, vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(RES_ORDER)))
    ax.set_xticklabels(RES_ORDER, rotation=35, ha='right', color=TEXT_COLOR)
    ax.set_yticks(range(len(RES_ORDER)))
    ax.set_yticklabels(RES_ORDER, color=TEXT_COLOR)
    ax.set_xlabel("CFD Mesh Resolution", color=TEXT_COLOR)
    ax.set_ylabel("MCRT Grid Resolution", color=TEXT_COLOR)
    ax.set_title(f"FTP Score — {method_name}", color=TEXT_COLOR,
                 fontsize=FONT_SIZE + 2, fontweight='bold', pad=10)
    ax.tick_params(colors=TEXT_COLOR)

    for i in range(len(RES_ORDER)):
        for j in range(len(RES_ORDER)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha='center', va='center',
                        fontsize=ANNOT_SIZE, color=ANNOT_COLOR, fontweight='bold')

    for k in range(len(RES_ORDER)):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                     fill=False, edgecolor=DIAG_COLOR, linewidth=2))

    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("FTP score", color=TEXT_COLOR)
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close()
    print(f"Saved: {out_path}")


# ================================================================
#  RUN — edit paths, call once per method
# ================================================================
if __name__ == "__main__":
    RESULTS_DIR = "D:/UIB_onedrive/OneDrive - Universitat de les Illes Balears/CFD_Studies/FTP_Analysis/FTP_Results"

    plot_ftp_matrix(f"{RESULTS_DIR}/FTP_matrix_BB.csv",
                     "BB",  f"{RESULTS_DIR}/ftp_matrix_BB.png")
    plot_ftp_matrix(f"{RESULTS_DIR}/FTP_matrix_kNN.csv",
                     "kNN", f"{RESULTS_DIR}/ftp_matrix_kNN.png")
    plot_ftp_matrix(f"{RESULTS_DIR}/FTP_matrix_KDE.csv",
                     "KDE", f"{RESULTS_DIR}/ftp_matrix_KDE.png")
