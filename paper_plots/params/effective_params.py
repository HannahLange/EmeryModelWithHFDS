#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduces full_params_map.pdf from precomputed effective single-band
parameters (see ../../downfolding_paper_plots.py for the downfolding).

Data files (in this folder):
    effective_params.npy : (Npoints, 11), one row per (L, Ud, Nh, Delta) run,
                           in plotting order. Columns:
        0  L         linear system size (Lx = Ly)
        1  Ud        Emery Ud / t_pd
        2  Nh        number of doped holes (negative: electron doping)
        3  Delta     charge-transfer energy / t_pd
        4  np/nd     ratio of p- and d-orbital densities
        5  t/t_pd    effective nearest-neighbor hopping
        6  U/t       effective on-site interaction
        7  t_n/t     effective density-assisted hopping
        8  W_t       Wannier overlap for t
        9  W_U       Wannier overlap for U
        10 W_tn      Wannier overlap for t_n
    n0_pd.npy            : (2,) [n^p, n^d] of the undoped 10x10 run at
                           Delta=3.5, Ud=6 (to place the PRB 108 reference point)
"""

import numpy as np
import matplotlib.pyplot as plt
import os


params = {
    "text.usetex": True,
    "font.family": "serif",
    "legend.fontsize": 12,
    "figure.figsize": (8, 10),
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2,
    "lines.markeredgewidth": 2,
    "lines.markersize": 6,
    "lines.marker": "o",
    "patch.edgecolor": "black",
}
plt.rcParams.update(params)

os.environ["PATH"] += os.pathsep + '/opt/local/bin'

here = os.path.dirname(os.path.abspath(__file__))

rows = np.load(os.path.join(here, "effective_params.npy"))
n0p, n0d = np.load(os.path.join(here, "n0_pd.npy"))

cm1 = plt.get_cmap('Blues')
cm2 = plt.get_cmap('Reds')
cm3 = plt.get_cmap('Greens')
fig, ax = plt.subplots(2, 3, figsize=(7, 4), sharex=True)
ms = {6: "white", 8: "lightgrey", 10: "dimgrey"}   # marker face per system size
mas = {6.: "o", 4.: "s"}                           # marker per Ud
cidx = 0.1+(6+np.abs(6))/(6*6*1/5*2)
ax[1,0].plot([], [], color=cm1(cidx), mfc="white", marker="o", label=r"$U_d=6.0t_{pd}$, $\delta=-1/5,\dots,1/5$", linewidth=0)
ax[1,0].plot([], [], color=cm1(cidx), mfc="white", marker="s", label=r"$U_d=4.0t_{pd}$, $\delta=1/8$", linewidth=0)

for L, Ud, Nh, Delta, n, t, U_t, tn_t, Wt, WU, Wtn in rows:
    L = int(L)
    m, ma = ms[L], mas[Ud]
    Nh_max = int(L*L*1/5//2*2)
    cidx = 0.1+(Nh+Nh_max)/(L*L*1/5*2)

    ax[1,0].plot(n, t, color=cm1(cidx), mfc=m, marker=ma)
    ax[1,1].plot(n, U_t, color=cm2(cidx), mfc=m, marker=ma)
    ax[1,2].plot(n, tn_t, color=cm3(cidx), mfc=m, marker=ma)

    ax[0,0].plot(n, Wt, color=cm1(cidx), mfc=m, marker=ma)
    ax[0,1].plot(n, WU, color=cm2(cidx), mfc=m, marker=ma)
    ax[0,2].plot(n, Wtn, color=cm3(cidx), mfc=m, marker=ma)

    # highlight cuprate / nickelate parameter sets at delta=1/8 (filled markers)
    if Nh == int(L*L*1/8//2*2) and ((Delta == 3.5 and Ud == 6.) or (Delta == 4.5 and Ud == 4.)):
        la = {4: r"nickelates ($\delta=0.125$)", 6: r"cuprates ($\delta=0.125$)"}[int(Ud)] if L == 6 else None
        ax[1,0].plot(n, t, color=cm1(cidx), marker=ma, label=la, linewidth=0)
        ax[1,1].plot(n, U_t, color=cm2(cidx), marker=ma)
        ax[1,2].plot(n, tn_t, color=cm3(cidx), marker=ma)

        ax[0,0].plot(n, Wt, color=cm1(cidx), marker=ma, label=la, linewidth=0)
        ax[0,1].plot(n, WU, color=cm2(cidx), marker=ma)
        ax[0,2].plot(n, Wtn, color=cm3(cidx), marker=ma)

ax[1,0].set_ylabel("$t/t_{pd}$")
ax[1,1].set_ylabel("$U/t$")
ax[1,2].set_ylabel("$t_n/t$")
ax[0,0].set_ylabel(r"$W_t$")
ax[0,1].set_ylabel(r"$W_U$")
ax[0,2].set_ylabel(r"$W_{t_n}$")

ax[0,0].set_ylim(-0.1, 0.1)
# reference point: PRB 108 (2023), delta=0.15
n_ref = (0.05021468327590775+n0p)/(0.0484496124031009+n0d)
ax[1,0].plot(n_ref, 0.27, marker="X", markersize=8, color="grey", label=r"PRB 108, 2023 ($\delta=0.15$)", linewidth=0)
ax[1,1].plot(n_ref, 12.6, marker="X", markersize=8, color="grey")
ax[1,2].plot(n_ref, 0.60, marker="X", markersize=8, color="grey")
ax[1,1].set_xlabel(r"$n^p/n^d$")
handles, labels = ax[1,0].get_legend_handles_labels()

fig.legend(handles, labels,
           loc="upper center",
           ncol=3,
           bbox_to_anchor=(0.5, 1.05))

plt.tight_layout(rect=[0, 0, 1, 0.92])  # leave room for legend
plt.savefig(os.path.join(here, "full_params_map.pdf"), bbox_inches='tight')
plt.show()
