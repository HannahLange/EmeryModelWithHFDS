#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduces n_k.pdf, n_k_L10.pdf and n_k_cuts.pdf from precomputed momentum
distributions (see ../calc_nk.py for how n(k) is obtained from the coherences).

Data files (in this folder), last axis = orbital (0: d, 1: px, 2: py),
n(k) summed over spin and rotation-symmetrized:
    Nhs_L8.npy           : (7,)          hole numbers Nh of the 8x8 runs
    nk_L8_NQS.npy        : (7, 9, 9, 3)  NQS n(k) on 8x8, one entry per Nh
    nk_L8_MF.npy         : (7, 9, 9, 3)  mean-field (U=6) n(k) on 8x8
    nk_L10_NQS_Nh12.npy  : (11, 11, 3)   NQS n(k) on 10x10, Nh=12
"""

import numpy as np
import matplotlib.pyplot as plt
import os


params = {
    "text.usetex": True,
    "font.family": "serif",
    "legend.fontsize": 12,
    "figure.figsize": (6, 6),
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 3,
    "lines.markeredgewidth": 2,
    "lines.markersize": 8,
    "lines.marker": "o",
    "patch.edgecolor": "black",
}
plt.rcParams.update(params)


os.environ["PATH"] += os.pathsep + '/opt/local/bin'
cm = plt.get_cmap('tab20')

here = os.path.dirname(os.path.abspath(__file__))


def plot_nk(nk_grid, kx_vals, ky_vals, vmin, vmax, ax, title="", cm="coolwarm"):
    im = ax.imshow(
        nk_grid.T,                 # transpose so axes match (kx horizontal)
        origin="lower",
        extent=[
            kx_vals.min(), kx_vals.max(),
            ky_vals.min(), ky_vals.max()
        ],
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        cmap=cm,
    )
    ax.set_title(title)
    return im


def grad_nk(nk_grid, kx_vals, ky_vals):
    nk_grid = nk_grid.sum(axis=-1)
    dkx = kx_vals[1] - kx_vals[0]
    dky = ky_vals[1] - ky_vals[0]

    dnk_dkx = np.gradient(nk_grid, dkx, axis=0)
    dnk_dky = np.gradient(nk_grid, dky, axis=1)

    return np.sqrt(dnk_dkx**2 + dnk_dky**2)


def k_grid(L):
    return 2 * np.pi * np.arange(-L//2, L//2+1) / L


# ------------------------------------------------------------------
# load data
# ------------------------------------------------------------------

Nhs = [int(N) for N in np.load(os.path.join(here, "Nhs_L8.npy"))]
data = dict(zip(Nhs, np.load(os.path.join(here, "nk_L8_NQS.npy"))))
dataMF = dict(zip(Nhs, np.load(os.path.join(here, "nk_L8_MF.npy"))))
umf = 6.

Nhs_ = [12]
data_ = {12: np.load(os.path.join(here, "nk_L10_NQS_Nh12.npy"))}


# ------------------------------------------------------------------
# n_k_L10.pdf  (10x10, Nh=12, orbital-resolved)
# ------------------------------------------------------------------

kx_vals = ky_vals = k_grid(10)

fig, ax = plt.subplots(1, 4, figsize=(8.8, 2), sharex=True, sharey=True)
for Nh in Nhs_:
    nk_rot = data_[Nh]
    im1 = plot_nk(np.sum(nk_rot, axis=-1), kx_vals, ky_vals, vmin=0, vmax=2, ax=ax[0], cm="Reds")
    im2 = plot_nk(nk_rot[:,:,0], kx_vals, ky_vals, vmin=0, vmax=1.1, ax=ax[1], cm="Reds")
    im3 = plot_nk(nk_rot[:,:,2], kx_vals, ky_vals, vmin=0, vmax=1.1, ax=ax[2], cm="Reds")
    im4 = plot_nk(nk_rot[:,:,1], kx_vals, ky_vals, vmin=0, vmax=1.1, ax=ax[3], cm="Reds")
ax[0].set_title(r"$n(\mathbf{k})$", fontsize=14)
ax[1].set_title(r"$n^d(\mathbf{k})$", fontsize=14)
ax[2].set_title(r"$n^{p_x}(\mathbf{k})$", fontsize=14)
ax[3].set_title(r"$n^{p_y}(\mathbf{k})$", fontsize=14)
fig.colorbar(im1, ax=ax[0])
fig.colorbar(im2, ax=ax[1])
fig.colorbar(im3, ax=ax[2])
fig.colorbar(im4, ax=ax[3])
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(here, "n_k_L10.pdf"))


# ------------------------------------------------------------------
# n_k_cuts.pdf  (doping dependence at nodal / antinodal k, 8x8)
# ------------------------------------------------------------------

def cuts(orbs):
    d1, d2 = [], []
    for Nh in Nhs:
        nk_rot = data[Nh][:,:,orbs].sum(axis=-1) - data[0][:,:,orbs].sum(axis=-1)
        d1.append(1/4*(nk_rot[6,6]+nk_rot[4,4]+nk_rot[4,6]+nk_rot[6,4]))  # (pi/2, pi/2)
        d2.append(1/4*(nk_rot[4,8]+nk_rot[8,4]+nk_rot[4,0]+nk_rot[0,4]))  # (pi, 0)
    return np.array(d1), np.array(d2)

delta = np.array(Nhs)/64

plt.figure(figsize=(6, 3.5))
d1, d2 = cuts(slice(None))
plt.plot(delta, d1, label=r"$\mathbf{k}=(\pi/2,\pi/2)$ ", color=cm(4), mfc="white")
plt.plot(delta, d2, label=r"$\mathbf{k}=(\pi,0)$ ", color=cm(2), mfc="white")
plt.fill_between(delta, 0, d1, color=cm(4), alpha=0.7)
plt.fill_between(delta, 0, d2, color=cm(2), alpha=0.7)

# p-orbital contribution
d1, d2 = cuts(slice(1, None))
plt.fill_between(delta, 0, d1, color=cm(5), alpha=1)
plt.fill_between(delta[:4], 0, d2[:4], color=cm(3), alpha=1)
plt.legend()
plt.xlabel(r"doping $\delta$")
plt.ylabel(r"$n(\mathbf{k},\delta)-n(\mathbf{k},0)$")
plt.tight_layout()
plt.savefig(os.path.join(here, "n_k_cuts.pdf"))
plt.show()
