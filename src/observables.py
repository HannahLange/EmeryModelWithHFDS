from jax import numpy as jnp
import numpy as np

from quantax.operator import (
    annihilate_d,
    annihilate_u,
    create_d,
    create_u,
    number_d,
    number_u,
)
from quantax.global_defs import get_sites


def calculate_coherences(L1, L2, state, sampler, filename, ext=""):
    """Calculate and save spin-resolved coherence matrices and variances."""
    sites = get_sites()
    N = sites.Nsites
    samples = sampler.sweep()
    coherences = np.zeros((2, N, N))
    coherences_v = np.zeros((2, N, N))

    for i in range(N):
        for j in range(N):
            op_u = create_u(i) @ annihilate_u(j)
            val_u, var_u = op_u.expectation(state, samples, return_var=True)
            op_d = create_d(i) @ annihilate_d(j)
            val_d, var_d = op_d.expectation(state, samples, return_var=True)
            coherences[0, i, j] = val_u
            coherences[1, i, j] = val_d
            coherences_v[0, i, j] = var_u
            coherences_v[1, i, j] = var_d

    jnp.save(
        ext + "results/coherences_" + filename + ".npy",
        jnp.array([coherences, coherences_v]),
    )
    return coherences, coherences_v


def _hop(i, j):
    hop_up = create_u(i) @ annihilate_u(j) + create_u(j) @ annihilate_u(i)
    hop_down = create_d(i) @ annihilate_d(j) + create_d(j) @ annihilate_d(i)
    return hop_up + hop_down


def EmeryHubbard(lattice, Ud, Up, Delta, tpd=1.0, tpp=1.0, pbc=(True, True)):
    """
    Three-band (Emery) Hubbard Hamiltonian.

    Orbital layers are Cu d_{x^2-y^2}, O p_x, and O p_y, respectively.
    """
    sites = get_sites()
    N = sites.Nsites

    # Hopping cutoffs use squared distances in the physical lattice.
    tpd_dist2 = 0.5**2
    tpp_dist2 = 0.5
    H = 0

    coords = sites.coord

    def min_image_delta(dx, L):
        return (dx + 0.5 * L) % L - 0.5 * L

    for i in range(N):
        xi = coords[i].astype("float32")
        print(i, xi)
        is_Cu_i = xi[0] % 1 == 0 and xi[1] % 1 == 0

        for j in range(i + 1, N):
            xj = coords[j].astype("float32")
            is_Cu_j = xj[0] % 1 == 0 and xj[1] % 1 == 0

            if is_Cu_i and is_Cu_j:
                continue

            dx = xi[0] - xj[0]
            dy = xi[1] - xj[1]
            if pbc[0]:
                dx = min_image_delta(dx, lattice._shape[1])
            if pbc[1]:
                dy = min_image_delta(dy, lattice._shape[2])

            dist2 = dx * dx + dy * dy

            if (is_Cu_i != is_Cu_j) and abs(dist2 - tpd_dist2) < 1e-6:
                H += -tpd * _hop(i, j)
            elif (
                not is_Cu_i
                and not is_Cu_j
                and abs(dist2 - tpp_dist2) < 1e-6
            ):
                H += -tpp * _hop(i, j)

    for i in range(N):
        x = coords[i].astype("float32")
        if x[0] % 1 == 0 and x[1] % 1 == 0:
            H += Ud * number_u(i) @ number_d(i)
        else:
            H += Up * number_u(i) @ number_d(i)
            H += Delta * (number_u(i) + number_d(i))

    return H
