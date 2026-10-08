# Three-band Emery model with neural quantum states

This repository contains code for variational calculations on the three-band
Emery Hubbard model used in our study of cuprates and nickelates.

## Hamiltonian

The lattice has three orbitals per unit cell: a Cu $d_{x^2-y^2}$ orbital and
the two oxygen orbitals $p_x$ and $p_y$. The Hamiltonian implemented in
`src/observables.py` is

$$
\begin{aligned}
H ={}&
-t_{pd}\sum_{\langle i,j\rangle_{\mathrm{Cu-O}},\sigma}
  \left(c^\dagger_{i\sigma}c_{j\sigma}+\mathrm{h.c.}\right)
-t_{pp}\sum_{\langle i,j\rangle_{\mathrm{O-O}},\sigma}
  \left(c^\dagger_{i\sigma}c_{j\sigma}+\mathrm{h.c.}\right)\\
&+U_d\sum_{i\in\mathrm{Cu}} n_{i\uparrow}n_{i\downarrow}
+U_p\sum_{i\in\mathrm{O}} n_{i\uparrow}n_{i\downarrow}
+\Delta\sum_{i\in\mathrm{O},\sigma}n_{i\sigma}.
\end{aligned}
$$

Here $t_{pd}$ and $t_{pp}$ are the Cu–O and O–O hopping amplitudes,
$U_d$ and $U_p$ are the onsite repulsions on copper and oxygen, and
$\Delta$ is the oxygen onsite-energy offset relative to copper. The hopping
terms connect the corresponding nearest-neighbor pairs, with the Hermitian
conjugate included. Periodic boundary conditions can be selected independently
along the two lattice directions.

In `src/run_nqs.py`, the default Hamiltonian parameters are
$t_{pd}=1$, $t_{pp}=0.5$, $U_d=8$, $U_p=3$, and $\Delta=3.5$.
The script exposes $t_{pp}$, $U_d$, $U_p$, and $\Delta$ as
command-line options; $t_{pd}$ is currently fixed to 1.

## Reference

Hannah Lange, Julius F. A. Tirpitz, and Annabelle Bohrdt,
“Neural-quantum-state based downfolding of the three-band Emery model for
cuprates and nickelates,” arXiv:2609.38150 (2026).
[arXiv:2609.38150](https://arxiv.org/abs/2609.38150)
