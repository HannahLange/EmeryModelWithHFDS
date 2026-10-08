import quantax as qtx
from jax import numpy as jnp
import numpy as np
import jax
from quantax.symmetry import SpinInverse, Identity, TransND, Translation
from quantax.operator import *
from observables import *
import equinox as eqx
import argparse
import time
import sys
import jax.random as jr
import gc

key = jr.PRNGKey(0)
qtx.set_default_dtype(jnp.float64)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-l", "--length", type=int, default=2, help="length of physical system"
)
parser.add_argument(
    "-w", "--width", type=int, default=4, help="width of physical system"
)
parser.add_argument(
    "-b",
    "--b",
    type=int,
    nargs=2,
    default=[1, 1],
    help="boundaries: 1 = pbc, 2 = obc",
)
parser.add_argument(
    "-Np", "--Np", type=int, default=8, help="Number of particles"
)
parser.add_argument(
    "-UMF",
    "--UMF",
    type=float,
    default=4.0,
    help="On-site repulsion strength for MF optimization",
)
parser.add_argument(
    "-DeltaMF",
    "--DeltaMF",
    type=float,
    default=3.0,
    help="Potential offset for MF optimization",
)
parser.add_argument(
    "-tppMF",
    "--tppMF",
    type=float,
    default=0.5,
    help="Hopping between orbitals",
)
parser.add_argument(
    "-Ud",
    "--Ud",
    type=float,
    default=8.0,
    help="On-site interaction for d sites",
)
parser.add_argument(
    "-Up", "--Up", type=float, default=3.0, help="On-site interaction for p sites"
)
parser.add_argument(
    "-tpp", "--tpp", type=float, default=0.5, help="Hopping between orbitals"
)
parser.add_argument(
    "-Delta", "--Delta", type=float, default=3.5, help="Potential offset"
)
parser.add_argument(
    "-pdd",
    "--pairing_dd",
    type=float,
    default=0.01,
    help="Strength of pairing term between d-d",
)
parser.add_argument(
    "-pdp",
    "--pairing_dp",
    type=float,
    default=0.01,
    help="Strength of pairing term between d-p",
)
parser.add_argument(
    "-c", "--c", type=float, default=0.1, help="Prefactor of (N-Ntarget)**2"
)
parser.add_argument(
    "-stepsMF",
    "--stepsMF",
    type=int,
    default=10000,
    help="Number of optimization steps for MF optimization",
)
parser.add_argument(
    "-steps", "--steps", type=int, default=1000, help="Number of optimization steps"
)
parser.add_argument(
    "-layers", "--layers", type=int, default=2, help="Number of network layers"
)
parser.add_argument(
    "-features", "--features", type=int, default=12, help="Number of features"
)
parser.add_argument(
    "-ks", "--kernelsize", type=int, default=3, help="kernelsize"
)
parser.add_argument(
    "-nhid", "--nhid", type=int, default=4, help="Number of hidden fermions"
)
parser.add_argument(
    "-nsamples", "--nsamples", type=int, default=1000, help="Number of samples"
)
parser.add_argument(
    "-rtol", "--rtol", type=float, default=1e-12, help="Tolerance for SR step"
)
parser.add_argument("-lr", "--lr", type=float, default=0.03, help="Learning rate")
parser.add_argument(
    "-loadMF",
    "--loadMF",
    type=int,
    default=1,
    help="if 1: loads MF, 0: doesnt load MF",
)
parser.add_argument(
    "-load",
    "--load",
    type=int,
    default=0,
    help="if 1: loads from previous runs, 0: runs the optimization",
)

args = parser.parse_args()
L1 = args.length
L2 = args.width
b1, b2 = args.b
Ntarget = args.Np

# Mean-field and neural quantum state optimization parameters.
uMF = args.UMF
deltaMF = args.DeltaMF
pairing_dd = args.pairing_dd
pairing_dp = args.pairing_dp
tppMF = args.tppMF
c = args.c
nstepsMF = args.stepsMF

ud = args.Ud
up = args.Up
delta = args.Delta
tpp = args.tpp
layers = args.layers
kernelsize = args.kernelsize
features = args.features
nhid = args.nhid
nsamples = args.nsamples
nsteps = args.steps
lr = args.lr
rtol = args.rtol
max_parallel = (10000,5000)
if L1 * L2 > 36:
    max_parallel = (5000,2000)
if L1 * L2 > 48:
    max_parallel = (2500,1000)

modelstring = "Det"
MFstring = "Det"
print("jax device: ", jax.devices())
print(
    "Running Optimization for Backflow"
    + modelstring
    + " and Mean-Field "
    + MFstring
    + " with paring_dd="
    + str(pairing_dd)
    + " and pairing_dp="
    + str(pairing_dp)
)

if args.loadMF == 1:
    filename = (
        f"Lx{L1}_Ly{L2}_b{b1}{b2}_Nt{Ntarget}_Ud{ud}_Up{up}_tpp{tpp}_Delta{delta}"
        f"_MF{uMF}_{deltaMF}_{pairing_dd}_{pairing_dp}_{c}_{nstepsMF}"
        f"_layers{layers}_features{features}_ks{kernelsize}_nhid{nhid}"
        f"_nsamples{nsamples}_nsteps{nsteps}_lr{lr}_rtol{rtol}"
    )
else:
    filename = (
        f"Lx{L1}_Ly{L2}_b{b1}{b2}_Nt{Ntarget}_Ud{ud}_Up{up}_tpp{tpp}_Delta{delta}"
        f"_layers{layers}_features{features}_ks{kernelsize}_nhid{nhid}"
        f"_nsamples{nsamples}_nsteps{nsteps}_lr{lr}_rtol{rtol}"
    )
filename += "_s"
load_filename = filename

lattice = qtx.sites.Lattice(
    [L1, L2],
    basis_vectors=[[1.0, 0.0], [0.0, 1.0]],
    site_offsets=[[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]],
    Nparticles=((Ntarget + 1) // 2, Ntarget // 2),
    boundary=(1, 1),
    particle_type=qtx.PARTICLE_TYPE(1),
    double_occ=True,
)

H = EmeryHubbard(
    lattice,
    Ud=ud,
    Up=up,
    Delta=delta,
    tpd=1.0,
    tpp=tpp,
    pbc=({1: True, 0: False}[b1], {1: True, 0: False}[b2]),
)
pg_symm = SpinInverse()
full_symm = pg_symm

run_optim = True
run_ED = False

if run_ED:
    E, wf = H.diagonalize()
    print(E)

net = qtx.model.ResConv(
    nblocks=layers,
    channels=features,
    kernel_size=kernelsize,
    final_activation=lambda x: x,
    trans_symm=Identity(),
)

MFfile = (
    f"MF_results/orbs_Lx{L1}_Ly{L2}_b{b1}{b2}_Nt{Ntarget}_U{uMF}_Delta{deltaMF}"
    f"_tpp{tppMF}_pairing{pairing_dd}_{pairing_dp}_c{c}_nsteps{nstepsMF}.npy"
)
if args.loadMF == 0:
    F = jr.normal(key, (lattice.Nfmodes, lattice.Nfmodes), dtype=jnp.float64) * 1
    MFmodel = qtx.model.GeneralPf(F=F, dtype=jnp.float64)
    MFstate = qtx.state.GeneralPfState(MFmodel)
    for i in range(1000):
        step = MFstate.get_step(H)
        MFstate.update(step * 0.1)
    F = MFstate.model.F
    np.save(MFfile, F)

energy_data = []
variance_data = []

U = jnp.load(MFfile)
MFmodel = qtx.model.GeneralDet(U=U, dtype=jnp.float64)
MFstate = qtx.state.GeneralDetState(MFmodel, max_parallel=max_parallel)
sampler = qtx.sampler.ParticleHop(MFstate, nsamples, thermal_steps=100)
samples = sampler.sweep()
print(
    "MFmodel (Det) energy on samples:",
    H.expectation(MFstate, samples),
)  # This differs from the MF optimization of exact_grad due to other Hamiltonian parameters.

# Backflow uses twice nhid because spinful models internally halve d.
model = qtx.model.DetBackflow(net=net, d=2 * nhid, U0=U)
state = qtx.state.Variational(model, max_parallel=max_parallel, symm=full_symm)
sampler = qtx.sampler.ParticleHop(state, nsamples, thermal_steps=100)
samples = sampler.sweep()

MFmodel = qtx.model.GeneralPf(F=F, dtype=jnp.float64)
MFstate = qtx.state.GeneralPfState(MFmodel, max_parallel=max_parallel)
sampler = qtx.sampler.ParticleHop(MFstate, nsamples, thermal_steps=100)
samples = sampler.sweep()
print("MFmodel (Pf) energy on samples:", H.expectation(MFstate, samples))

model = qtx.model.PfBackflow(net=net, d=2 * nhid, U0=U0, J0=J0)
state = qtx.state.Variational(model, max_parallel=max_parallel, symm=full_symm)
sampler = qtx.sampler.ParticleHop(state, nsamples, thermal_steps=100)
samples = sampler.sweep()

if args.load == 1:
    load_filename = (
        filename.split(f"nsteps{nsteps}")[0]
        + f"nsteps{2000}"
        + filename.split(f"nsteps{nsteps}")[1]
    )
    print("load state:", load_filename)
    state = qtx.state.Variational(
        model,
        max_parallel=max_parallel,
        param_file="states/" + load_filename,
        symm=full_symm,
    )
    sampler = qtx.sampler.ParticleHop(state, nsamples, thermal_steps=100)
    energy_data = list(np.load("results/energy_" + load_filename + ".npy"))
    variance_data = list(
        np.load("results/energy_variance_" + load_filename + ".npy")
    )
    nsteps = nsteps - len(energy_data)
    if nsteps < 10:
        run_optim = False

for i in range(10):
    samples = sampler.sweep()
    print(i)

params = eqx.filter(model, eqx.is_inexact_array)
total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"Total parameters: {total_params}")

tdvp = qtx.optimizer.SR(state,H,solver=qtx.optimizer.auto_pinv_eig(rtol=rtol))

print("model energy on samples:", H.expectation(state, samples))

if run_optim:
    lr_decay = (lr - lr / 2) / nsteps
    E, VarE = H.expectation(state, samples, return_var=True)
    print("initial", E, VarE)
    energy_data.append(E)
    variance_data.append(VarE)
    for i in range(nsteps):
        start_time = time.time()
        samples = sampler.sweep()
        step = tdvp.get_step(samples)
        state.update(step * lr)
        E = tdvp.energy
        energy_data.append(E)
        VarE = tdvp.VarE
        variance_data.append(VarE)
        end_time = time.time()
        lr -= lr_decay
        del step
        gc.collect()

        print(i, "/", nsteps, E, tdvp.VarE, "(", end_time - start_time, "s)")

        if jnp.isnan(E):
            break
        if i % 10 == 0 or i == nsteps - 1:
            np.save("results/energy_" + filename + ".npy", energy_data)
            np.save("results/energy_variance_" + filename + ".npy", variance_data)
            state.save("states/" + filename)

jnp.save(
    "results/samples_" + filename + ".npy",
    jnp.asarray(samples.spins).copy(),
)
