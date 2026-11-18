"""
resource_reduction.py
=====================
Library for the *Intermediate* challenge ("do more with less"):
reduce the quantum resources needed to reach chemical accuracy for LiH / BeH2.

Everything the notebook and the experiment scripts need lives here:

  * problem construction (freeze core, drop orbitals)        -> build_problem
  * fermion-to-qubit mappers (JW / Parity / BK / tapered)    -> make_mapper
  * ansatz factory (UCCSD, PUCCD, SUCCD, hardware-efficient)  -> make_ansatz
  * exact references                                          -> exact_energy_sector
  * fast noiseless evaluator (Aer, batched gradients)         -> ExactEvaluator
  * resource metrics (qubits, params, CNOTs, depth, groups)   -> circuit_metrics, operator_metrics
  * measurement cost (variance-optimal shot counts)           -> shots_required
  * shot-noise simulator for the energy                       -> ShotEnergy
  * VQE drivers (exact/SLSQP, shot-based/SPSA, custom ADAPT)  -> run_vqe_exact, run_vqe_shots, run_adapt

Energy conventions
------------------
All energies are *total* energies in Hartree:
    E_total = <H_qubit> + shift,   shift = sum(problem.hamiltonian.constants.values())
(nuclear repulsion + the frozen-core energy).  Errors are always measured against the
**full, unreduced** exact energy of the same molecule/geometry, so any error introduced
by freezing / dropping orbitals is charged against the 1.6 mHa budget.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import (
    BravyiKitaevMapper,
    JordanWignerMapper,
    ParityMapper,
)
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.second_q.circuit.library import PUCCD, SUCCD, UCCSD, HartreeFock
from qiskit.circuit.library import EvolvedOperatorAnsatz

warnings.filterwarnings("ignore")

CHEMICAL_ACCURACY = 1.6e-3  # Hartree

# --------------------------------------------------------------------------------------
# Molecules.  `drop` lists spatial-orbital indices (HF energy ordering, *before* freezing)
# that are removed from the active space.
# --------------------------------------------------------------------------------------
MOLECULES = {
    "LiH": dict(
        atom=lambda d: f"Li 0 0 0; H 0 0 {d}",
        eq=1.595,
        drop=[3, 4],  # 2px, 2py (pi-type) orbitals
    ),
    "BeH2": dict(
        atom=lambda d: f"Be 0 0 0; H 0 0 {d}; H 0 0 {-d}",
        eq=1.33,
        drop=[3, 4],  # pi_u pair
    ),
}


# --------------------------------------------------------------------------------------
# 1. Problem construction
# --------------------------------------------------------------------------------------
_PROBLEM_CACHE: dict = {}


def build_problem(molecule="LiH", distance=None, freeze_core=False, drop=None, basis="sto3g"):
    """Electronic-structure problem, optionally with frozen core and removed orbitals."""
    spec = MOLECULES[molecule]
    d = spec["eq"] if distance is None else distance
    key = (molecule, round(float(d), 6), basis)
    if key not in _PROBLEM_CACHE:
        driver = PySCFDriver(atom=spec["atom"](d), basis=basis, charge=0, spin=0)
        _PROBLEM_CACHE[key] = driver.run()
    problem = _PROBLEM_CACHE[key]
    drop = list(drop) if drop else []
    if freeze_core or drop:
        tr = FreezeCoreTransformer(freeze_core=freeze_core, remove_orbitals=drop or None)
        problem = tr.transform(problem)
    return problem


def energy_shift(problem) -> float:
    """Constant to add to <H_qubit> to get the total molecular energy."""
    return float(sum(problem.hamiltonian.constants.values()))


# --------------------------------------------------------------------------------------
# 2. Mappers
# --------------------------------------------------------------------------------------
MAPPER_KINDS = ["jw", "parity", "bk", "jw+taper", "parity+taper"]


def make_mapper(kind: str, problem):
    if kind == "jw":
        return JordanWignerMapper()
    if kind == "parity":  # two-qubit reduction (needs the particle numbers)
        return ParityMapper(num_particles=problem.num_particles)
    if kind == "bk":
        return BravyiKitaevMapper()
    if kind == "jw+taper":  # Z2 symmetry tapering
        return problem.get_tapered_mapper(JordanWignerMapper())
    if kind == "parity+taper":
        return problem.get_tapered_mapper(ParityMapper(num_particles=problem.num_particles))
    raise ValueError(kind)


# --------------------------------------------------------------------------------------
# 3. Ansatz factory
# --------------------------------------------------------------------------------------
ANSATZ_KINDS = ["uccsd", "succd", "puccd", "hwe"]


def make_ansatz(kind, problem, mapper, hwe_reps=2):
    n_orb, n_part = problem.num_spatial_orbitals, problem.num_particles
    hf = HartreeFock(n_orb, n_part, mapper)
    if kind == "uccsd":
        return UCCSD(n_orb, n_part, mapper, initial_state=hf)
    if kind == "succd":
        return SUCCD(n_orb, n_part, mapper, initial_state=hf)
    if kind == "puccd":
        return PUCCD(n_orb, n_part, mapper, initial_state=hf)
    if kind == "hwe":  # hardware-efficient layer stack on top of the HF bit-string
        n = hf.num_qubits
        qc = QuantumCircuit(n)
        qc.compose(hf, inplace=True)
        qc.compose(EfficientSU2(n, reps=hwe_reps, entanglement="linear"), inplace=True)
        return qc
    raise ValueError(kind)


# --------------------------------------------------------------------------------------
# A fully-specified experiment configuration
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    name: str
    molecule: str = "LiH"
    distance: float | None = None
    freeze_core: bool = False
    drop: bool = False  # drop the molecule's `drop` orbitals (active-space reduction)
    mapper: str = "jw"
    ansatz: str = "uccsd"
    hwe_reps: int = 2


@dataclass
class Setup:
    cfg: Config
    problem: object
    mapper: object
    qubit_op: SparsePauliOp
    ansatz: QuantumCircuit
    shift: float
    n_qubits: int = 0
    n_params: int = 0


def build_setup(cfg: Config) -> Setup:
    spec = MOLECULES[cfg.molecule]
    problem = build_problem(
        cfg.molecule, cfg.distance, cfg.freeze_core, spec["drop"] if cfg.drop else None
    )
    mapper = make_mapper(cfg.mapper, problem)
    qubit_op = mapper.map(problem.hamiltonian.second_q_op()).simplify()
    ansatz = make_ansatz(cfg.ansatz, problem, mapper, cfg.hwe_reps)
    return Setup(
        cfg, problem, mapper, qubit_op, ansatz, energy_shift(problem),
        n_qubits=qubit_op.num_qubits, n_params=ansatz.num_parameters,
    )


# --------------------------------------------------------------------------------------
# 4. Exact references (Jordan-Wigner, restricted to the (N_alpha, N_beta) sector)
# --------------------------------------------------------------------------------------
def exact_energy_sector(problem) -> float:
    """Exact ground-state *total* energy of `problem` in its (N_alpha, N_beta) sector.

    Independent of whatever mapper/ansatz is used later, so it serves as the reference
    for the same (possibly reduced) active space.
    """
    n = problem.num_spatial_orbitals
    na, nb = problem.num_particles
    op = JordanWignerMapper().map(problem.hamiltonian.second_q_op())
    H = op.to_matrix(sparse=True).tocsr()
    idx = np.arange(2 ** (2 * n))
    pop = np.array([bin(i).count("1") for i in idx])
    alpha = np.array([bin(i & (2**n - 1)).count("1") for i in idx])
    beta = pop - alpha
    sel = np.where((alpha == na) & (beta == nb))[0]
    Hs = H[sel][:, sel].toarray()
    return float(np.linalg.eigvalsh(Hs)[0].real + energy_shift(problem))


_FULL_EXACT_CACHE: dict = {}


def full_exact_energy(molecule, distance=None) -> float:
    """Exact energy of the FULL (all-electron, all-orbital) problem: the honest reference."""
    spec = MOLECULES[molecule]
    d = spec["eq"] if distance is None else distance
    key = (molecule, round(float(d), 6))
    if key not in _FULL_EXACT_CACHE:
        _FULL_EXACT_CACHE[key] = exact_energy_sector(build_problem(molecule, d))
    return _FULL_EXACT_CACHE[key]


# --------------------------------------------------------------------------------------
# 5. Fast noiseless evaluator (Aer statevector, batched parameter binds)
# --------------------------------------------------------------------------------------
_BASIS = ["rz", "rx", "ry", "h", "cx", "x", "s", "sdg"]


class ExactEvaluator:
    """E(theta) = <psi(theta)|H|psi(theta)> from an exact statevector simulation.

    The ansatz is decomposed once; each call only rebinds parameters.  `energies`
    evaluates many parameter sets in one Aer job (used for finite-difference gradients).
    """

    def __init__(self, ansatz: QuantumCircuit, qubit_op: SparsePauliOp):
        self.params = list(ansatz.parameters)
        qc = transpile(ansatz, basis_gates=_BASIS, optimization_level=0)
        qc.save_statevector()
        self.qc = qc
        self.H = qubit_op.to_matrix(sparse=True).tocsr()
        self.sim = AerSimulator(method="statevector", max_parallel_threads=0,
                                max_parallel_experiments=0)
        self.n_evals = 0

    def _run(self, thetas: np.ndarray) -> np.ndarray:
        thetas = np.atleast_2d(thetas)
        if len(self.params) == 0:
            res = self.sim.run(self.qc).result()
            return np.asarray(res.get_statevector(0))[None, :]
        binds = {p: [float(v) for v in thetas[:, i]] for i, p in enumerate(self.params)}
        res = self.sim.run(self.qc, parameter_binds=[binds]).result()
        return np.stack([np.asarray(res.get_statevector(i)) for i in range(thetas.shape[0])])

    def states(self, thetas):
        return self._run(np.asarray(thetas, dtype=float))

    def energies(self, thetas) -> np.ndarray:
        psi = self._run(np.asarray(thetas, dtype=float))
        self.n_evals += psi.shape[0]
        Hpsi = (self.H @ psi.T).T
        return np.real(np.sum(psi.conj() * Hpsi, axis=1))

    def energy(self, theta) -> float:
        return float(self.energies(np.asarray(theta, dtype=float)[None, :])[0])

    def gradient(self, theta, h=1e-5, central=True):
        theta = np.asarray(theta, dtype=float)
        P = len(theta)
        eye = np.eye(P) * h
        if central:
            e = self.energies(np.vstack([theta + eye, theta - eye]))
            return (e[:P] - e[P:]) / (2 * h)
        e = self.energies(np.vstack([theta[None, :], theta + eye]))
        return (e[1:] - e[0]) / h


# --------------------------------------------------------------------------------------
# 6. Resource metrics
# --------------------------------------------------------------------------------------
def circuit_metrics(ansatz: QuantumCircuit, seed=7, optimization_level=3) -> dict:
    """Two-qubit gate count / depth after transpiling to {cx, rz, sx, x}.

    Two targets are reported: all-to-all connectivity and a linear chain (a deliberately
    conservative proxy for a real heavy-hex device).
    """
    out = {}
    basis = ["cx", "rz", "sx", "x"]
    n = ansatz.num_qubits
    for label, cm in (("a2a", None), ("line", CouplingMap.from_line(n) if n > 1 else None)):
        qc = transpile(ansatz, basis_gates=basis, coupling_map=cm,
                       optimization_level=optimization_level, seed_transpiler=seed)
        ops = qc.count_ops()
        out[f"cx_{label}"] = int(ops.get("cx", 0))
        out[f"depth_{label}"] = int(qc.depth())
    return out


def operator_metrics(qubit_op: SparsePauliOp) -> dict:
    nonid = [i for i, p in enumerate(qubit_op.paulis) if p.x.any() or p.z.any()]
    return dict(
        n_pauli=len(nonid),
        n_groups_qwc=len(qubit_op.group_commuting(qubit_wise=True)),
        n_groups_gc=len(qubit_op.group_commuting(qubit_wise=False)),
        l1_norm=float(np.sum(np.abs(qubit_op.coeffs[nonid]))),
    )


# --------------------------------------------------------------------------------------
# 7. Measurement cost / shot-noise simulation
# --------------------------------------------------------------------------------------
def _pauli_masks(op: SparsePauliOp):
    """Per-term support mask and per-qubit measurement basis (0=I,1=X,2=Y,3=Z)."""
    n = op.num_qubits
    xs, zs = op.paulis.x, op.paulis.z
    basis = np.zeros((len(op), n), dtype=int)
    basis[(xs) & (~zs)] = 1
    basis[(xs) & (zs)] = 2
    basis[(~xs) & (zs)] = 3
    mask = (basis != 0) @ (1 << np.arange(n))
    return mask, basis


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDG = np.diag([1, -1j])
_ROT = {1: _H, 2: _H @ _SDG}  # X -> H ; Y -> H Sdg ; Z -> identity


def _apply_1q(psi, U, q, n):
    t = psi.reshape([2] * n)
    ax = n - 1 - q  # qubit q <-> axis n-1-q for little-endian flattening
    t = np.moveaxis(np.tensordot(U, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


class MeasurementPlan:
    """Qubit-wise-commuting grouping of H together with everything needed to sample it."""

    def __init__(self, qubit_op: SparsePauliOp, grouping="qwc"):
        self.n = qubit_op.num_qubits
        self.dim = 2**self.n
        self.grouping = grouping
        coeffs = np.real(qubit_op.coeffs)
        paulis = qubit_op.paulis
        ident = ~(paulis.x.any(axis=1) | paulis.z.any(axis=1))
        self.const = float(np.sum(coeffs[ident]))
        rest = SparsePauliOp(paulis[~ident], coeffs[~ident])
        if grouping == "qwc":
            groups = rest.group_commuting(qubit_wise=True)
        elif grouping == "none":
            groups = [SparsePauliOp(rest.paulis[i], rest.coeffs[i:i + 1]) for i in range(len(rest))]
        else:
            raise ValueError(grouping)
        idx = np.arange(self.dim)
        parity = np.zeros(self.dim, dtype=np.int8)  # parity[i] = popcount(i) mod 2
        for b in range(self.n):
            parity ^= ((idx >> b) & 1).astype(np.int8)
        self.groups = []
        for g in groups:
            mask, basis = _pauli_masks(g)
            # for each qubit, the (unique) non-identity basis over the group
            qb = {}
            for q in range(self.n):
                nz = basis[:, q][basis[:, q] != 0]
                if len(nz):
                    assert len(set(nz)) == 1, "group is not qubit-wise commuting"
                    qb[q] = int(nz[0])
            # sign matrix S[j, idx] = (-1)^popcount(idx & mask_j)
            S = np.array([1 - 2 * parity[idx & int(m)] for m in mask], dtype=float)
            c = np.real(g.coeffs)
            self.groups.append(dict(rot=qb, S=S, c=c, l1=float(np.abs(c).sum())))
        self.n_groups = len(self.groups)

    def probs(self, psi, g):
        for q, b in g["rot"].items():
            if b in (1, 2):
                psi = _apply_1q(psi, _ROT[b], q, self.n)
        p = np.abs(psi) ** 2
        return p / p.sum()

    def group_stats(self, psi):
        """Exact per-group mean and variance of the measured group operator."""
        out = []
        for g in self.groups:
            p = self.probs(psi, g)
            val = g["c"] @ g["S"]  # operator value on every outcome
            mean = float(p @ val)
            var = float(p @ val**2 - mean**2)
            out.append((mean, max(var, 0.0)))
        return out

    def shots_required(self, psi, eps=CHEMICAL_ACCURACY):
        """Total shots so the standard error of E is <= eps, with variance-optimal allocation."""
        sd = np.sqrt([v for _, v in self.group_stats(psi)])
        return float(np.sum(sd) ** 2 / eps**2)

    def sample_energy(self, psi, total_shots, rng):
        """Simulate measuring H with `total_shots`, allocated proportional to group L1 norms."""
        l1 = np.array([g["l1"] for g in self.groups])
        alloc = np.maximum(1, np.round(total_shots * l1 / l1.sum())).astype(int)
        e = self.const
        for g, ns in zip(self.groups, alloc):
            p = self.probs(psi, g)
            counts = rng.multinomial(ns, p)
            ev = g["S"] @ (counts / ns)
            e += float(g["c"] @ ev)
        return e, int(alloc.sum())


def shots_required(setup: Setup, theta, eps=CHEMICAL_ACCURACY, ev: ExactEvaluator | None = None):
    """Shots to hit `eps` standard error at the state prepared by `theta` (grouped & ungrouped)."""
    ev = ev or ExactEvaluator(setup.ansatz, setup.qubit_op)
    psi = ev.states(np.asarray(theta)[None, :])[0]
    res = {}
    for grouping in ("none", "qwc"):
        plan = MeasurementPlan(setup.qubit_op, grouping)
        res[grouping] = plan.shots_required(psi, eps)
    res["l1_bound"] = (operator_metrics(setup.qubit_op)["l1_norm"] / eps) ** 2
    return res


# --------------------------------------------------------------------------------------
# 8. VQE drivers
# --------------------------------------------------------------------------------------
def run_vqe_exact(setup: Setup, x0=None, maxiter=300, central=None, seed=0, tol=1e-10,
                  ev: ExactEvaluator | None = None, verbose=False):
    """Noiseless VQE with SLSQP and batched finite-difference gradients."""
    ev = ev or ExactEvaluator(setup.ansatz, setup.qubit_op)
    P = setup.n_params
    x0 = np.zeros(P) if x0 is None else np.asarray(x0, dtype=float)
    central = (P <= 40) if central is None else central
    hist = []

    def f(x):
        e = ev.energy(x) + setup.shift
        hist.append(e)
        return e

    def g(x):
        return ev.gradient(x, central=central, h=1e-5 if central else 1e-6)

    t0 = time.time()
    res = minimize(f, x0, jac=g, method="SLSQP",
                   options=dict(maxiter=maxiter, ftol=tol, disp=False))
    return dict(energy=float(res.fun), theta=res.x, nit=int(res.nit), n_evals=ev.n_evals,
                time=time.time() - t0, history=hist, success=bool(res.success))


def spsa_minimize(f, x0, maxiter, rng, c=0.1, step0=0.03, alpha=0.602, gamma=0.101,
                  n_calib=4, avg_frac=0.25):
    """Plain SPSA with Spall's decaying gains.

    The gain `a` is calibrated so that the first update moves each parameter by about
    `step0` radians.  The returned point is the average of the last `avg_frac` of the
    iterates, which suppresses the residual shot-noise jitter.  Calibration evaluations
    are counted by the caller (they go through `f`).
    """
    x = np.array(x0, dtype=float)
    P = len(x)
    A = 0.1 * maxiter
    g0 = []
    for _ in range(n_calib):
        d = rng.choice([-1.0, 1.0], P)
        g0.append(np.abs((f(x + c * d) - f(x - c * d)) / (2 * c)))
    a = step0 * (A + 1) ** alpha / max(np.mean(g0), 1e-8)
    tail = []
    for k in range(maxiter):
        ak = a / (k + 1 + A) ** alpha
        ck = c / (k + 1) ** gamma
        d = rng.choice([-1.0, 1.0], P)
        g = (f(x + ck * d) - f(x - ck * d)) / (2 * ck) * d
        x = x - ak * g
        if k >= int(maxiter * (1 - avg_frac)):
            tail.append(x.copy())
    return np.mean(tail, axis=0)


def run_vqe_shots(setup: Setup, shots_per_eval, maxiter=120, grouping="qwc", seed=0,
                  x0=None, ev: ExactEvaluator | None = None):
    """Shot-noise VQE with SPSA. Returns the *noiseless* energy at the final parameters (the
    true quality of the solution) together with the total number of shots consumed."""
    rng = np.random.default_rng(seed)
    ev = ev or ExactEvaluator(setup.ansatz, setup.qubit_op)
    plan = MeasurementPlan(setup.qubit_op, grouping)
    P = setup.n_params
    x0 = np.zeros(P) if x0 is None else np.asarray(x0, dtype=float)
    used = [0]
    n_ev = [0]

    def f(x):
        psi = ev.states(np.asarray(x)[None, :])[0]
        e, n = plan.sample_energy(psi, shots_per_eval, rng)
        used[0] += n
        n_ev[0] += 1
        return e + setup.shift

    t0 = time.time()
    theta = spsa_minimize(f, x0, maxiter, rng)
    e_true = ev.energy(theta) + setup.shift
    return dict(energy=float(e_true), theta=theta, total_shots=int(used[0]),
                n_evals=n_ev[0], time=time.time() - t0)


def run_adapt(setup_template: Setup, max_ops=8, grad_tol=1e-4, delta=1e-4):
    """Small, transparent ADAPT-VQE: greedily grow an ansatz from the UCCSD operator pool.

    The pool is the UCCSD excitation list.  At every step the candidate whose appended
    rotation lowers the energy fastest at theta=0 is added, then all parameters are
    re-optimised (noiseless SLSQP).  Returns the final circuit and the energy history.
    """
    cfg = setup_template.cfg
    base = make_ansatz("uccsd", setup_template.problem, setup_template.mapper)
    pool = list(base.operators)
    hf = HartreeFock(setup_template.problem.num_spatial_orbitals,
                     setup_template.problem.num_particles, setup_template.mapper)
    chosen, theta = [], np.zeros(0)
    history = []
    ansatz = None
    for it in range(max_ops):
        best, best_g = None, 0.0
        for k, op in enumerate(pool):
            trial = EvolvedOperatorAnsatz(chosen + [op], initial_state=hf)
            ev = ExactEvaluator(trial, setup_template.qubit_op)
            x = np.append(theta, 0.0)
            e = ev.energies(np.vstack([x + delta * np.eye(len(x))[-1], x - delta * np.eye(len(x))[-1]]))
            g = abs(e[0] - e[1]) / (2 * delta)
            if g > best_g:
                best, best_g = k, g
        if best is None or best_g < grad_tol:
            break
        chosen.append(pool[best])
        ansatz = EvolvedOperatorAnsatz(chosen, initial_state=hf)
        s = Setup(cfg, setup_template.problem, setup_template.mapper, setup_template.qubit_op,
                  ansatz, setup_template.shift, setup_template.n_qubits, ansatz.num_parameters)
        out = run_vqe_exact(s, x0=np.append(theta, 0.0))
        theta = out["theta"]
        history.append(dict(step=it + 1, op_index=int(best), grad=float(best_g),
                            energy=out["energy"]))
    final = Setup(cfg, setup_template.problem, setup_template.mapper, setup_template.qubit_op,
                  ansatz, setup_template.shift, setup_template.n_qubits, ansatz.num_parameters)
    return dict(setup=final, theta=theta, history=history, energy=history[-1]["energy"],
                n_ops=len(chosen))


# --------------------------------------------------------------------------------------
# 9. The standard resource-reduction ladder
# --------------------------------------------------------------------------------------
def ladder_configs(molecule="LiH", distance=None):
    kw = dict(molecule=molecule, distance=distance)
    return [
        Config("0 Baseline: JW, UCCSD", **kw),
        Config("1 + freeze core", freeze_core=True, **kw),
        Config("2 + drop pi orbitals", freeze_core=True, drop=True, **kw),
        Config("3 + Parity (2-qubit red.)", freeze_core=True, drop=True, mapper="parity", **kw),
        Config("4a + tapering (from JW)", freeze_core=True, drop=True, mapper="jw+taper", **kw),
        Config("4b + tapering (from Parity)", freeze_core=True, drop=True,
               mapper="parity+taper", **kw),
        Config("5 BK mapper (6q, for comparison)", freeze_core=True, drop=True,
               mapper="bk", **kw),
    ]


def extra_ladder_configs(molecule="LiH", distance=None):
    """Symmetry tapering that *keeps* the pi orbitals (spatial symmetry gives extra Z2s)."""
    kw = dict(molecule=molecule, distance=distance)
    return [
        Config("A Taper only (no freeze)", mapper="jw+taper", **kw),
        Config("1b + freeze core + taper (keep pi)", freeze_core=True, mapper="jw+taper", **kw),
    ]


def describe(setup: Setup, reference_full: float, e_exact_active: float | None = None,
             do_transpile=True) -> dict:
    row = dict(name=setup.cfg.name, mapper=setup.cfg.mapper, ansatz=setup.cfg.ansatz,
               qubits=setup.n_qubits, params=setup.n_params)
    row.update(operator_metrics(setup.qubit_op))
    if do_transpile:
        row.update(circuit_metrics(setup.ansatz))
    if e_exact_active is not None:
        row["e_exact_active"] = e_exact_active
        row["truncation_err_mHa"] = 1e3 * (e_exact_active - reference_full)
    return row
