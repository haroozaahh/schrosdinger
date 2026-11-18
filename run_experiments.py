"""
run_experiments.py -- reproduces every number in the notebook and saves it to results/*.json

Usage:  python run_experiments.py <phase> [<phase> ...]
Phases: ladder | ansatz | shots | pes | beh2 | all
        (ladder includes the 12-qubit baseline VQE and takes a while; run it once.)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from resource_reduction import (
    CHEMICAL_ACCURACY, Config, ExactEvaluator, MeasurementPlan, build_setup, describe,
    exact_energy_sector, extra_ladder_configs, full_exact_energy, ladder_configs, run_adapt, run_vqe_exact,
    run_vqe_shots, shots_required, circuit_metrics, operator_metrics,
)

RES = Path(__file__).parent / "results"
RES.mkdir(exist_ok=True)
MHA = 1e3


def save(name, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=2, default=conv))
    print(f"[saved] results/{name}.json", flush=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------------------
# Phase 1+2: baseline and the lever-by-lever ladder
# ---------------------------------------------------------------------------------------
def phase_ladder(molecule="LiH", out_name="ladder", with_vqe=True, configs=None):
    ref = full_exact_energy(molecule)
    log(f"[{molecule}] full exact energy {ref:.8f} Ha")
    rows = []
    for cfg in (configs if configs is not None else ladder_configs(molecule)):
        log("config:", cfg.name)
        s = build_setup(cfg)
        ex = exact_energy_sector(s.problem)
        row = describe(s, ref, ex)
        ev = ExactEvaluator(s.ansatz, s.qubit_op)
        row["e_hf"] = ev.energy(np.zeros(s.n_params)) + s.shift
        if with_vqe:
            out = run_vqe_exact(s, ev=ev)
            row.update(e_vqe=out["energy"], vqe_nit=out["nit"], vqe_evals=out["n_evals"],
                       vqe_time_s=out["time"],
                       vqe_err_vs_active_mHa=MHA * (out["energy"] - ex),
                       vqe_err_vs_full_mHa=MHA * (out["energy"] - ref))
            row["within_chem_acc"] = bool(abs(row["vqe_err_vs_full_mHa"]) < CHEMICAL_ACCURACY * MHA)
            sh = shots_required(s, out["theta"], ev=ev)
            row.update(shots_ungrouped=sh["none"], shots_qwc=sh["qwc"], shots_l1_bound=sh["l1_bound"])
            row["theta_opt"] = out["theta"]
        log(f"   q={row['qubits']} P={row['params']} cx_line={row.get('cx_line')} "
            f"err_full={row.get('vqe_err_vs_full_mHa', float('nan')):.4f} mHa")
        rows.append(row)
        save(out_name, dict(molecule=molecule, e_full_exact=ref, rows=rows))
    return rows


def phase_ladder_extra():
    return phase_ladder("LiH", "ladder_extra", True, extra_ladder_configs("LiH"))


# ---------------------------------------------------------------------------------------
# Phase 3: shorter circuits (4-qubit LiH, compare ansatze)
# ---------------------------------------------------------------------------------------
def phase_ansatz():
    ref = full_exact_energy("LiH")
    base = dict(molecule="LiH", freeze_core=True, drop=True, mapper="parity")
    rows = []
    rng = np.random.default_rng(1)

    def record(name, s, out, extra=None):
        ex = exact_energy_sector(s.problem)
        row = dict(name=name, qubits=s.n_qubits, params=s.n_params,
                   e_vqe=out["energy"],
                   err_vs_active_mHa=MHA * (out["energy"] - ex),
                   err_vs_full_mHa=MHA * (out["energy"] - ref))
        row.update(circuit_metrics(s.ansatz))
        if extra:
            row.update(extra)
        rows.append(row)
        log(f"  {name:28s} P={s.n_params:2d} cx_line={row['cx_line']:3d} "
            f"err_active={row['err_vs_active_mHa']:+.5f} mHa")

    for kind in ("uccsd", "succd", "puccd"):
        s = build_setup(Config(f"{kind}", ansatz=kind, **base))
        record(kind.upper(), s, run_vqe_exact(s))
    # hardware-efficient: random restarts (it has no good start point)
    for reps in (1, 2, 3):
        s = build_setup(Config(f"hwe{reps}", ansatz="hwe", hwe_reps=reps, **base))
        ev = ExactEvaluator(s.ansatz, s.qubit_op)
        best = None
        for r in range(6):
            x0 = rng.uniform(-0.3, 0.3, s.n_params) if r else np.zeros(s.n_params)
            out = run_vqe_exact(s, x0=x0, ev=ev)
            if best is None or out["energy"] < best["energy"]:
                best = out
        record(f"HWE (EfficientSU2, reps={reps})", s, best)
    # ADAPT-VQE
    tmpl = build_setup(Config("adapt", ansatz="uccsd", **base))
    ad = run_adapt(tmpl, max_ops=8, grad_tol=1e-4)
    record("ADAPT-VQE (UCCSD pool)", ad["setup"],
           dict(energy=ad["energy"]), extra=dict(adapt_history=ad["history"]))
    save("ansatz", dict(e_full_exact=ref, rows=rows))


# ---------------------------------------------------------------------------------------
# Phase 4: measurement cost
# ---------------------------------------------------------------------------------------
def phase_shots():
    ref = full_exact_energy("LiH")
    cfgs = {
        "JW 6q (drop pi)": Config("JW 6q", freeze_core=True, drop=True, mapper="jw"),
        "BK 6q (drop pi)": Config("BK 6q", freeze_core=True, drop=True, mapper="bk"),
        "Parity 4q (drop pi)": Config("Parity 4q", freeze_core=True, drop=True, mapper="parity"),
        "Tapered 4q (drop pi)": Config("Tapered 4q", freeze_core=True, drop=True, mapper="jw+taper"),
        "Tapered 6q (keep pi)": Config("Tapered 6q", freeze_core=True, mapper="jw+taper"),
    }
    # (a) analytic: how many shots does each representation need for chemical accuracy?
    analytic = []
    thetas = {}
    for name, cfg in cfgs.items():
        s = build_setup(cfg)
        ev = ExactEvaluator(s.ansatz, s.qubit_op)
        out = run_vqe_exact(s, ev=ev)
        thetas[name] = out["theta"]
        sh = shots_required(s, out["theta"], ev=ev)
        om = operator_metrics(s.qubit_op)
        analytic.append(dict(name=name, qubits=s.n_qubits, **om,
                             shots_ungrouped=sh["none"], shots_qwc=sh["qwc"],
                             shots_l1_bound=sh["l1_bound"]))
        log(f"  {name:16s} paulis={om['n_pauli']} groups={om['n_groups_qwc']} "
            f"shots(ungrouped)={sh['none']:.2e} shots(qwc)={sh['qwc']:.2e}")
    save("shots_analytic", dict(rows=analytic))

    # (b) simulated shot-noise VQE at fixed total shot budgets, 4 seeds each
    maxiter = 120
    runs = []
    for name in ("JW 6q (drop pi)", "Tapered 4q (drop pi)", "Tapered 6q (keep pi)"):
        s = build_setup(cfgs[name])
        ev = ExactEvaluator(s.ansatz, s.qubit_op)
        ex = exact_energy_sector(s.problem)
        for grouping in ("qwc", "none"):
            for budget in (1e5, 1e6, 1e7, 1e8):
                per_eval = int(budget / (2 * (maxiter + 4)))
                errs, excess, used = [], [], []
                for seed in range(4):
                    out = run_vqe_shots(s, per_eval, maxiter=maxiter, grouping=grouping,
                                        seed=seed, ev=ev)
                    errs.append(MHA * (out["energy"] - ref))
                    excess.append(MHA * (out["energy"] - ex))
                    used.append(out["total_shots"])
                runs.append(dict(config=name, grouping=grouping, budget=budget,
                                 shots_per_eval=per_eval, total_shots=float(np.mean(used)),
                                 err_vs_full_mHa=errs, excess_vs_active_mHa=excess,
                                 err_mean_mHa=float(np.mean(errs)),
                                 excess_mean_mHa=float(np.mean(excess)),
                                 excess_std_mHa=float(np.std(excess))))
                log(f"  {name:22s} {grouping:4s} budget={budget:.0e} "
                    f"excess={np.mean(excess):+.3f}+-{np.std(excess):.3f} mHa "
                    f"total_err={np.mean(errs):+.3f}")
    save("shots_vqe", dict(e_full_exact=ref, maxiter=maxiter, runs=runs))


# ---------------------------------------------------------------------------------------
# Phase 5 extra: potential energy surface with the reduced model
# ---------------------------------------------------------------------------------------
def phase_pes():
    dists = [1.0, 1.2, 1.4, 1.595, 1.8, 2.0, 2.4, 2.8, 3.2, 4.0]
    rows = []
    rng = np.random.default_rng(3)
    for d in dists:
        ref = full_exact_energy("LiH", d)
        s = build_setup(Config("pes", distance=d, freeze_core=True, drop=True, mapper="parity"))
        ev = ExactEvaluator(s.ansatz, s.qubit_op)
        ex = exact_energy_sector(s.problem)
        best = None
        for r in range(5):  # a few starts: stretched geometries are more correlated
            x0 = np.zeros(s.n_params) if r == 0 else rng.uniform(-0.2, 0.2, s.n_params)
            out = run_vqe_exact(s, x0=x0, ev=ev)
            if best is None or out["energy"] < best["energy"]:
                best = out
        e_hf = ev.energy(np.zeros(s.n_params)) + s.shift
        rows.append(dict(d=d, e_full=ref, e_active_exact=ex, e_vqe=best["energy"], e_hf=e_hf,
                         err_vs_full_mHa=MHA * (best["energy"] - ref),
                         trunc_mHa=MHA * (ex - ref)))
        log(f"  d={d:.3f} full={ref:.5f} vqe={best['energy']:.5f} "
            f"err={rows[-1]['err_vs_full_mHa']:+.3f} mHa (trunc {rows[-1]['trunc_mHa']:+.3f})")
    save("pes", dict(rows=rows))


# ---------------------------------------------------------------------------------------
# Stretch: BeH2 (7 spatial orbitals -> 14 qubits unreduced)
# ---------------------------------------------------------------------------------------
def phase_beh2():
    ref = full_exact_energy("BeH2")
    log(f"BeH2 full exact {ref:.8f}")
    rows = []
    for cfg in ladder_configs("BeH2") + extra_ladder_configs("BeH2"):
        log("config:", cfg.name)
        s = build_setup(cfg)
        ex = exact_energy_sector(s.problem)
        row = describe(s, ref, ex, do_transpile=True)
        if s.n_qubits <= 8:  # VQE only where it is cheap; baseline gets metrics only
            ev = ExactEvaluator(s.ansatz, s.qubit_op)
            out = run_vqe_exact(s, ev=ev)
            row.update(e_vqe=out["energy"], vqe_err_vs_active_mHa=MHA * (out["energy"] - ex),
                       vqe_err_vs_full_mHa=MHA * (out["energy"] - ref))
            sh = shots_required(s, out["theta"], ev=ev)
            row.update(shots_ungrouped=sh["none"], shots_qwc=sh["qwc"])
        log(f"   q={row['qubits']} P={row['params']} trunc={row.get('truncation_err_mHa', 0):.3f} mHa")
        rows.append(row)
        save("beh2", dict(e_full_exact=ref, rows=rows))


PHASES = dict(ladder=phase_ladder, ladder_extra=phase_ladder_extra, ansatz=phase_ansatz, shots=phase_shots, pes=phase_pes,
              beh2=phase_beh2)

if __name__ == "__main__":
    todo = sys.argv[1:] or ["all"]
    if "all" in todo:
        todo = list(PHASES)
    for ph in todo:
        log("=== phase:", ph)
        PHASES[ph]()
