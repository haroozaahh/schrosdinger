"""Plotting + table helpers for the notebook (reads results/*.json written by run_experiments.py)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(__file__).parent / "results"
CHEM = 1.6  # mHa

# fixed categorical order (reference palette, light surface)
BLUE, ORANGE, AQUA, VIOLET, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#8a8984"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "figure.dpi": 110,
})


def load(name):
    return json.loads((RES / f"{name}.json").read_text())


def _save(fig, name):
    fig.savefig(RES / f"{name}.png", dpi=150, bbox_inches="tight")
    return fig


def short(name):
    return name.split(" ", 1)[1] if name[0].isdigit() or name[:2] in ("A ",) else name


# ----------------------------------------------------------------------------- tables
def ladder_table(rows):
    df = pd.DataFrame(rows)
    cols = dict(name="configuration", qubits="qubits", params="params", n_pauli="Pauli terms",
                n_groups_qwc="QWC groups", cx_a2a="CNOT (all-to-all)", cx_line="CNOT (line)",
                depth_line="depth (line)", truncation_err_mHa="truncation (mHa)",
                vqe_err_vs_full_mHa="VQE error vs FULL exact (mHa)", within_chem_acc="< 1.6 mHa")
    keep = [c for c in cols if c in df]
    out = df[keep].rename(columns=cols)
    return out


# ----------------------------------------------------------------------------- figures
def fig_ladder(rows, title="LiH resource-reduction ladder", save=None):
    names = [short(r["name"]) for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(1, 4, figsize=(15, 0.55 * len(rows) + 1.6), sharey=True)
    panels = [("qubits", "Qubits", BLUE, False), ("params", "Variational parameters", BLUE, True),
              ("cx_line", "CNOTs after transpiling (line)", BLUE, True)]
    for a, (k, lab, col, lg) in zip(ax[:3], panels):
        v = [r[k] for r in rows]
        a.barh(y, v, color=col, height=0.6)
        if lg:
            a.set_xscale("log")
        for yi, vi in zip(y, v):
            a.text(vi * (1.08 if lg else 1) + (0 if lg else 0.15), yi, f"{vi:,}", va="center", fontsize=9, color=INK2)
        a.set_title(lab)
        a.set_xlim(right=max(v) * (4 if lg else 1.2))
    a = ax[3]
    err = [abs(r.get("vqe_err_vs_full_mHa", np.nan)) for r in rows]
    cols = [RED if e > CHEM else AQUA for e in err]
    a.barh(y, err, color=cols, height=0.6)
    a.axvline(CHEM, color=INK2, ls="--", lw=1)
    a.text(CHEM, y.max() + 0.75, " 1.6 mHa", color=INK2, fontsize=9, va="bottom")
    for yi, e in zip(y, err):
        a.text(e + 0.05, yi, f"{e:.2f}", va="center", fontsize=9, color=INK2)
    a.set_title("|Error| vs full exact (mHa)")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(names)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, save) if save else fig


def fig_pareto(ansatz_rows, ladder_rows, save=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    floor = 1e-3
    for r in ladder_rows:
        e = max(abs(r["vqe_err_vs_full_mHa"]), floor)
        ax.scatter(r["cx_line"], e, s=60, color=BLUE, zorder=3, edgecolor=SURF, linewidth=1.5)
        ax.annotate(f'{r["qubits"]}q', (r["cx_line"], e), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color=INK2)
    offs = {"UCCSD": (6, 14), "SUCCD": (6, -12), "PUCCD": (6, -12), "HWE r1": (6, 8),
            "HWE r2": (-4, -16), "HWE r3": (8, 8), "ADAPT-VQE": (6, -16)}
    for r in ansatz_rows:
        e = max(abs(r["err_vs_full_mHa"]), floor)
        ax.scatter(r["cx_line"], e, s=60, color=ORANGE, marker="s", zorder=3, edgecolor=SURF, linewidth=1.5)
        lab = r["name"].split(" (")[0]
        if "reps=" in r["name"]:
            lab += f' r{r["name"].split("reps=")[1][0]}'
        ax.annotate(lab, (r["cx_line"], e), xytext=offs.get(lab, (6, -12)), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.axhline(CHEM, color=INK2, ls="--", lw=1)
    ax.text(ax.get_xlim()[0], CHEM * 1.15, " chemical accuracy (1.6 mHa)", color=INK2, fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("CNOTs after transpiling to a linear chain"); ax.set_ylabel("|error| vs full exact (mHa)")
    ax.set_title("Accuracy vs circuit cost (lower-left is better)")
    ax.scatter([], [], color=BLUE, s=60, label="reduction ladder (UCCSD)")
    ax.scatter([], [], color=ORANGE, marker="s", s=60, label="ansatz comparison (4-qubit LiH)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, save) if save else fig


def fig_shots_analytic(rows, save=None):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(rows)); w = 0.38
    u = [r["shots_ungrouped"] for r in rows]; g = [r["shots_qwc"] for r in rows]
    ax.bar(x - w / 2, u, w - 0.04, color=ORANGE, label="no grouping")
    ax.bar(x + w / 2, g, w - 0.04, color=BLUE, label="qubit-wise-commuting groups")
    for xi, a_, b_ in zip(x, u, g):
        ax.text(xi - w / 2, a_ * 1.1, f"{a_:.1e}", ha="center", fontsize=8, color=INK2)
        ax.text(xi + w / 2, b_ * 1.1, f"{b_:.1e}", ha="center", fontsize=8, color=INK2)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([r["name"] for r in rows], fontsize=9)
    ax.set_ylabel("shots for 1.6 mHa standard error")
    ax.set_title("Measurement cost of ONE energy estimate")
    ax.set_ylim(top=max(u) * 6); ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, save) if save else fig


def fig_shots_vqe(runs, save=None):
    configs = list(dict.fromkeys(r["config"] for r in runs))
    fig, axes = plt.subplots(1, len(configs), figsize=(4.6 * len(configs), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, configs):
        for grouping, col, lab in (("none", ORANGE, "no grouping"), ("qwc", BLUE, "QWC grouping")):
            rr = [r for r in runs if r["config"] == c and r["grouping"] == grouping]
            x = [r["total_shots"] for r in rr]
            y = [max(abs(r["err_mean_mHa"]), 1e-3) for r in rr]
            lo = [max(abs(min(np.abs(r["err_vs_full_mHa"]))), 1e-3) for r in rr]
            hi = [max(np.abs(r["err_vs_full_mHa"])) for r in rr]
            ax.plot(x, y, "-o", color=col, lw=2, ms=6, label=lab)
            ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
        ax.axhline(CHEM, color=INK2, ls="--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(c); ax.set_xlabel("total shots spent in the whole VQE")
    axes[0].set_ylabel("|error| vs full exact (mHa)")
    axes[0].legend(loc="lower left")
    axes[0].text(axes[0].get_xlim()[0], CHEM * 1.15, " 1.6 mHa", color=INK2, fontsize=9)
    fig.suptitle("Shot-noise VQE (SPSA, 4 seeds): error vs shot budget", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, save) if save else fig


def fig_pes(rows, save=None):
    d = [r["d"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(d, [r["e_full"] for r in rows], color=INK2, lw=2, label="full exact (12q space)")
    ax[0].plot(d, [r["e_vqe"] for r in rows], "o", color=BLUE, ms=6, label="reduced VQE (4 qubits)")
    ax[0].plot(d, [r["e_hf"] for r in rows], "--", color=ORANGE, lw=1.5, label="Hartree-Fock")
    ax[0].set_xlabel("Li-H distance (Å)"); ax[0].set_ylabel("total energy (Ha)")
    ax[0].set_title("LiH potential energy curve"); ax[0].legend()
    ax[1].plot(d, [abs(r["err_vs_full_mHa"]) for r in rows], "-o", color=BLUE, lw=2, ms=6,
               label="reduced VQE error")
    ax[1].axhline(CHEM, color=INK2, ls="--", lw=1)
    ax[1].text(d[-1], CHEM * 1.03, "1.6 mHa ", ha="right", color=INK2, fontsize=9)
    ax[1].set_xlabel("Li-H distance (Å)"); ax[1].set_ylabel("|error| vs full exact (mHa)")
    ax[1].set_title("Error of the 4-qubit model along the curve"); ax[1].set_ylim(0)
    fig.tight_layout()
    return _save(fig, save) if save else fig
