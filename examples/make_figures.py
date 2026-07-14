"""make_figures.py — Figure del paper dai JSON dei risultati (repo root)."""
import json, sys
from math import log, pi, sqrt
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def zg(m):
    z = sqrt(2 * log(m))
    return z - (log(log(m)) + log(4 * pi)) / (2 * z)


def fig_capacity():
    d = json.loads((ROOT / "seed10_results.json").read_text())["law4"]["per_d"]
    dims = sorted(int(k) for k in d)
    ns = [d[str(k)][0] for k in dims]
    cis = [d[str(k)][1] for k in dims]
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.errorbar(dims, ns, yerr=cis, fmt="o", color="#1a6faf", capsize=3,
                label="misurato (10 seed, CI95)")
    xs = np.linspace(400, 4400, 100)
    ax.plot(xs, [0.92 * 2 * x / (pi * zg(2 * n + 11) ** 2)
                 for x, n in zip(xs, np.interp(xs, dims, ns))],
            "--", color="#c0392b", label="teoria: k·2D/(π·z_G²), k=0.92")
    ax.set_xlabel("D (dimensione)"); ax.set_ylabel("N* (collasso 50%)")
    ax.set_title("Law IV — capacità")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fig1_capacity.png")


def fig_depth():
    d = json.loads((ROOT / "depth_scaling_results.json").read_text())["dmin"]
    hs = sorted(int(k) for k in d)
    ds = [d[str(h)]["dmin"] for h in hs]
    sd = [d[str(h)]["sd"] for h in hs]
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.errorbar(hs, ds, yerr=sd, fmt="o-", color="#1a6faf",
                capsize=3, label="D_min misurato (acc ≥95%)")
    ax.plot(hs, [ds[0] * h for h in hs], ":", color="#999",
            label="crescita lineare (esclusa)")
    ax.set_xscale("log", base=2); ax.set_ylim(0, 12000)
    ax.set_xlabel("h (hop)"); ax.set_ylabel("D minimo")
    ax.set_title("Prop. 3.6 — la profondità è log-economica")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fig2_depth.png")


def fig_proofwriter():
    d = json.loads((ROOT / "seed10_results.json").read_text())["proofwriter"]
    depths = sorted(int(k) for k in d)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.errorbar(depths, [d[str(k)]["acc"] * 100 for k in depths],
                yerr=[d[str(k)]["ci95"] * 100 for k in depths],
                fmt="o-", color="#1a6faf", capsize=3, label="ABM (10 seed)")
    ax.axhline(42, ls="--", color="#c0392b", label="baseline maggioranza")
    ax.set_ylim(0, 105); ax.set_xlabel("profondità di inferenza")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("ProofWriter — oracolo di verità algebrico")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fig3_proofwriter.png")


def fig_compiler():
    d = json.loads((ROOT / "compiler_bench_results.json").read_text())
    ns = sorted(int(k) for k in d)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.plot(ns, [d[str(n)]["naive"] * 100 for n in ns], "o-",
            color="#c0392b", label="interpretato (2 cleanup)")
    ax.plot(ns, [d[str(n)]["p2"] * 100 for n in ns], ":",
            color="#c0392b", alpha=0.6, label="p² (Law V)")
    ax.plot(ns, [d[str(n)]["compiled"] * 100 for n in ns], "s-",
            color="#1a6faf", label="compilato (1 cleanup su T₂)")
    ax.set_xlabel("catene a 2 hop"); ax.set_ylabel("accuracy (%)")
    ax.set_title("Calcolo §5.2 — compilazione a sleep-time")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fig4_compiler.png")


def fig_contract():
    d = json.loads((ROOT / "capacity_contract_results.json").read_text())
    ns = sorted(int(k) for k in d)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.plot(ns, [d[str(n)]["pred"] * 100 for n in ns], "--",
            color="#c0392b", label="predetto (zero parametri)")
    ax.plot(ns, [d[str(n)]["meas"] * 100 for n in ns], "o-",
            color="#1a6faf", label="misurato")
    ax.set_xlabel("N fatti in 1 KB (D=8192)"); ax.set_ylabel("accuracy (%)")
    ax.set_title("Capacity contract — |err| medio 4.2%")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "fig5_contract.png")


if __name__ == "__main__":
    for f in (fig_capacity, fig_depth, fig_proofwriter, fig_compiler,
              fig_contract):
        f()
        print("✓", f.__name__)
    print("→", OUT)
