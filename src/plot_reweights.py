import os
import json
import numpy as np
from analysis_tools import id_from_fname
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.lines import Line2D

colors = list(TABLEAU_COLORS)
markers = ["o", "s", "^", "*"]
symbols = {
    "PolyakovLoop": r"$|\Phi|$",
    "PolyakovSusceptibility": r"$\chi_\Phi$",
    "PolyakovBinder": r"$B_4^\Phi$",
}

plt.style.use("styles/paperdraft.mplstyle")


parser = ArgumentParser()
parser.add_argument("--rwfiles", nargs="+", required=True)
parser.add_argument("--datafiles", nargs="+", required=True)
parser.add_argument(
    "--observable",
    required=True,
    choices=["PolyakovLoop", "PolyakovSusceptibility", "PolyakovBinder"],
)
parser.add_argument("--output", required=True)
args = parser.parse_args()


if True:  # args.observable == "PolyakovSusceptibility":
    figsize = (7, 4.6)
else:
    figsize = (3.4, 2.8)
fig, ax = plt.subplots(figsize=figsize, layout="constrained")

rwfiles = [f for f in args.rwfiles if os.path.exists(f)]

for i, f in enumerate(sorted(rwfiles)):
    beta, polyakov, susc, binder = np.loadtxt(f).T

    B = []
    MEAN = []
    ERROR = []

    if args.observable == "PolyakovLoop":
        obs = polyakov
    elif args.observable == "PolyakovSusceptibility":
        obs = susc
    elif args.observable == "PolyakovBinder":
        obs = binder
    for b in np.unique(beta):
        B.append(b)
        idx = np.where(beta == b)
        MEAN.append(np.mean(obs[idx]))
        ERROR.append(np.std(obs[idx]))

    mean = np.array(MEAN)
    err = np.array(ERROR)

    plt.fill_between(
        np.array(B),
        mean - err,
        mean + err,
        alpha=0.5,
        color=colors[i],
        edgecolor="none",
        linewidth=0,
    )
    ax.plot(np.array(B), mean, color=colors[i])


Ls = []
data = {}
datafiles = [f for f in args.datafiles if os.path.exists(f)]

for dataf in datafiles:
    L, beta = id_from_fname(dataf)

    if L not in Ls:
        Ls.append(L)
        data[L] = {}

    with open(dataf, "r") as f:
        datum = json.load(f)

    data[L][beta] = datum[args.observable]

for i, L in enumerate(sorted(Ls)):
    idx = np.argsort(list(data[L].keys()))
    betas = np.array(list(data[L].keys()))[idx]

    values = []
    errors = []
    for b in betas:
        values.append(data[L][b]["value"])
        errors.append(data[L][b]["error"])

    ax.errorbar(
        betas,
        values,
        errors,
        color=colors[i],
        marker=markers[i],
        ls="none",
        label=f"$L = {L}$",
    )


# Figure formatting
auto_handles, auto_labels = ax.get_legend_handles_labels()

legend_elements = [
    Line2D([0], [0], color="k", marker="o", linestyle="None", label="Data"),
    Line2D([0], [0], color="k", lw=0.5, label="Reweighted data"),
]

ax.legend(handles=legend_elements + auto_handles)

ax.set_ylabel(symbols[args.observable])
ax.set_xlabel(r"$\beta$")

ax.set_ylim(bottom=0)

plt.savefig(args.output)
