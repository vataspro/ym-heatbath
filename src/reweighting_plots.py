"""

Analysis of input data with reweighting.

Saves output figures for the Polyakov loop,
its Susceptibility and Binder Cumulant.

Assumes that the input is a text file with
rows of measurements, with the
first row being the plaquette and the
second column being the Polyakov loop.

"""

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from copy import deepcopy
from glob import glob
from analysis_tools import id_from_fname, analyse, NTHERM
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.lines import Line2D
from pyRw.mrw import MultiRw

# Template input data: raw_data/L20/B7.6125/data.txt
D_TEMPLATE = {20: {}, 24: {}, 28: {}, 32: {}}
L_TEMPLATE = {20: [], 24: [], 28: [], 32: []}

# Command line arguments
parser = ArgumentParser()
parser.add_argument("--polyakov_plot", required=True)
parser.add_argument("--susceptibility_plot", required=True)
parser.add_argument("--binder_plot", required=True)
args = parser.parse_args()

# Main
if __name__ == "__main__":
    # Plotting parameters
    plt.style.use("styles/paperdraft.mplstyle")
    colours = list(TABLEAU_COLORS)

    markers = ["o", "s", "^", "*"]

    # Input data files
    files = glob("raw_data/**/data.txt", recursive=True)

    # Symbols for each observable (for y axis)
    symbols = [r"$|\Phi|$", r"$\chi_\Phi$", r"$B_4^\Phi$"]
    # names = ["Polyakov", "Susceptibility", "Binder"]
    output_files = [args.polyakov_plot, args.susceptibility_plot, args.binder_plot]

    # Initialise data dictionaries
    POL_VAL = deepcopy(D_TEMPLATE)
    POL_ERR = deepcopy(D_TEMPLATE)
    SUSC_VAL = deepcopy(D_TEMPLATE)
    SUSC_ERR = deepcopy(D_TEMPLATE)
    BIND_VAL = deepcopy(D_TEMPLATE)
    BIND_ERR = deepcopy(D_TEMPLATE)

    OBS = deepcopy(L_TEMPLATE)
    ACTION = deepcopy(L_TEMPLATE)
    BETA = deepcopy(L_TEMPLATE)

    for f in files:
        L, beta = id_from_fname(f)
        plaquette, polyakov = np.genfromtxt(f)[NTHERM:].T

        polyakov = polyakov
        action = (6 * L**3) * 6 * (1 - plaquette)

        mean, error, susc_mean, susc_err, binder_mean, binder_error, tau = analyse(
            polyakov, L
        )

        POL_VAL[L][beta] = mean
        POL_ERR[L][beta] = error

        SUSC_VAL[L][beta] = susc_mean
        SUSC_ERR[L][beta] = susc_err

        BIND_VAL[L][beta] = binder_mean
        BIND_ERR[L][beta] = binder_error

        OBS[L].append(polyakov.tolist())
        ACTION[L].append(action.tolist())
        BETA[L].append(beta)

    MRW = {}
    for L in [20, 24, 28, 32]:
        mrw = MultiRw(BETA[L], ACTION[L], autocorr=True, verbose=True)

        beta = np.linspace(np.min(BETA[L]), np.max(BETA[L]), 200)

        obs = [np.abs(o) for o in OBS[L]]
        q = mrw.reweight(obs, beta)
        q2 = mrw.reweight(obs, beta, n=2)
        q3 = mrw.reweight(obs, beta, n=3)
        q4 = mrw.reweight(obs, beta, n=4)

        mom2 = q2 - q**2
        mom4 = q4 - 4*q3*q + 6*q2*q**2 - 3*q**4

        susc = 6 * L**3 * mom2
        binder = 1 - q4 / (3.*q2**2)#mom4 / (3.*mom2 ** 2)#1 - q4 / (3. * q2**2)#mom4 / (mom2**2) # this is actually curtosis
        MRW[L] = list(map(lambda x: deepcopy(x), [beta, q, susc, binder]))

    for i, OBS in enumerate(
        [(POL_VAL, POL_ERR), (SUSC_VAL, SUSC_ERR), (BIND_VAL, BIND_ERR)]
    ):
        fig, ax = plt.subplots()

        for clr_idx, key in enumerate(OBS[0].keys()):
            # Analysis (points)
            x = []
            y = []
            err = []
            for k in OBS[0][key].keys():
                x.append(float(k))
                y.append(OBS[0][key][k])
                err.append(OBS[1][key][k])
            ax.errorbar(x, y, err, ls="none", label=f"L = {key}", c=colours[clr_idx], marker=markers[clr_idx])
            

            # Reweight
            ax.plot(MRW[key][0], MRW[key][i + 1], color=colours[clr_idx])

        auto_handles, auto_labels = ax.get_legend_handles_labels()

        legend_elements = [Line2D([0], [0], color="k", marker="o", label="data"),
                           Line2D([0], [0], color="k", lw=0.5, label="reweight")]

        ax.legend(handles=legend_elements + auto_handles)
        ax.set_ylabel(symbols[i])
        ax.set_xlabel(r"$\beta$")

        plt.savefig(output_files[i])
