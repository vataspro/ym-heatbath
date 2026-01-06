"""

Analysis of input data with no reweighting.

Saves output figures for the Polyakov loop,
its Susceptibility and Binder Cumulant.

Assumes that the input is a text file with
rows of measurements, with the
second column being the Polyakov loop.

"""

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from copy import deepcopy
from glob import glob
from analysis_tools import id_from_fname, analyse

# Template input data: raw_data/L20/B7.6125/data.txt
D_TEMPLATE = {20: {}, 24: {}, 28: {}, 32: {}}

# Command line arguments
parser = ArgumentParser()
parser.add_argument("--polyakov_plot", required=True)
parser.add_argument("--susceptibility_plot", required=True)
parser.add_argument("--binder_plot", required=True)
args = parser.parse_args()

# Main
if __name__ == "__main__":
    plt.style.use("styles/paperdraft.mplstyle")

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

    for f in files:
        polyakov = np.abs(np.genfromtxt(f)[:, 1])

        L, beta = id_from_fname(f)
        mean, error, susc_mean, susc_err, binder_mean, binder_error, tau = analyse(
            polyakov, L
        )

        POL_VAL[L][beta] = mean
        POL_ERR[L][beta] = error

        SUSC_VAL[L][beta] = susc_mean
        SUSC_ERR[L][beta] = susc_err

        BIND_VAL[L][beta] = binder_mean
        BIND_ERR[L][beta] = binder_error

    for i, OBS in enumerate(
        [(POL_VAL, POL_ERR), (SUSC_VAL, SUSC_ERR), (BIND_VAL, BIND_ERR)]
    ):
        fig, ax = plt.subplots()

        for key in OBS[0].keys():
            x = []
            y = []
            err = []
            for k in OBS[0][key].keys():
                x.append(float(k))
                y.append(OBS[0][key][k])
                err.append(OBS[1][key][k])
            ax.errorbar(x, y, err, marker="o", ls="none", label=key)

        ax.legend()
        ax.set_ylabel(symbols[i])
        ax.set_xlabel(r"$\beta$")

        plt.savefig(output_files[i])
