import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from analysis_tools import id_from_fname, NTHERM


LS = [20, 24, 28, 32]

for L in LS:
    # Load files
    files = glob(f"raw_data/L{L}/**/data.txt", recursive=True)

    files = sorted(files, key=lambda x: float(x.split("/")[2].replace("B", "")))

    fig, ax = plt.subplots()

    for f in files:
        # Get L, beta
        _, beta = id_from_fname(f)

        plaq, _ = np.genfromtxt(f)[NTHERM:].T

        # Get the action
        action = (6 * L**3) * 6 * (1 - plaq)

        # Plot the action histograms for every beta in the corresponding L subplot
        ax.hist(action, bins=30, label=beta, alpha=0.5)

    # Save
    ax.legend()
    ax.set_xlabel(r"$S_E$")
    ax.set_ylabel("counts")
    ax.set_title(f"Action Histogram for L = {L}")
    plt.savefig(f"assets/plots/histograms_L{L}.pdf")
