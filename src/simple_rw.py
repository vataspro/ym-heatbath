"""

Try to reweight L20.

"""

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from analysis_tools import id_from_fname, ms_autocorr_time
from pyRw.mrw import MultiRw
import os

plt.style.use("styles/paperdraft.mplstyle")

parser = ArgumentParser()
parser.add_argument("--files", nargs="+", required=True)
args = parser.parse_args()

files = [f for f in args.files if os.path.exists(f)]

NTHERM = 0

for bs_sample_num in range(100):
    BETAS = []
    ACTIONS = []
    MEASUREMENTS = []

    for f in files:
        plaq, polyakov = np.loadtxt(f).T

        L, beta = id_from_fname(f)

        V = 6 * L**3

        action = 6 * V * (1 - plaq)

        tau, _ = ms_autocorr_time(np.abs(polyakov))

        BETAS.append(beta)

        # create bootstrap sample
        len_sample = int(len(action) // tau)
        idx = np.random.randint(len(action), size=len_sample)

        # For reweighting
        ACTIONS.append(action[idx])
        MEASUREMENTS.append(np.abs(polyakov[idx]))

    mrw = MultiRw(BETAS, ACTIONS, autocorr=False, verbose=False)

    b_target = np.linspace(np.min(BETAS), np.max(BETAS), 100)

    Oavr = mrw.reweight(MEASUREMENTS, beta=b_target)
    O2avr = mrw.reweight(MEASUREMENTS, beta=b_target, n=2) 
    O4avr = mrw.reweight(MEASUREMENTS, beta=b_target, n=4) 

    rw_susc = V * (O2avr - Oavr**2)
    binder = 1 - O4avr / (3.*O2avr**2)

    for b, oavr, osusc, bind in zip(b_target, Oavr, rw_susc, binder):
        print(b, oavr, osusc, bind)
