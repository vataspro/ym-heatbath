import numpy as np
import json
from analysis_tools import ms_autocorr_time, id_from_fname
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--file", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

class observable:
    def __init__(self, name):
        self.name = name
        self.d = {"value":None,
                  "error":None}

    def __getitem__(self, key):
        return self.d[key]

    def __setitem__(self, key, value):
        self.d[key] = value

    def __repr__(self):
        return f"{self.name}:\n{repr(self.d)}"

def bs_sample_once(arr, tau=None, f = lambda x : np.mean(x)):

    if tau is None:
        tau, _ = ms_autocorr_time(arr)
    size = int(len(arr) // np.ceil(tau))
    idx = np.random.randint(len(arr), size=size)

    return f(arr[idx])

if not os.path.exists(args.file):
    print(f"file {args.file} does not exist, exiting with failure")
    exit(1)

try:
    plaq, polyakov = np.loadtxt(args.file).T
except:
    raise ValueError(f"file {args.file} does not have correct format")

polyakov = np.abs(polyakov)

# Define observables:
polyakov_loop = observable("PolyakovLoop")
polyakov_susceptibility = observable("PolyakovSusceptibility")
polyakov_binder = observable("PolyakovBinder")

OBSERVABLE_LIST = [polyakov_loop, polyakov_susceptibility, polyakov_binder]

L, beta = id_from_fname(args.file)
V = 6 * L**3
LAMBDA_Fs = [lambda x : np.mean(x), 
             lambda x : V * (np.mean(x**2) - np.mean(x)**2),
             lambda x : 1  - np.mean(x**4) / (3. * np.mean(x**2)**2)]


tau, _ = ms_autocorr_time(polyakov)
for obs, lambda_f in zip(OBSERVABLE_LIST, LAMBDA_Fs):
    obs["value"] = lambda_f(polyakov)

    bs_samples = []
    for bs_sample_num in range(200):
        bs_samples.append(bs_sample_once(polyakov, tau=tau, f=lambda_f))

    obs["error"] = np.std(bs_samples)


# Write data
data = {obs.name : obs.d for obs in OBSERVABLE_LIST}

with open(args.output, "w") as f:
    json.dump(data, f)
