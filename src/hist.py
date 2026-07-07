import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Argument parsing
parser = ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output_action", required=True)
parser.add_argument("--output_polyakov", required=True)
parser.add_argument("--L", required=True)
parser.add_argument("--T", required=True)
#parser.add_argument("--beta", required=True)
args = parser.parse_args()

# Process inputs
T = int(args.T)
L = int(args.L)
#beta = float(args.beta)

# Read input file observables
plaq, pol = np.loadtxt(args.input).T
action = 6 * T * L**3 * (1 - plaq)

# Action histogram
fig, ax = plt.subplots()
ax.hist(action, bins=50, color="tab:orange")
ax.set_xlabel("$S$")
plt.savefig(args.output_action, bbox_inches="tight")
plt.close(fig)

# Polyakov loop histogram
fig, ax = plt.subplots()
ax.hist(pol, bins=50)
ax.set_xlabel(r"$\Phi$")
plt.savefig(args.output_polyakov, bbox_inches="tight")
plt.close(fig)
