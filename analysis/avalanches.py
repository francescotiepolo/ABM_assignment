import matplotlib.pyplot as plt
import sys
import os
import random
import numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from model.core import RecyclingModel

random.seed(133)
np.random.seed(133)

def compute_avalanches(series, delta=0.05):
    diffs = np.diff(series)
    avalanches = diffs[diffs > delta]
    return avalanches

# 1. Parameters for avalanche analysis
T = 1000                    # number of time steps
seeds = range(0, 10)         # multiple runs for statistics
delta = 0.01                # threshold for defining an avalanche

N = 150
L = int(np.ceil(np.sqrt(N)))

# Parameter set
params = dict(
    N=N,                         # households
    L=L,                         # grid size
    M=9,                         # bins
    k=4,                         # average degree
    beta=0.1,                    # rewiring probability
    delta=0.7,                   # surcharge factor
    c=0.3,                       # base cost
    kappa=0.05,                  # distance cost factor
    epsilon=0.05,                # fraction of eco-champions
    alpha=0.5,                   # social influence weight             
    K_default=10,                # bin capacity
    memory_length=100,           # memory length for weighted average
    logit=True,                  # use logit choice model
    lambda_param=20,             # logit scaling parameter
    activation='simultaneous',   # activation type
    decay=0.8                    # decay factor for weighted average
)

# 2. Collect avalanches across runs
all_avalanches = []
for seed in seeds:
    params['seed'] = seed
    model = RecyclingModel(**params)
    r_series = []
    for round in range(T):
        model.step()
        r = np.mean([h.s for h in model.households.values()])
        r_series.append(r) if round > 100 else None
    aval = compute_avalanches(r_series, delta=delta)
    all_avalanches.extend(aval)

# 3. Compute CCDF with log-spaced bins
min_size = np.min(all_avalanches)
max_size = np.max(all_avalanches)
bins = np.logspace(np.log10(min_size), np.log10(max_size), num=50)
ccdf_smooth = np.array([np.mean(np.array(all_avalanches) >= b) for b in bins])

# 4. Plot CCDF
fig_dir = os.path.join(project_root, 'figures')

plt.figure(figsize=(6,4))
plt.loglog(bins, ccdf_smooth, linestyle='-', linewidth=2)
plt.scatter(bins, ccdf_smooth, s=20)
plt.xlabel('Avalanche size (jump in R)')
plt.ylabel('Pr(size ≥ x)')
plt.title(f'Avalanche CCDF (δ={delta})')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f"avalanches_ccdf_delta_{delta}.png"), dpi=300)
plt.show()

# 5. Print summary statistics
print(f"Total avalanches detected: {len(all_avalanches)}")
print(f"Mean size: {np.mean(all_avalanches):.4f}")
print(f"Median size: {np.median(all_avalanches):.4f}")
print(f"Max size: {np.max(all_avalanches):.4f}")