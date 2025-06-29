import matplotlib.pyplot as plt
import sys
import os
import random
import numpy as np
import networkx as nx
import powerlaw
from joblib import Parallel, delayed
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from model.core import RecyclingModel

random.seed(133)
np.random.seed(133)

def compute_avalanches(series, delta=0.05):
    '''Compute avalanches (of size of at least delta) from a time series of recycling rates'''
    diffs = np.diff(series)
    avalanches = diffs[diffs > delta]
    return avalanches

# Parameters
T = 1000 # number of time steps
seeds = range(10) # number of runs
delta = 0.01 # threshold for defining an avalanche
N = 150 # number of households
L = int(np.ceil(np.sqrt(N))) # grid size

params = dict(
    N=N,                        # households
    L=L,                        # grid size
    M=9,                        # bins
    k=4,                        # average degree
    beta=0.1,                   # rewiring probability
    delta=0.7,                  # surcharge factor
    c=0.3,                      # base cost
    kappa=0.05,                 # distance cost factor
    epsilon=0.05,               # fraction of eco-champions
    alpha=0.5,                  # social influence weight
    K_default=7,                # bin capacity
    memory_length=100,          # memory length for weighted average
    logit=True,                 # use logit choice model
    lambda_param=5,             # logit scaling parameter
    activation='simultaneous',  # activation type
    decay=0.8                   # decay factor for weighted average
)

# Collect metrics
def simulate_single_seed(seed):
    '''Run a single simulation with given seed and collect metrics'''
    local_params = params.copy()
    np.random.seed(seed)
    model = RecyclingModel(**local_params)
    prev_choices = {i: h.s for i, h in model.households.items()} # Store previous choices to detect switches

    series = {key: [] for key in metrics} # Initialize series for metrics

    for t in range(T):
        # Run the model for one step
        model.step()
        agents = list(model.households.values())
        bins = list(model.bins.values())

        # Collect metrics
        s_vals = np.array([h.s for h in agents]) # Recycling choices
        rho_vals = np.array([h.rho for h in agents]) # Recycling rates
        deltaC_vals = np.array([h.deltaC for h in agents]) # Surcharge values

        r = np.mean(s_vals) # Global recycling rate
        frac_over = sum(b.Q_m > b.K_m for b in bins) / len(bins) # Fraction of overloaded bins

        adopters = [i for i, h in model.households.items() if h.s] 
        if adopters: # If there are any adopters
            # Compute largest connected component and number of clusters in the subgraph of adopters
            subG = model.G.subgraph(adopters)
            largest_cc = max((len(c) for c in nx.connected_components(subG)), default=0)
            num_clusters = len(list(nx.connected_components(subG)))
        else:
            largest_cc = 0
            num_clusters = 0

        switches = sum(1 for i, h in model.households.items() if h.s != prev_choices[i]) # Count switches in recycling choices
        adopted = sum(1 for i, h in model.households.items() if h.s and not prev_choices[i]) # Count new adopters
        abandoned = sum(1 for i, h in model.households.items() if not h.s and prev_choices[i]) # Count abandoned recyclers

        overloaded_bins = {m for m, b in model.bins.items() if b.Q_m > b.K_m} # Set of overloaded bins
        overloaded_agents = sum(1 for h in agents if h.bin_id in overloaded_bins) # Count agents with overloaded bins

        for i, h in model.households.items(): # Update previous choices
            prev_choices[i] = h.s

        if t > 100: # Start collecting metrics after initial warm-up period
            series['recycle_rate'].append(r)                             # Global recycling rate
            series['frac_overloaded'].append(frac_over)                  # Fraction of overloaded bins
            series['largest_cc'].append(largest_cc / N)                  # Normalized size of largest connected component
            series['frac_switches'].append(switches / N)                 # Fraction of agents that switched their recycling choice
            series['avg_surcharge'].append(np.mean(deltaC_vals))         # Average surcharge
            series['var_rho'].append(np.var(rho_vals))                   # Variance of recycling rates
            series['num_overloaded_agents'].append(overloaded_agents)    # Number of agents with overloaded bins
            series['num_clusters'].append(num_clusters)                  # Number of clusters in the subgraph of adopters
            series['num_adopted'].append(adopted)                        # Number of new adopters
            series['num_abandoned'].append(abandoned)                    # Number of abandoned recyclers

    return {k: compute_avalanches(series[k], delta=delta) for k in series} # Collect avalanches for each metric

# Initialize metrics dictionary, metrics definitions are defined above
metrics = {
    'recycle_rate': [],
    'frac_overloaded': [],
    'largest_cc': [],
    'frac_switches': [],
    'avg_surcharge': [],
    'var_rho': [],
    'num_overloaded_agents': [],
    'num_clusters': [],
    'num_adopted': [],
    'num_abandoned': []
}

# Run simulations in parallel
results = Parallel(n_jobs=-1)(delayed(simulate_single_seed)(seed) for seed in tqdm(seeds))
for res in results:
    for k in metrics:
        metrics[k].extend(res[k])

# Plot CCDFs
fig_dir = os.path.join(project_root, 'figures/multiple_outputs')
os.makedirs(fig_dir, exist_ok=True)

# Print summary statistics, fit power-law distributions and plot for each metric
for key, data in metrics.items():
    data = np.array(data)
    data = data[data > 0]
    if len(data) == 0:
        continue

    print(f"Metric: {key}\n  Total avalanches detected: {len(data)}\n  Mean size: {np.mean(data):.4f}\n  Median size: {np.median(data):.4f}\n  Max size: {np.max(data):.4f}")

    try:
        fit = powerlaw.Fit(data, verbose=False, xmin=0.5)
    except Exception as e:
        print(f"Fit failed for {key}: {e}")
        continue

    x_emp = np.logspace(np.log10(fit.xmin), np.log10(data.max()), num=50)
    ccdf_emp = np.array([np.mean(data >= x) for x in x_emp])

    fig, ax = plt.subplots(figsize=(6,4))
    ax.loglog(x_emp, ccdf_emp, 'o-', label='Empirical CCDF (tail)', markersize=4)

    R_exp, p_exp = fit.distribution_compare('power_law', 'exponential')
    R_cut, p_cut = fit.distribution_compare('power_law', 'truncated_power_law')

    if R_cut > 0 and p_cut >= 0.05:
        tp = fit.truncated_power_law
        y_trunc = tp.ccdf(x_emp)
        if len(y_trunc) == len(x_emp):
            ax.loglog(x_emp, y_trunc, '--', label=f'Trunc. PL, α={tp.alpha:.2f}, λ={tp.Lambda:.2f}')

    if R_exp > 0 or p_exp >= 0.05:
        pl = fit.power_law
        y_pl = pl.ccdf(x_emp)
        if len(y_pl) == len(x_emp):
            ax.loglog(x_emp, y_pl, '-.', label=f'PL fit, α={pl.alpha:.2f}')

    ax.set_xlabel(f'{key} jump size')
    ax.set_ylabel('Pr(size ≥ x)')
    ax.set_title(f'Avalanche CCDF for {key} (xmin={fit.xmin:.2f})')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"avalanches_ccdf_{key}.png"), dpi=300)
    plt.close()
