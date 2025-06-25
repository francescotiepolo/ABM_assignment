import matplotlib.pyplot as plt
import sys
import os
import random
import numpy as np
import networkx as nx
import powerlaw
import warnings

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from model.core import RecyclingModel

random.seed(133)
np.random.seed(133)

def compute_avalanches(series, delta=0.05):
    diffs = np.diff(series)
    avalanches = diffs[diffs > delta]
    return avalanches

# Parameters
T = 1000
seeds = range(0, 10)
delta = 0.01
N = 150
L = int(np.ceil(np.sqrt(N)))

params = dict(
    N=N,
    L=L,
    M=9,
    k=4,
    beta=0.1,
    delta=0.7,
    c=0.3,
    kappa=0.05,
    epsilon=0.05,
    alpha=0.5,
    K_default=7,
    memory_length=100,
    logit=True,
    lambda_param=5,
    activation='simultaneous',
    decay=0.8
)

# Collect avalanches for all metrics
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

for seed in seeds:
    params['seed'] = seed
    model = RecyclingModel(**params)
    prev_choices = {i: h.s for i, h in model.households.items()}

    series = {key: [] for key in metrics}

    for t in range(T):
        model.step()

        agents = list(model.households.values())
        bins = list(model.bins.values())

        s_vals = np.array([h.s for h in agents])
        rho_vals = np.array([h.rho for h in agents])
        deltaC_vals = np.array([h.deltaC for h in agents])

        r = np.mean(s_vals)
        frac_over = sum(b.Q_m > b.K_m for b in bins) / len(bins)

        adopters = [i for i, h in model.households.items() if h.s]
        if adopters:
            subG = model.G.subgraph(adopters)
            largest_cc = max((len(c) for c in nx.connected_components(subG)), default=0)
            num_clusters = len(list(nx.connected_components(subG)))
        else:
            largest_cc = 0
            num_clusters = 0

        switches = sum(1 for i, h in model.households.items() if h.s != prev_choices[i])
        adopted = sum(1 for i, h in model.households.items() if h.s and not prev_choices[i])
        abandoned = sum(1 for i, h in model.households.items() if not h.s and prev_choices[i])

        overloaded_bins = {m for m, b in model.bins.items() if b.Q_m > b.K_m}
        overloaded_agents = sum(1 for h in agents if h.bin_id in overloaded_bins)

        for i, h in model.households.items():
            prev_choices[i] = h.s

        if t > 100:
            series['recycle_rate'].append(r)
            series['frac_overloaded'].append(frac_over)
            series['largest_cc'].append(largest_cc / N)
            series['frac_switches'].append(switches / N)
            series['avg_surcharge'].append(np.mean(deltaC_vals))
            series['var_rho'].append(np.var(rho_vals))
            series['num_overloaded_agents'].append(overloaded_agents)
            series['num_clusters'].append(num_clusters)
            series['num_adopted'].append(adopted)
            series['num_abandoned'].append(abandoned)

    for key in metrics:
        aval = compute_avalanches(series[key], delta=delta)
        metrics[key].extend(aval)

# Plot CCDFs
fig_dir = os.path.join(project_root, 'figures/multiple_outputs')
os.makedirs(fig_dir, exist_ok=True)

for key, data in metrics.items():
    if len(data) == 0:
        continue
    data = np.array(data)
    data = data[data > 0]
    if len(data) == 0:
        continue

    min_size, max_size = np.min(data), np.max(data)
    bins = np.logspace(np.log10(min_size), np.log10(max_size), num=50)
    ccdf_smooth = np.array([np.mean(data >= b) for b in bins])

    print(f"Metric: {key}")
    print(f"  Total avalanches detected: {len(data)}")
    print(f"  Mean size: {np.mean(data):.4f}")
    print(f"  Median size: {np.median(data):.4f}")
    print(f"  Max size: {np.max(data):.4f}")

    # Fit power law and compare distributions
    fit = powerlaw.Fit(data, verbose=False)
    R_exp, p_exp = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    R_cut, p_cut = fit.distribution_compare('power_law', 'truncated_power_law', normalized_ratio=True)

    ks_pl = fit.power_law.KS()
    ks_trunc = fit.truncated_power_law.KS()

    print(f"  Power law alpha: {fit.alpha:.2f}, xmin: {fit.xmin:.2f}")
    print(f"  Power law vs Exponential: R = {R_exp:.2f}, p = {p_exp:.4f}")
    print(f"  Power law vs Truncated: R = {R_cut:.2f}, p = {p_cut:.4f}")
    print(f"  KS (Power law): {ks_pl:.4f}")
    print(f"  KS (Truncated power law): {ks_trunc:.4f}")

    # 1) define your x-grid from xmin to xmax
    x_emp = np.logspace(np.log10(fit.xmin), np.log10(max_size), num=50)

    # 2) compute empirical CCDF on that grid
    ccdf_emp = np.array([np.mean(data >= x) for x in x_emp])

    # 3) plot the empirical tail yourself
    fig, ax = plt.subplots(figsize=(6,4))
    ax.loglog(x_emp, ccdf_emp, 'o-', label='Empirical CCDF (tail)', markersize=4)

    # 4) overlay truncated PL if it’s the preferred model
    if R_cut > 0 and p_cut >= 0.05:
        tp = fit.truncated_power_law
        y_trunc = tp.ccdf(x_emp)
        ax.loglog(x_emp, y_trunc, '--', label=f'Trunc. PL, α={tp.alpha:.2f}, λ={tp.Lambda:.2f}')

    # 5) overlay pure PL if it’s preferred
    if R_exp > 0 or p_exp >= 0.05:
        # note power_law.ccdf() starts at 1.0 at xmin
        pl = fit.power_law
        y_pl = pl.ccdf(x_emp)
        ax.loglog(x_emp, y_pl, '-.', label=f'PL fit, α={pl.alpha:.2f}')

    ax.set_xlabel(f'{key} jump size')
    ax.set_ylabel('Pr(size ≥ x)')
    ax.set_title(f'Avalanche CCDF for {key} (xmin={fit.xmin:.2f})')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"avalanches_ccdf_{key}.png"), dpi=300)
    plt.close()