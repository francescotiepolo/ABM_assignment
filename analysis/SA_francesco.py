from mesa.batchrunner import FixedBatchRunner, BatchRunner
from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from model.core import RecyclingModel

# Create output directory
plot_dir = os.path.join(project_root, 'figures/SA')
data_dir = os.path.join(project_root, 'data')

# Define problem for sensitivity analysis 
problem = {
    'num_vars': 10,
    'names': ['M', 'delta', 'c', 'kappa', 'epsilon', 'alpha', 'K_default', 'memory_length', 'lambda_param', 'decay'],
    'bounds': [
        [3, 20],           # M
        [0.1, 1.0],        # delta
        [0.1, 1.0],        # c
        [0.01, 0.2],       # kappa
        [0.01, 0.2],       # epsilon
        [0.1, 1.0],        # alpha
        [5, 20],           # K_default
        [10, 200],         # memory_length
        [1, 50],           # lambda_param
        [0.1, 1.0]         # decay
    ]
}

default_values = {
    'N': 100,                       # households
    'L': 10,                        # grid size
    'M': 9,                         # bins
    'k': 4,                         # average degree
    'beta': 0.1,                    # rewiring probability
    'delta': 0.7,                   # surcharge factor
    'c': 0.3,                       # base cost
    'kappa': 0.05,                  # distance cost factor
    'epsilon': 0.05,                # fraction of eco-champions
    'alpha': 0.5,                   # social influence weight             
    'K_default': 10,                # bin capacity
    'memory_length': 100,           # memory length for weighted average
    'logit': True,                  # use logit choice model
    'lambda_param': 20,             # logit scaling parameter
    'activation': 'simultaneous',   # activation type
    'decay': 0.8                   # decay factor for weighted average
}

# OFAT Sensitivity 
replicates = 250
max_steps = 100
samples_per_param = 20

def run_single_ofat_run(var, val, max_steps, fixed_params, seed=None):
    # Create model with single parameter value
    # Set random seed if needed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    model_params = {var: val}
    model_params.update(fixed_params)
    model = RecyclingModel(**model_params)
    for _ in range(max_steps):
        model.step()
    recycle_rate = np.mean([h.s for h in model.households.values()])
    return recycle_rate

data_ofat = {}

for i, var in enumerate(problem['names']):
    values = np.linspace(*problem['bounds'][i], num=samples_per_param)
    fixed_params = {k: v for k, v in default_values.items() if k != var}

    all_dfs = []
    for val in tqdm(values, desc=f"OFAT param {var}"):
        # Parallelize over replicates
        results = Parallel(n_jobs=-1)(
            delayed(run_single_ofat_run)(var, val, max_steps, fixed_params, seed) for seed in range(replicates)
        )
        # Aggregate results into DataFrame
        df_val = pd.DataFrame({
            var: val,
            "Recycle_Rate": results
        })
        all_dfs.append(df_val)

    data_ofat[var] = pd.concat(all_dfs, ignore_index=True)
    data_ofat[var].to_csv(os.path.join(data_dir, f"OFAT_{var}.csv"), index=False)


def plot_param_var_conf(ax, df, var, param):
    x = df.groupby(var).mean().reset_index()[var]
    y = df.groupby(var).mean()[param]
    replicates = df.groupby(var)[param].count()
    err = 1.96 * df.groupby(var)[param].std() / np.sqrt(replicates)

    ax.plot(x, y, c='k')
    ax.fill_between(x, y - err, y + err)
    ax.set_xlabel(var)
    ax.set_ylabel(param)


def plot_all_ofat(data, param):
    f, axs = plt.subplots(3, figsize=(8, 12))
    for i, var in enumerate(problem['names']):
        plot_param_var_conf(axs[i], data[var], var, param)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"OFAT_{param}.png"))
    plt.close()


plot_all_ofat(data_ofat, "Recycle_Rate")

# Sobol

D = len(problem['names'])
distinct_samples = 128
# Compute number of replicates to hit ~5000 runs
target_total_runs = 5000
replicates = max(1, target_total_runs // (distinct_samples * (D + 2)))

print(f"Using {distinct_samples} distinct samples and {replicates} replicates")
print(f"Total runs: {distinct_samples * (D + 2) * replicates}")

data_sobol = pd.DataFrame(columns=problem['names'] + ["Recycle_Rate", "Run"])

def run_model(run_index, vals):
    var_dict = {k: v for k, v in zip(problem['names'], vals)}
    model = RecyclingModel(**var_dict)
    for _ in range(max_steps):
        model.step()
    recycle_rate = np.mean([h.s for h in model.households.values()])
    row = dict(zip(problem['names'], vals))
    row.update({"Recycle_Rate": recycle_rate, "Run": run_index})
    return row

param_values = saltelli.sample(problem, distinct_samples, calc_second_order=False)
total_runs = replicates * len(param_values)

run_inputs = [
    (r * len(param_values) + i, vals)
    for r in range(replicates)
    for i, vals in enumerate(param_values)
]

results = Parallel(n_jobs=-1)(  
    delayed(run_model)(run_index, vals)
    for run_index, vals in tqdm(run_inputs, desc="Running simulations")
)

data_sobol = pd.DataFrame(results)
data_sobol.to_csv(os.path.join(project_root, 'data', 'sobol_data.csv'), index=False)

# Run Sobol analysis
Si = sobol.analyze(problem, data_sobol['Recycle_Rate'].values, print_to_console=True, calc_second_order=False)


def plot_index(s, params, i, title=''):
    if i == '2':
        p = len(params)
        params = list(combinations(params, 2))
        indices = s['S' + i].reshape((p ** 2))
        indices = indices[~np.isnan(indices)]
        errors = s['S' + i + '_conf'].reshape((p ** 2))
        errors = errors[~np.isnan(errors)]
    else:
        indices = s['S' + i]
        errors = s['S' + i + '_conf']
        plt.figure()

    l = len(indices)
    plt.title(title)
    plt.ylim([-0.2, l - 0.8])
    plt.yticks(range(l), params)
    plt.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o')
    plt.axvline(0, c='k')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"Sobol_order_{i}.png"))
    plt.close()


for idx in ('1', 'T'):
    plot_index(Si, problem['names'], idx, title=f"Sobol Order {idx} Sensitivity: Recycle Rate")