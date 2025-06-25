from mesa.batchrunner import FixedBatchRunner, BatchRunner
from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from joblib import Parallel, delayed
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
    'num_vars': 3,
    'names': ['delta', 'alpha', 'lambda_param'],
    'bounds': [[0.1, 1.0], [0.1, 1.0], [0.1, 5.0]]
}

# OFAT Sensitivity 
replicates = 10
max_steps = 100
samples_per_param = 10

data_ofat = {}

for i, var in enumerate(problem['names']):
    values = np.linspace(*problem['bounds'][i], num=samples_per_param)

    batch = FixedBatchRunner(RecyclingModel,
                             max_steps=max_steps,
                             iterations=replicates,
                             parameters_list=[{var: val} for val in values],
                             fixed_parameters=None,
                             model_reporters={"Recycle_Rate": lambda m: np.mean([h.s for h in m.households.values()])},
                             display_progress=True)

    batch.run_all()
    df = batch.get_model_vars_dataframe()
    df[var] = np.repeat(values, replicates)
    df.to_csv(os.path.join(data_dir, f"OFAT_{var}.csv"), index=False)
    data_ofat[var] = df


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
distinct_samples = 128  # Needs to be a power of 2 for Saltelli
replicates = 10

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

param_values = saltelli.sample(problem, distinct_samples)
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
Si = sobol.analyze(problem, data_sobol['Recycle_Rate'].values, print_to_console=True)


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


for idx in ('1', '2', 'T'):
    plot_index(Si, problem['names'], idx, title=f"Sobol Order {idx} Sensitivity: Recycle Rate")