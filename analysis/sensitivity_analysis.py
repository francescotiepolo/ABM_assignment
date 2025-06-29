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

# Set to True to use saved data for plotting, False to run new simulations
USE_SAVED_DATA_OFAT = True
USE_SAVED_DATA_SOBOL = True

# Exclude eco-champions from sensitivity analysis
EXCLUDE_CHAMPIONS = False

# Create output directory
plot_dir = os.path.join(project_root, 'figures/SA')
data_dir = os.path.join(project_root, 'data')

# Define problem for sensitivity analysis 
problem = {
    'num_vars': 10,
    'names': ['M', 'delta', 'c', 'kappa', 'epsilon', 'alpha', 'K_default', 'memory_length', 'lambda_param', 'decay'],
    'bounds': [
        [1, 25],            # M
        [0.01, 1.0],        # delta
        [0.01, 1.0],        # c
        [0.01, 1.0],        # kappa
        [0.01, 0.5],        # epsilon
        [0.01, 1.0],        # alpha
        [1, 20],            # K_default
        [1, 20],            # memory_length
        [1, 10],            # lambda_param
        [0.01, 1.0]         # decay
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
    'decay': 0.8                    # decay factor for weighted average
}

int_params = {'M', 'K_default', 'memory_length'}

# OFAT Sensitivity

replicates = 50 # Number of replicates for each parameter value
max_steps = 100 # Number of steps to run the model for each parameter value
distinct_samples = 30 # Number of distinct parameter values to sample for each parameter

def run_single_ofat_run(var, val, max_steps, fixed_params, seed=None):
    '''Run a single run with a specific parameter set and return the average recycle rate'''

    if seed is not None:
        np.random.seed(seed)

    # Create a model instance with the specified parameter set
    model_params = {var: val}
    model_params.update(fixed_params)
    model = RecyclingModel(**model_params)
    for _ in range(max_steps): # Run the model for max_steps
        model.step()
    # Get the average recycle rate over the last 20 steps
    # Use "Global_Recycle_Rate_noecochampions" if EXCLUDE_CHAMPIONS is True, otherwise use "Global_Recycle_Rate"
    recycle_series = model.datacollector.get_model_vars_dataframe()["Global_Recycle_Rate_noecochampions" if EXCLUDE_CHAMPIONS else "Global_Recycle_Rate"]
    recycle_rate = recycle_series[-20:].mean()
    return recycle_rate

data_ofat = {}

for i, var in enumerate(problem['names']): # Iterate over each parameter
    filepath = os.path.join(data_dir, f"OFAT_{var}.csv")
    
    # Check if saved data exists and load it if available (and if wanted)
    if USE_SAVED_DATA_OFAT and os.path.exists(filepath) and not EXCLUDE_CHAMPIONS:
        print(f"Loading saved data for {var}")
        data_ofat[var] = pd.read_csv(filepath)
    elif USE_SAVED_DATA_OFAT and os.path.exists(filepath.replace('.csv', '_nochamps.csv')) and EXCLUDE_CHAMPIONS:
        print(f"Loading saved data for {var} without eco-champions")
        data_ofat[var] = pd.read_csv(filepath.replace('.csv', '_nochamps.csv'))
    else:
        print(f"Running OFAT for {var}")
        values = np.linspace(*problem['bounds'][i], num=distinct_samples)
        fixed_params = {k: v for k, v in default_values.items() if k != var}

        all_dfs = []
        for val in tqdm(values, desc=f"OFAT param {var}"): # Iterate over each value of the parameter
            if var in int_params:
                val = int(val)

            # Parallelize over replicates
            results = Parallel(n_jobs=-1)(
                delayed(run_single_ofat_run)(var, val, max_steps, fixed_params, seed) for seed in range(replicates)
            )
            df_val = pd.DataFrame({
                var: val,
                "Recycle_Rate": results
            })
            all_dfs.append(df_val)

        data_ofat[var] = pd.concat(all_dfs, ignore_index=True)
        if EXCLUDE_CHAMPIONS:
            data_ofat[var].to_csv(os.path.join(data_dir, f"OFAT_{var}_nochamps.csv"), index=False)
        else:
            data_ofat[var].to_csv(os.path.join(data_dir, f"OFAT_{var}.csv"), index=False)


def plot_param_var_conf(ax, df, var, param):
    '''Plot the mean and confidence interval of a parameter against a variable'''
    x = df.groupby(var).mean().reset_index()[var]
    y = df.groupby(var).mean()[param]
    replicates = df.groupby(var)[param].count()
    err = 1.96 * df.groupby(var)[param].std() / np.sqrt(replicates)

    ax.plot(x, y, c='k')
    ax.fill_between(x, y - err, y + err)
    ax.set_xlabel(var, fontsize=18)
    ax.set_ylabel(param, fontsize=18)
    ax.tick_params(axis='both', labelsize=16)


def plot_all_ofat(data, param):
    '''Plot all OFAT results'''
    n_params = len(problem['names'])
    n_cols = 2
    n_rows = (n_params + n_cols - 1) // n_cols

    f, axs = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), constrained_layout=True)
    axs = axs.flatten()

    for i, var in enumerate(problem['names']):
        plot_param_var_conf(axs[i], data[var], var, param)

    for j in range(i + 1, len(axs)):
        f.delaxes(axs[j])

    plt.tight_layout()
    if EXCLUDE_CHAMPIONS:
        plt.savefig(os.path.join(plot_dir, f"OFAT_{param}_nochamps.png"))
    else:
        plt.savefig(os.path.join(plot_dir, f"OFAT_{param}.png"))
    plt.close()


plot_all_ofat(data_ofat, "Recycle_Rate")

# Sobol

sobol_path = os.path.join(data_dir, 'sobol_data.csv')

# Check if saved Sobol data exists and load it if available (and if wanted)
if USE_SAVED_DATA_SOBOL and os.path.exists(sobol_path) and not EXCLUDE_CHAMPIONS:
    print("Loading saved Sobol data")
    data_sobol = pd.read_csv(sobol_path)
elif USE_SAVED_DATA_SOBOL and os.path.exists(sobol_path.replace('.csv', '_nochamps.csv')) and EXCLUDE_CHAMPIONS:
    print("Loading saved Sobol data without eco-champions")
    data_sobol = pd.read_csv(sobol_path.replace('.csv', '_nochamps.csv'))
else:
    print("Running Sobol sensitivity analysis")
    D = problem['num_vars']
    distinct_samples = 512 # Must be a power of 2 for convergence
    replicates = 50 # Number of replicates for each parameter value

    print(f"Using {distinct_samples} distinct samples and {replicates} replicates")
    print(f"Number of distinct results: {distinct_samples * (D + 2)}")

    data_sobol = pd.DataFrame(columns=problem['names'] + ["Recycle_Rate", "Run"])

    def run_model(run_index, vals):
        '''Run the model with a specific set of parameter values and return the recycle rate'''
        var_dict = {k: int(v) if k in int_params else v for k, v in zip(problem['names'], vals)}
        model = RecyclingModel(**var_dict)
        for _ in range(max_steps):
            model.step()
        recycle_series = model.datacollector.get_model_vars_dataframe()["Global_Recycle_Rate_noecochampions" if EXCLUDE_CHAMPIONS else "Global_Recycle_Rate"]
        recycle_rate = recycle_series[-20:].mean()
        row = dict(zip(problem['names'], vals))
        row.update({"Recycle_Rate": recycle_rate, "Run": run_index})
        return row

    param_values = saltelli.sample(problem, distinct_samples, calc_second_order=False) # Generate parameter values using Saltelli's sampling method
    total_runs = replicates * len(param_values)

    # Prepare inputs for parallel execution
    run_inputs = [
        (r * len(param_values) + i, vals)
        for r in range(replicates)
        for i, vals in enumerate(param_values)
    ]

    # Run the model in parallel for all parameter combinations
    results = Parallel(n_jobs=-1)(  
        delayed(run_model)(run_index, vals)
        for run_index, vals in tqdm(run_inputs, desc="Running simulations")
    )

    data_sobol = pd.DataFrame(results)
    fname = 'sobol_data.csv' if not EXCLUDE_CHAMPIONS else 'sobol_data_nochamps.csv'
    data_sobol.to_csv(os.path.join(data_dir, fname), index=False)

# Run Sobol analysis
Si = sobol.analyze(problem, data_sobol['Recycle_Rate'].values, print_to_console=True, calc_second_order=False)


def plot_index(s, params, i, title=''):
    '''Plot the Sobol sensitivity indices'''
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
    if EXCLUDE_CHAMPIONS:
        plt.savefig(os.path.join(plot_dir, f"Sobol_order_{i}_nochamps.png"))
    else:
        plt.savefig(os.path.join(plot_dir, f"Sobol_order_{i}.png"))
    plt.close()


for idx in ('1', 'T'):
    if EXCLUDE_CHAMPIONS:
        plot_index(Si, problem['names'], idx, title=f"Sobol Order {idx} Sensitivity: Recycle Rate (no eco-champions)")
    else:
        plot_index(Si, problem['names'], idx, title=f"Sobol Order {idx} Sensitivity: Recycle Rate")