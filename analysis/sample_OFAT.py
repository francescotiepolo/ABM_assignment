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

alpha_values = np.linspace(0.0, 1.0, 50)
steady_state_rates = []
T = 150  # steps to simulate
steady_T = 50  # number of final steps to average over
R_t = []

common_params = dict(
    N=100,
    L=10,
    M=9,
    k=4,
    beta=0.1,
    delta=0.7,
    c=0.3,
    kappa=0.05,
    epsilon=0.05,
    K_default=10,
    memory_length=100,
    logit=True,
    lambda_param=20,
    activation='simultaneous',
    decay=0.8,
    seed=133
)

for alpha in alpha_values:
    params = common_params.copy()
    params['alpha'] = alpha
    model = RecyclingModel(**params)
    Rt_series = []
    for t in range(T):
        model.step()
        # record recycling rate at each step
        R_t_now = np.mean([h.s for h in model.households.values()])
        Rt_series.append(R_t_now)
    # average over some final steps
    R_ss = np.mean(Rt_series[-steady_T:])
    steady_state_rates.append(R_ss)

# Plot
fig_dir = os.path.join(project_root, 'figures')

plt.figure()
plt.plot(alpha_values, steady_state_rates, marker='o')
plt.xlabel('Social influence weight (α)')
plt.ylabel('Steady‑state recycling rate R')
plt.title('Average R vs. α')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'steady_state_recycling_rate_vs_alpha.png'), dpi=300)
plt.show()