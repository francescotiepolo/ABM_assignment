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

# Define model parameters
model = RecyclingModel(
    N=100,                       # households
    L=10,                        # grid size
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

# Run for T steps
T = 500
for _ in range(T):
    model.step()

# Get model data
model_data = model.datacollector.get_model_vars_dataframe()

# Plot Recycling Rate over time
fig_dir = os.path.join(project_root, 'figures')

plt.figure(figsize=(6,4))
plt.plot(model_data["Global_Recycle_Rate"], lw=2)
plt.xlabel("Round $t$")
plt.ylabel("Fraction Recycling $R(t)$")
plt.title("Global Recycling Rate Over Time")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "recycling_rate_over_time.png"), dpi=300)
plt.show()

# Print summary statistics
final_R = model_data["Global_Recycle_Rate"].iloc[-1]
avg_last10 = model_data["Global_Recycle_Rate"].iloc[-10:].mean()
print(f"Final fraction recycling R(T) = {final_R:.3f}")
print(f"Average over last 10 rounds   = {avg_last10:.3f}")
print(f"Min fraction recycling        = {model_data['Global_Recycle_Rate'].min():.3f}")
print(f"Max fraction recycling        = {model_data['Global_Recycle_Rate'].max():.3f}")