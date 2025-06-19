from mesa import Model
from mesa.time import SimultaneousActivation, RandomActivation
from mesa.space import MultiGrid, NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
from .agents import Household, Bin

class RecyclingModel(Model):
    """
    Mesa Model:
    - SimultaneousActivation scheduler
    - NetworkGrid for social network
    - MultiGrid for spatial bin placement
    """

    def __init__(self, N=100, L=10, M=9, k=4, beta=0.1,
                 delta=0.5, c=0.3, kappa=0.05, epsilon=0.05,
                 alpha=0.4, K_default=10, memory_length=10, logit=False, lambda_param=1.0, seed=133, activation='random', decay=0.8):
        super().__init__()
        # 1. Seed for reproducibility
        self.seed = seed if seed is not None else 0
        self.random = np.random.RandomState(self.seed)
        nx_seed = self.random.randint(1e9)

        # Store parameters
        self.num_agents = N
        self.grid_size = L
        self.num_bins = M
        self.delta = delta
        self.c = c
        self.kappa = kappa
        self.epsilon = epsilon
        self.alpha = alpha
        self.K_default = K_default
        self.memory_length = memory_length
        self.logit = logit
        self.lambda_param = lambda_param
        self.decay = decay

        # 2. Build social network with NetworkX and create grid and scheduler
        self.G = nx.watts_strogatz_graph(n=N, k=k, p=beta, seed=nx_seed)
        self.net = NetworkGrid(self.G)

        self.grid = MultiGrid(width=L, height=L, torus=False)
        self.schedule = RandomActivation(self) if activation == 'random' else SimultaneousActivation(self)


        # 3. Initialize agents
        self.households = {}
        self.bins = {}     

        # 3a. Place bins on a 3×3 subgrid within L×L
        bin_positions = [(x, y) for x in [2, 5, 8] for y in [2, 5, 8]]
        for m, (x_m, y_m) in enumerate(bin_positions):
            bin_id = N + m 
            bin_agent = Bin(unique_id=bin_id, model=self, K_m=K_default, pos=(x_m, y_m))
            self.bins[m] = bin_agent
            self.grid.place_agent(bin_agent, (x_m, y_m))
            self.schedule.add(bin_agent)

        # 3b. Initialize household attributes
        coords = [(i, j) for i in range(L) for j in range(L)]
        coords = coords[:N]
        P_vals = np.random.rand(N)
        num_champ = int(epsilon * N)
        champions = self.random.choice(N, size=num_champ, replace=False)
        for i in champions:
            P_vals[i] = 2.0  # high preference for eco‐champions

        # Compute assigned bin and base cost C0 for each household
        for i in range(N):
            x_i, y_i = coords[i]
            # find nearest bin by Euclidean distance
            dists = [np.linalg.norm(np.array((x_i, y_i)) - np.array(self.bins[m].pos))
                     for m in self.bins]
            bin_id = int(np.argmin(dists))
            dist_to_bin = dists[bin_id]
            C0_i = c + kappa * dist_to_bin
            if i in champions:
                C0_i = 0.0  # champion has no base cost

            # Create Household agent and add to scheduler
            house_agent = Household(unique_id=i,
                                    model=self,
                                    P=P_vals[i],
                                    C0=C0_i,
                                    alpha=1.0 if i in champions else alpha,
                                    bin_id=bin_id)
            self.households[i] = house_agent
            self.schedule.add(house_agent)

            # Place the household on the grid (not on bin cell)
            self.grid.place_agent(house_agent, coords[i])

            # Add to the social network grid at node i
            self.net.place_agent(house_agent, i)

        # 4. Initialize previous rho and deltaC
        for i, agent in self.households.items():
            neigh_ids = self.net.get_neighbors(i)
            if neigh_ids:
                agent.rho = np.mean([self.households[j].s for j in neigh_ids])
            else:
                agent.rho = 0.0
            agent.deltaC = 0.0  # no surcharge for first decision
            agent.hat_rho = agent.rho
            agent.hat_deltaC = 0.0

        # 5. DataCollector to track metrics each round
        self.datacollector = DataCollector(
            model_reporters={
                "Global_Recycle_Rate": lambda m: np.mean([h.s for h in m.households.values()]),
                "Average_Rho":         lambda m: np.mean([h.rho for h in m.households.values()]),
                "Overloaded_Bins":     lambda m: sum(1 for b in m.bins.values() if b.Q_m > b.K_m)
            },
            agent_reporters={
                "Strategy": lambda a: a.s if isinstance(a, Household) else None,
                "Rho":      lambda a: a.rho if isinstance(a, Household) else None,
                "DeltaC":   lambda a: a.deltaC if isinstance(a, Household) else None
            }
        )


    def step(self):

        # 1. Reset bin counts
        for bin_agent in self.bins.values():
            bin_agent.Q_m = 0

        # 2.
        self.schedule.step()

        # 3. Update surcharges and prepare for next round
        for i, agent in self.households.items():
            # Realized rho was set in agent.rho during advance()
            # Surcharge for next period:
            bin_agent = self.bins[agent.bin_id]
            overload = max(0, bin_agent.Q_m - bin_agent.K_m)
            agent.deltaC = self.delta * overload

        # 4. Collect data
        self.datacollector.collect(self)