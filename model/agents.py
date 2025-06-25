from mesa import Agent
from collections import deque
import numpy as np
from .functions import compute_choice, weighted_average

class Household(Agent):
    """
    Household agent that chooses whether to recycle (s=True) or not (s=False)
    """

    def __init__(self, unique_id, model, P, C0, alpha, bin_id):
        super().__init__(unique_id, model)
        self.P = P
        self.C0 = C0
        self.alpha = alpha
        self.bin_id = bin_id
        self.lambda_param = model.lambda_param
        self.logit = model.logit  # Use logit choice model if True
        self.memory_length = model.memory_length  # Number of previous rounds the agent remembers
        self.rho_history = deque(maxlen=self.memory_length)
        self.deltaC_history = deque(maxlen=self.memory_length)
        self.decay = model.decay  # Decay factor for weighted average


        # Initialize variables
        self.s = self.random.random() < 0.5  # initial 50‐50 choice
        self.hat_rho = 0.0
        self.hat_deltaC = 0.0

        # Realized terms (after choices)
        self.rho = 0.0
        self.deltaC = 0.0

    def step(self):

        # Use average of history
        if self.rho_history:
            self.hat_rho = weighted_average(np.array(self.rho_history), decay=self.decay)
        if self.deltaC_history:
            self.hat_deltaC = weighted_average(np.array(self.deltaC_history), decay=self.decay)
        
        rng_val = self.random.random()
        self.s = compute_choice(self.P, self.C0, self.hat_deltaC, self.alpha, self.hat_rho, self.logit, self.lambda_param, rng_val)


    def advance(self):
        """
        Realize current outcomes:
          - Compute realized rho = fraction of neighbors who chose R
          - Add this agent’s choice to its bin’s Q_m (number of recyclers this period at bin m).
        """
        # 1. Realized neighbor fraction
        neigh_ids = self.model.net.get_neighbors(self.unique_id)
        if len(neigh_ids) > 0:
            neigh_choices = [self.model.households[j].s for j in neigh_ids]
            self.rho = np.mean(neigh_choices)
        else:
            self.rho = 0.0

        # 2. Contribute to bin’s Q_m if this agent recycled
        if self.s:
            bin_agent = self.model.bins[self.bin_id]
            bin_agent.Q_m += 1

        # Add to memory
        self.rho_history.append(self.rho)
        self.deltaC_history.append(self.deltaC)


class Bin(Agent):
    """
    Recycling bin agent placed on a 2D grid cell.
    Attributes:
      - K_m: capacity
      - Q_m: number of recyclers assigned this period
      - x_m, y_m: spatial coordinates
    """

    def __init__(self, unique_id, model, K_m, pos):
        super().__init__(unique_id, model)
        self.K_m = K_m
        self.Q_m = 0
        self.pos = pos

    def step(self):
        pass  # Bin has no decision to make

    def advance(self):
        pass  # Overload handled by Household.advance after Q_m increment