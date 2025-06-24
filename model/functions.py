import math
import numpy as np
from numba import njit

@njit
def weighted_average(history, decay=0.8):
    n = len(history)
    weights = np.empty(n)
    for i in range(n):
        weights[i] = decay ** (n - i - 1)
    weights /= weights.sum()
    return (history * weights).sum()

@njit
def compute_choice(P, C0, hat_deltaC, alpha, hat_rho, logit, lambda_param, rng_val):
    U_R = P - (C0 + hat_deltaC) + alpha * hat_rho
    U_N = - alpha * hat_rho

    if logit:
        m = max(U_R, U_N)
        exp_R = math.exp((U_R - m) * lambda_param)
        exp_N = math.exp((U_N - m) * lambda_param)
        p_R = exp_R / (exp_R + exp_N)
        return rng_val < p_R
    else:
        return U_R > U_N
    
@njit
def assign_bin(x_i, y_i, bin_positions):
    min_dist = 1e10
    bin_id = -1
    for m in range(len(bin_positions)):
        x_m, y_m = bin_positions[m, 0], bin_positions[m, 1]
        dist = math.sqrt((x_i - x_m)**2 + (y_i - y_m)**2)
        if dist < min_dist:
            min_dist = dist
            bin_id = m
    return bin_id, min_dist