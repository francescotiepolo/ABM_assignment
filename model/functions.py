import math
import numpy as np
from numba import njit

@njit(cache=True)
def weighted_average(history, decay=0.8):
    '''Compute a weighted average of a history of values with exponential decay'''
    n = len(history)
    weights = np.empty(n)
    for i in range(n):
        weights[i] = decay ** (n - i - 1)
    weights /= weights.sum()
    return (history * weights).sum()

@njit(cache=True)
def compute_choice(P, C0, hat_deltaC, alpha, hat_rho, logit, lambda_param, rng_val):
    '''Compute the choice of a household agent based on its expected utility function'''
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
    
@njit(cache=True)
def assign_bin(x_i, y_i, bin_positions):
    '''Assign a household to the nearest bin based on its coordinates'''
    min_dist = 1e10
    bin_id = -1
    for m in range(len(bin_positions)):
        x_m, y_m = bin_positions[m, 0], bin_positions[m, 1]
        dist = (x_i-x_m)**2 + (y_i-y_m)**2
        if dist < min_dist:
            min_dist = dist
            bin_id = m
    return bin_id, min_dist

@njit(cache=True)
def compute_deltas(bin_ids, Qm, Km, delt):
    '''Compute the surcharge deltas for each bin based on its current and maximum capacity'''
    n = bin_ids.shape[0]
    out = np.empty(n)
    for i in range(n):
        overload = Qm[bin_ids[i]] - Km[bin_ids[i]]
        out[i] = delt * (overload if overload > 0 else 0)
    return out

@njit(cache=True)
def bin_positions(L, M):
    '''Generate positions for M bins on a grid of size L x L'''
    n_side = int(np.ceil(np.sqrt(M)))
    positions = np.empty((M, 2), dtype=np.int64)

    xs = np.linspace(1, L - 2, n_side).astype(np.int64)
    ys = np.linspace(1, L - 2, n_side).astype(np.int64)

    count = 0
    for x in xs:
        for y in ys:
            if count < M:
                positions[count, 0] = x
                positions[count, 1] = y
                count += 1
            else:
                break
    return positions