import numpy as np
import pandas as pd
from types import SimpleNamespace
import matplotlib.pyplot as plt

par = SimpleNamespace()
par.J = 3
par.N = 10
par.K = 10000

par.F = np.arange(1,par.N+1)
par.sigma = 2

par.v = np.array([1,2,3])
par.c = 1

## Q1
def simulate_q1(par):
    # First we initialize the variables in which we will store the results
    expected_u = np.zeros(par.J)
    avg_realized_u = np.zeros(par.J)
    
    epsilon = np.random.normal(0, par.sigma, size=(par.K, par.J))  # We generate epsilon   
    
    # For each j (in this case 1, 2, 3) we simulate random draws to calculate epsilon and the utilities
    for j in range(par.J):
        v_j = par.v[j]  # Accessing the j-th element of par.v
        expected_u[j] = v_j + np.sum(epsilon[:, j]) / par.K  # Calculate expected utility
        avg_realized_u[:, j] = v_j + epsilon[:, j]  # Calculate average realized utility

    # Print the results
    for j in range(par.J):
        print(f"Career choice {j + 1}:")
        print(f"  Expected Utility: {expected_u[j]}")
        print(f"  Average Realized Utility: {avg_realized_u[j]}")
    
    return expected_u, avg_realized_u

def simulate_q1_2(par, state=None):
    """Simulate the dynamic model for Q1"""
    
    if state is not None:
        np.random.set_state(state)

    # Preallocate simulation variables
    epsilon = np.empty((par.K, par.J))
    expected_u = np.zeros(par.J)
    avg_realized_u = np.zeros(par.J)

    # Draw shocks
    epsilon[:, :] = np.random.normal(0, par.sigma, size=(par.K, par.J))

    # Compute expected and realized utilities
    for j in range(par.J):
        v_j = par.v  # Assuming v_j is defined somewhere or passed as a parameter
        expected_u[j] = v_j + np.sum(epsilon[:, j]) / par.K
        avg_realized_u[j] = v_j + np.mean(epsilon[:, j])

    # Optionally, print the results
    for j in range(par.J):
        print(f"Career choice {j + 1}:")
        print(f"  Expected Utility: {expected_u[j]}")
        print(f"  Average Realized Utility: {avg_realized_u[j]}")

    return expected_u, avg_realized_u
