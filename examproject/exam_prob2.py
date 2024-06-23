import numpy as np
import pandas as pd
from types import SimpleNamespace
import matplotlib.pyplot as plt
par = SimpleNamespace()

## Q1
def simulate_q1(par):
    # First we initialize the variables in which we will store the results
    expected_u = np.zeros(par.J)
    avg_realized_u = np.zeros(par.J)
    
    epsilon = np.random.normal(0, par.sigma, size=(par.K, par.J))  # We generate epsilon   
    
    # For each j (in this case 1, 2, 3) we simulate random draws to calculate epsilon and the utilities
    for j in range(par.J):
        v_j = par.v[j]  # Accessing the j-th element of par.v
        expected_u[j] = v_j  # Calculate expected utility
        avg_realized_u[j] = v_j + np.mean(epsilon[:, j])  # Calculate average realized utility   

    # Print the results
    for j in range(par.J):
        print(f"Career choice {j + 1}:")
        print(f"  Expected Utility: {expected_u[j]}")
        print(f"  Average Realized Utility: {avg_realized_u[j]}")
    
    return expected_u, avg_realized_u