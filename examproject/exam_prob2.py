import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
par = SimpleNamespace()

# For question 1
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

# For question 2
def simulate_q2(par, set_seed):
        # We initialize containers to store results
    career = np.zeros((par.N, par.K), dtype=int)
    prior_expect = np.zeros((par.N, par.K))
    realized_u = np.zeros((par.N, par.K))

    career_shares = np.zeros((par.N, par.J))
    avg_prior_expect = np.zeros(par.N)
    avg_realized_u = np.zeros(par.N)

    # We then simulate the process for K draws and N friends
    for k in range(par.K):
        for i in range(par.N):
            F_i = i + 1
            
            # We draw the epsilon values for each friend and for the graduate him-/her-self
            epsilon_F = np.random.normal(0, par.sigma, (F_i, par.J))
            epsilon_grad = np.random.normal(0, par.sigma, par.J)
            
            # We then calculate the graduates expected utility for each career track
            expected_u = par.v + np.mean(epsilon_F, axis=0)
                    
            # We choose the career track with the highest expected utility
            j_star = np.argmax(expected_u)
            
            # Finally we store the results in the containers
            career[i, k] = j_star
            prior_expect[i, k] = expected_u[j_star]
            realized_u[i, k] = par.v[j_star] + epsilon_grad[j_star]

    # We calculate shares of graduates choosing each career track and the average expectations and realized utilities
    for i in range(par.N):
        for j in range(par.J):
            career_shares[i, j] = np.sum(career[i, :] == j) / par.K  # Calculate share directly
        avg_prior_expect[i] = np.sum(prior_expect[i, :]) / par.K  # Calculate average prior expectation
        avg_realized_u[i] = np.sum(realized_u[i, :]) / par.K

    # We set up x for the Friends for plotting
    x = np.arange(1, par.N + 1)

    # We plot share of graduates choosing each career
    plt.figure(figsize=(10, 5))
    for j in range(par.J):
        plt.bar(x + (j-1)*0.2, career_shares[:, j], width=0.2, label=f'Career {j+1}')
    plt.xlabel('Graduate i')
    plt.ylabel('Share')
    plt.title('Shares of Career Choice for Graduates')
    plt.legend()
    plt.show()

    # We plot the avg expected utility for graduates
    plt.figure(figsize=(10, 5))
    plt.bar(x, avg_prior_expect, color='lightblue')
    plt.xlabel('Graduate i')
    plt.ylabel('Average Expected Utility')
    plt.title('Average Expected Utility for Graduates')
    plt.show()

    # We plot the avg realized utility for graduates
    plt.figure(figsize=(10, 5))
    plt.bar(x, avg_realized_u, color='purple')
    plt.xlabel('Graduate i')
    plt.ylabel('Average Realized Utility')
    plt.title('Average Realized Utility for Graduates')
    plt.show()