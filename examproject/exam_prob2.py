import numpy as np
import pandas as pd
from types import SimpleNamespace
import matplotlib.pyplot as plt

## Q1
def calculate_expected_utility(par):
    # Function for the initial problem
    J = par.J
    K = par.K
    sigma = par.sigma
    v = par.v

    np.random.seed(par.seed)
    
    expected_utilities = np.zeros(J)
    realized_utilities = np.zeros(J)
    mean_epsilons = np.zeros(J)

    for j in range(J):
        epsilon = np.random.normal(0, sigma, K)
        expected_utility = v[j]
        expected_utilities[j] = expected_utility
        realized_utility = v[j] + np.mean(epsilon)
        realized_utilities[j] = realized_utility
        mean_epsilons[j] = np.mean(epsilon)
    
    return expected_utilities, realized_utilities, mean_epsilons

## Q2
def simulate_scenario(par):
    # Function for the new scenario
    J = par.J
    N = par.N
    K = par.K
    sigma = par.sigma
    v = par.v

    choices = np.zeros((N, J))
    avg_subjective_utilities = np.zeros(N)
    avg_realized_utilities = np.zeros(N)

    for k in range(K):
        for i in range(1, N + 1):
            F_i = i
            prior_utilities = np.zeros(J)
            realized_utilities = np.zeros(J)
            
            # Step 1: Draw J * F_i values of epsilon and calculate prior expected utility
            for j in range(J):
                friend_epsilons = np.random.normal(0, sigma, F_i)
                prior_utilities[j] = np.mean(friend_epsilons)
            
            # Step 2: Each person i chooses the career track with the highest expected utility
            chosen_career = np.argmax(prior_utilities)
            
            # Step 3: Draw the graduate's own noise terms and calculate the realized utility
            own_epsilon = np.random.normal(0, sigma)
            realized_utility = v[chosen_career] + own_epsilon
            
            # Store results
            choices[i-1, chosen_career] += 1
            avg_subjective_utilities[i-1] += prior_utilities[chosen_career]
            avg_realized_utilities[i-1] += realized_utility

    choices /= K
    avg_subjective_utilities /= K
    avg_realized_utilities /= K

    results = []
    for i in range(N):
        for j in range(J):
            results.append({
                'Graduate': i + 1,
                'Career Choice': f'Career {j + 1}',
                'Share Choosing': choices[i, j],
                'Avg Subjective Utility': avg_subjective_utilities[i],
                'Avg Realized Utility': avg_realized_utilities[i]
            })

    df = pd.DataFrame(results)
    
    return df

def plot_share_choosing(df):
    plt.figure(figsize=(10, 6))
    # Reshape the data for plotting
    data = df.pivot(index='Graduate', columns='Career Choice', values='Share Choosing')
    data.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Share of Graduates Choosing Each Career')
    plt.xlabel('Graduate')
    plt.ylabel('Share Choosing')
    plt.legend(title='Career Choice', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plot_share_choosing.png')
    plt.show()

def plot_avg_subjective_utility(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Graduate'].unique(), df.groupby('Graduate')['Avg Subjective Utility'].first(), marker='o')
    plt.title('Average Subjective Expected Utility')
    plt.xlabel('Graduate')
    plt.ylabel('Avg Subjective Utility')
    plt.tight_layout()
    plt.savefig('plot_avg_subjective_utility.png')
    plt.show()

def plot_avg_realized_utility(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Graduate'].unique(), df.groupby('Graduate')['Avg Realized Utility'].first(), marker='o')
    plt.title('Average Realized Utility')
    plt.xlabel('Graduate')
    plt.ylabel('Avg Realized Utility')
    plt.tight_layout()
    plt.savefig('plot_avg_realized_utility.png')
    plt.show()

def main():
    par = SimpleNamespace()
    par.J = 3
    par.N = 10
    par.K = 10000
    par.sigma = 2
    par.v = np.array([1, 2, 3])
    par.c = 1
    par.seed = 42
    
    # Run the initial problem (example)
    expected_utilities, realized_utilities, mean_epsilons = calculate_expected_utility(par)
    
    # Run the new scenario simulation
    df = simulate_scenario(par)
    
    # Plot the results
    plot_share_choosing(df)
    plot_avg_subjective_utility(df)
    plot_avg_realized_utility(df)

if __name__ == "__main__":
    main()
