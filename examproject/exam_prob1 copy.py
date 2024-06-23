import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

#First we define the model

# Functions for the two firms
def l_star(p, w, par):
    return (p * par.A * par.gamma / w) ** (1 / (1 - par.gamma))

def y_star(l, par):
    return par.A * l ** par.gamma

def profit_star(p, w, par):
    l = l_star(p, w, par)
    y = y_star(l, par)
    return p * y - w * l

# Functions for the one consumer with utilitily as well
def c1_star(l, w, pi1, pi2, p1, par):
    return par.alpha * (w * l + par.T + pi1 + pi2) / p1

def c2_star(l, w, pi1, pi2, p2, par):
    return (1 - par.alpha) * (w * l + par.T + pi1 + pi2) / (p2 + par.tau)

def max_u(l, w, p1, p2, par):
    pi1 = profit_star(p1, w, par)
    pi2 = profit_star(p2, w, par)
    c1 = c1_star(l, w, pi1, pi2, p1, par)
    c2 = c2_star(l, w, pi1, pi2, p2, par)
    return -(np.log(c1 ** par.alpha * c2 ** (1 - par.alpha)) - par.nu * l ** (1 + par.epsilon) / (1 + par.epsilon)) # We take the negative because we are to maximize using a minimizer

#For question 1

# We set up a function to check the market clearing conditions
def check_market_clearing(p, w, par):
    p1, p2 = p

    # We optimize the consumer problem
    result = optimize.minimize(lambda l: max_u(l, w, p1, p2, par), x0=1.0, bounds=[(0, None)])
    l_star_consumer = result.x[0]

    pi1 = profit_star(p1, w, par)
    pi2 = profit_star(p2, w, par)

    c1 = c1_star(l_star_consumer, w, pi1, pi2, p1, par)
    c2 = c2_star(l_star_consumer, w, pi1, pi2, p2, par)

    l1 = l_star(p1, w, par)
    l2 = l_star(p2, w, par)

    y1 = y_star(l1, par)
    y2 = y_star(l2, par)

    # The market clearing conditions are set so we can see if they approach zero
    labor_market = l_star_consumer - (l1 + l2)
    goods_market_1 = c1 - y1
    goods_market_2 = c2 - y2

    return labor_market, goods_market_1, goods_market_2

# We create a function to plot the best combination of p1 and p2
def plot_best_combination(p1_vals, p2_vals, w, par):
    
    # We want to store all results
    results = []

    # We loop to check market clearing conditions for each combination of p1 and p2
    for p1 in p1_vals:
        for p2 in p2_vals:
            labor_market, goods_market_1, goods_market_2 = check_market_clearing([p1, p2], w, par)
                        
            result = {
                'p1': p1,
                'p2': p2,
                'labor_market': labor_market,
                'goods_market_1': goods_market_1,
                'goods_market_2': goods_market_2,
                'clearing_sum': abs(labor_market) + abs(goods_market_1) + abs(goods_market_2) # we take the absolute of the sum of the market clearings to see wich combinations of p1 and p2 brings us closest
            }
            
            results.append(result)

    # We convert the results to numpy arrays for plotting
    p1_vals_plot = np.array([res['p1'] for res in results])
    p2_vals_plot = np.array([res['p2'] for res in results])
    clearing_sums = np.array([res['clearing_sum'] for res in results])

    # We then find the result that brings us closest to market clearing in all three markets
    closest_result = min(results, key=lambda x: x['clearing_sum'])

    # We plot the results
    plt.figure(figsize=(10, 8))
    plt.scatter(p1_vals_plot, p2_vals_plot, c=clearing_sums, cmap='viridis', s=100)
    plt.colorbar(label='Sum of absolute market clearing conditions')
    plt.scatter(closest_result['p1'], closest_result['p2'], color='red', s=200, label='Closest to market clearing')
    plt.xlabel('p1')
    plt.ylabel('p2')
    plt.title('Market Clearing Conditions')
    plt.legend(loc='center left', bbox_to_anchor=(1.2, 1))
    plt.grid(True)
    plt.show()

    # We print the closest result (same as previous)
    print(f"Closest to solution: p1 = {closest_result['p1']:.3f}, p2 = {closest_result['p2']:.3f} -> "
        f"labor market = {closest_result['labor_market']:.3f}, c1 = {closest_result['goods_market_1']:.3f}, c2 = {closest_result['goods_market_2']:.3f}")

#For question 2
#We define a moderated function from the market clearing for the Walras equilibrium
def obj_walras(p, w, par):
    p1, p2 = p

    # We maximize the utility for the consumer to Consumer optimization
    result = optimize.minimize(lambda l: max_u(l, w, p1, p2, par), x0=1.0, bounds=[(0, None)])
    l_star_consumer = result.x[0]

    #What we will use to check the two markets that are clearing
    pi1 = profit_star(p1, w, par)
    pi2 = profit_star(p2, w, par)
    c1 = c1_star(l_star_consumer, w, pi1, pi2, p1, par)
    l1 = l_star(p1, w, par)
    l2 = l_star(p2, w, par)
    y1 = y_star(l1, par)
    
    # We set the market clearing conditions for Walras equilibrium
    labor_market = l_star_consumer - (l1 + l2)
    goods_market_1 = c1 - y1

    # We take the sum of the squares of the market clearing conditions to ensure simultaneous clearing and to avoid negative values
    return labor_market**2 + goods_market_1**2


#For question 3
def SW_func(p, w, par):
    p1, p2 = p

    # We optimize the consumer problem as we want to find optimal labor supply
    result = optimize.minimize(lambda l: max_u(l, w, p1, p2, par), x0=1.0, bounds=[(0, None)])
    l_star_consumer = result.x[0]

    pi1 = profit_star(p1, w, par)
    pi2 = profit_star(p2, w, par)
    c1 = c1_star(l_star_consumer, w, pi1, pi2, p1, par)
    c2 = c2_star(l_star_consumer, w, pi1, pi2, p2, par)
    l2 = l_star(p2, w, par)
    
    # We calculate utility
    U = -(np.log(c1 ** par.alpha * c2 ** (1 - par.alpha)) - par.nu * l_star_consumer ** (1 + par.epsilon) / (1 + par.epsilon))

    # Then we calculate y2 star
    y2 = y_star(l2, par)

    return -(U - par.kappa * y2)  # Negative because we are maximizing

# We set up the function to find optimal tau and T
def find_optimal_tau_T(w, par, p1_opt, p2_opt):
    
    # We calculate optimal profits and optimize consumer behavior
    pi1_opt = profit_star(p1_opt, w, par)
    pi2_opt = profit_star(p2_opt, w, par)
    result_consumer_opt = optimize.minimize(lambda l: max_u(l, w, p1_opt, p2_opt, par), x0=1.0, bounds=[(0, None)])
    l_star_consumer_opt = result_consumer_opt.x[0]

    #We have optimal consumtion
    c1_opt = c1_star(l_star_consumer_opt, w, pi1_opt, pi2_opt, p1_opt, par)
    c2_opt = c2_star(l_star_consumer_opt, w, pi1_opt, pi2_opt, p2_opt, par)

    # We can calculate tau and T
    tau_opt = (1 - (1 - par.alpha) * c2_opt / ((1 - par.alpha) * (w * l_star_consumer_opt + par.T + pi1_opt + pi2_opt)))
    T_opt = ((w * l_star_consumer_opt) + par.T + pi1_opt + pi2_opt - (1 - par.alpha) * (w * l_star_consumer_opt) - pi1_opt)

    # We print and find optimal tau and T
    print(f"Optimal tau: {tau_opt}")
    print(f"Optimal T: {T_opt}")

    return tau_opt, T_opt