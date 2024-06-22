import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from types import SimpleNamespace

# Parameters
par = SimpleNamespace()
par.A = 1.0
par.gamma = 0.5
par.alpha = 0.3
par.nu = 1.0
par.epsilon = 2.0
par.tau = 0.0
par.T = 0.0

# Define wage rate
w = 1.0

# Firms behavior functions
def optimal_labor_demand(w, p):
    return ((p * par.A * par.gamma) / w)**(1 / (1 - par.gamma))

def optimal_output(w, p):
    ell_star = optimal_labor_demand(w, p)
    return par.A * ell_star**par.gamma

def optimal_profit(w, p):
    ell_star = optimal_labor_demand(w, p)
    pi_star = (1 - par.gamma) / par.gamma * w * (p * par.A * par.gamma / w)**(1 / (1 - par.gamma))
    return pi_star

# Consumer's utility function 
def utility(c, l):
    return np.log(c[0]**par.alpha * c[1]**(1 - par.alpha)) - par.nu * l**(1 + par.epsilon) / (1 + par.epsilon)

# Budget constraint
def budget_constraint(c, l, p1, p2, tau):
    return p1 * c[0] + (p2 + tau) * c[1] - w * l - par.T - optimal_profit(w, p1) - optimal_profit(w, p2)

# Solve Consumer problem (optimization)
def solve_consumer_problem(p1, p2, tau):
    x0 = np.array([1.0, 1.0, 1.0])  # Initial guess for c1, c2, l
    obj = lambda x: -utility(x[:2], x[2])
    cons = ({'type': 'eq', 'fun': lambda x: budget_constraint(x[:2], x[2], p1, p2, tau)})
    result = minimize(obj, x0, constraints=cons, method='SLSQP')
    c1_star, c2_star, ell_star = result.x[:2], result.x[2], result.x[2]
    return c1_star, c2_star, ell_star

# Check market clearing conditions
def check_market_clearing(p1_grid, p2_grid):
    market_clearing = np.zeros((len(p1_grid), len(p2_grid)), dtype=bool)
    for i, p1 in enumerate(p1_grid):
        for j, p2 in enumerate(p2_grid):
            c1_star, c2_star, ell_star = solve_consumer_problem(p1, p2, par.tau)
            y1_star = optimal_output(w, p1)
            y2_star = optimal_output(w, p2)
            labor_market_clearing = np.isclose(ell_star, optimal_labor_demand(w, p1) + optimal_labor_demand(w, p2))
            good_market1_clearing = np.isclose(c1_star, y1_star)
            good_market2_clearing = np.isclose(c2_star, y2_star)
            market_clearing[i, j] = labor_market_clearing and good_market1_clearing and good_market2_clearing

            print(f"p1: {p1}, p2: {p2}")
            print(f"c1_star: {c1_star}, c2_star: {c2_star}, ell_star: {ell_star}")
            print(f"y1_star: {y1_star}, y2_star: {y2_star}")
            print(f"Labor market clearing: {labor_market_clearing}")
            print(f"Good market 1 clearing: {good_market1_clearing}")
            print(f"Good market 2 clearing: {good_market2_clearing}")
            print(f"Market clearing result: {market_clearing[i, j]}")
            print("-" * 50)
    return market_clearing

