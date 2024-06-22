from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def labor_demand(w, p, par):
    return (p * par.A * par.gamma / w) ** (1 / (1 - par.gamma))

def production(w,p,par):
    l_star = labor_demand(w, p, par)
    return par.A * l_star ** par.gamma

def profit(w, p, par):
    l_star = labor_demand(w, p, par)
    return (1 - par.gamma) / par.gamma * w * (l_star) ** (1 / (1 - par.gamma))

def consumption(l, w, pi1, pi2, p, par):
    budget = w * l + par.T + pi1 + pi2
    return par.alpha * budget / (p + par.tau)

def utility(l, w, pi1, pi2, p1, p2, par):
    c1 = par.alpha * (w * l + par.T + pi1 + pi2) / p1
    c2 = (1 - par.alpha) * (w * l + par.T + pi1 + pi2) / (p2 + par.tau)
    return -1 * (np.log(c1 ** par.alpha * c2 ** (1 - par.alpha)) - par.nu * l ** (1 + par.epsilon) / (1 + par.epsilon))

def check_market_clearing(p1_grid, p2_grid, par, w=1):
    results = []
    for p1 in p1_grid:
        for p2 in p2_grid:
            result = optimize.minimize(lambda l: utility(l, w, profit(w, p1, par), profit(w, p2, par), p1, p2, par), 0.5, bounds=[(0, None)])
            l_star = result.x[0]
            l1_star = labor_demand(w, p1, par)
            l2_star = labor_demand(w, p2, par)
            y1_star = production(l1_star, par)
            y2_star = production(l2_star, par)
            c1_star = consumption(l_star, w, profit(w, p1, par), profit(w, p2, par), p1, par)
            c2_star = consumption(l_star, w, profit(w, p1, par), profit(w, p2, par), p2, par)
            results.append({
                'p1': p1, 'p2': p2,
                'l_star': l_star, 'l_total': l1_star + l2_star,
                'c1_star': c1_star, 'y1_star': y1_star,
                'c2_star': c2_star, 'y2_star': y2_star
            })
    return results