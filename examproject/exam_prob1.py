from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def labor_star(w, p, par):
    return (p * par.A * par.gamma / w) ** (1 / (1 - par.gamma))

def production_star(w, p, par):
    l_star = labor_star(w, p, par)
    return par.A * l_star ** par.gamma

def profit(w, p, par):
    l_star = labor_star(w, p, par)
    return (1 - par.gamma) / par.gamma * w * l_star

def c1_l(w, p1, p2, T, par, l):
    total_income = w * l + T + profit(w, p1, par) + profit(w, p2, par)
    return par.alpha * total_income / p1

def c2_l(w, p1, p2, tau, T, par, l):
    total_income = w * l + T + profit(w, p1, par) + profit(w, p2, par)
    return (1 - par.alpha) * total_income / (p2 + tau)

def utility(w, p1, p2, tau, T, par, l):
    c1 = c1_l(w, p1, p2, tau, T, par, l)
    c2 = c2_l(w, p1, p2, tau, T, par, l)
    utility_value = np.log(c1**par.alpha * c2**(1 - par.alpha)) - par.nu * (l**(1 + par.epsilon)) / (1 + par.epsilon)
    return -utility_value 