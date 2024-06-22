import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def v_second(m2, par):

    def utility(c2, m2):
        a2 = m2 - c2
        if a2 < 0:
            return -np.inf  # Infeasible if a2 < 0
        return (c2**(1-par.rho))/(1-par.rho) + par.nu*((a2 + par.kappa)**(1-par.rho))/(1-par.rho)
    
    return utility

def solve_v_second(par):
    m2_values = np.linspace(par.m_min, par.m_max, 100)
    v2_values = []
    c2_values = []

    for m2 in m2_values:
        utility = v_second(m2, par)
        result = minimize(lambda c2: -utility(c2, m2), x0=m2/2, bounds=[(0, m2)])
        c2_opt = result.x[0]
        v2_opt = -result.fun
        v2_values.append(v2_opt)
        c2_values.append(c2_opt)

    return v2_values, c2_values, m2_values

def v_first(m1, par, v2_function):
    rho = par.rho
    beta = par.beta
    tau = par.tau
    gamma = par.gamma
    ybar = par.ybar
    r = par.r
    p = par.p
    Delta = par.Delta

    def expected_v2(a1, s):
        m2 = (1 + r) * a1 + ybar + gamma * s
        v2_no_risk = v2_function(m2)
        return p * v2_function(m2 + Delta) + (1 - p) * v2_function(m2 - Delta)
    
    def utility(c1, m1, s):
        a1 = m1 - c1 - tau * s
        if a1 < 0:
            return -np.inf  # Infeasible if a1 < 0
        return (c1**(1-rho))/(1-rho) + beta * expected_v2(a1, s)
    
    return utility

def solve_v_first(par, v2_function):
    m1_values = np.linspace(par.m_min, par.m_max, 100)
    v1_values = []
    c1_opt_values = []
    s_opt_values = []

    for m1 in m1_values:
        utility = v_first(m1, par, v2_function)
        results = []
        for s in [0, 1]:
            result = minimize(lambda c1: -utility(c1, m1, s), x0=m1/2, bounds=[(0, m1-par.tau*s)])
            c1_opt = result.x[0]
            v1_opt = -result.fun
            results.append((v1_opt, c1_opt, s))
        
        best_result = max(results, key=lambda x: x[0])
        v1_values.append(best_result[0])
        c1_opt_values.append(best_result[1])
        s_opt_values.append(best_result[2])

    return v1_values, c1_opt_values, m1_values, s_opt_values

def solve_model(par):
    # Solve second period problem
    v2_values, c2_values, m2_values = solve_v_second(par)

    # Interpolate v2 function for use in first period
    v2_function = interp1d(m2_values, v2_values, kind='linear', fill_value="extrapolate")

    # Solve first period problem
    v1_values, c1_opt_values, m1_values, s_opt_values = solve_v_first(par, v2_function)

    return v1_values, c1_opt_values, m1_values, s_opt_values, v2_values, c2_values, m2_values

def plot_results(m1_values, v1_values, c1_opt_values, s_opt_values, m2_values, v2_values, c2_values):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(m1_values, v1_values, label='$v_1(m_1)$')
    plt.plot(m2_values, v2_values, label='$v_2(m_2)$')
    plt.xlabel('$m_1$')
    plt.ylabel('$v_1(m_1)$')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(m1_values, c1_opt_values, label='$c_1^*(m_1)$')
    plt.plot(m2_values, c2_values, label='$c_2^*(m_2)$')
    plt.xlabel('$m_1$')
    plt.ylabel('$c_1^*(m_1)$')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(m1_values, s_opt_values, label='$\mathbb{I}^{s*}(m_1)$')
    plt.xlabel('$m_1$')
    plt.ylabel('$s^*(m_1)$')
    plt.legend()

    plt.tight_layout()
    plt.show()