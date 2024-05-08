import numpy as np
from scipy import optimize
import sympy as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets


##FOR THE NUMERICAL PART##

# We define the profit function for firm 1
def profit_1_duo(a,b,q1,q2,mc):
        return (a-b*q1-b*q2-mc)*q1

# We define the best response function for firm 1
def BR1_duo(q2,a,b,mc):
       sol_BR1_duo = optimize.minimize(lambda q: -profit_1_duo(a,b,q,q2,mc),x0=0.1,bounds=[(0, None)])
       return sol_BR1_duo.x[0]

# We define the profit function for firm 2     
def profit_2_duo(a,b,q1,q2,mc):
        return (a-b*q1-b*q2-mc)*q2

# We define the best response function for firm 2
def BR2_duo(q1,a,b,mc):
       sol_BR2_duo = optimize.minimize(lambda q: -profit_2_duo(a,b,q1,q,mc),x0=0.1,bounds=[(0, None)])
       return sol_BR2_duo.x[0]

# W now define the h_duo function which computes the errors in Nash EQ
def H_duo(q, a, b, mc):
    q1,q2 = q
    err1 = BR1_duo(q2,a,b,mc)-q1
    err2 = BR2_duo(q1,a,b,mc)-q2
    return [err1, err2]

# We define a function to solve for the Nash EQ using a root finder
def nash_eq_duo(a, b, mc_duo):
    result_duo = optimize.root(lambda q: H_duo(q, a, b, mc_duo), [0.1, 0.1])
    q_star = result_duo.x
    q1_star, q2_star = q_star
    total_output = q1_star + q2_star

    print("Nash Equilibrium for mc1 = mc2 = {:.0f}".format(mc_duo))
    print("q1* = {:.3f}".format(q1_star))
    print("q2* = {:.3f}".format(q2_star))
    print("Total Output = {:.3f}".format(total_output))
    
    return q1_star, q2_star, total_output

def update_plot(mc_duo):
    q_values = np.linspace(0, 15, 100)
    q1_values = [BR1_duo(q2,a,b,mc) for q2 in q_values]
    q2_values = [BR2_duo(q1,a,b,mc) for q1 in q_values]
    
    result = optimize.root(lambda q: H_duo(q, a, b, mc), [0.1, 0.1])
    q_star = result.x
    q1_star, q2_star = q_star
    
    plt.figure(figsize=(8, 6))
    plt.plot(q1_values, q_values, label='Best Response Firm 1', color='blue')
    plt.plot(q_values, q2_values, label='Best Response Firm 2', color='red')
    plt.scatter(q1_star, q2_star, color='green', label='Nash Equilibrium')
    plt.plot([q1_star, q1_star], [0, q2_star], color='green', linestyle='--')
    plt.plot([0, q1_star], [q2_star, q2_star], color='green', linestyle='--')
    plt.xlim(0, None)  # Setting x-axis limit to only show positive values
    plt.ylim(0, None)  # Setting y-axis limit to only show positive values
    plt.xlabel('q1')
    plt.ylabel('q2')
    plt.title('Best Response Functions and Nash Equilibrium')
    plt.legend()
    plt.grid(True)
    plt.show()

widgets.interact(update_plot, mc=(1, 30, 1))


##FOR THE NUMERICAL PART##