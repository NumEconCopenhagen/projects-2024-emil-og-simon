import numpy as np
from scipy import optimize
import sympy as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets


##CODE FOR THE COURNOT DUOPOL (NUMERICAL SOLUTION)##

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

#We define a function to plot the Nash EQ for changing MC
def plot_duo(mc_duo):
    
    a = 50
    b = 3

    q_values = np.linspace(0, 15, 100)
    q1_values = [BR1_duo(q2, a, b, mc_duo) for q2 in q_values]
    q2_values = [BR2_duo(q1, a, b, mc_duo) for q1 in q_values]
    
    result_duo = optimize.root(lambda q: H_duo(q, a, b, mc_duo), [0.1, 0.1])
    q_star = result_duo.x
    q1_star, q2_star = q_star
    
    plt.figure(figsize=(8, 6))
    plt.plot(q1_values, q_values, label='Best Response Firm 1', color='blue')
    plt.plot(q_values, q2_values, label='Best Response Firm 2', color='red')
    plt.scatter(q1_star, q2_star, color='green', label='Nash Equilibrium')
    plt.plot([q1_star, q1_star], [0, q2_star], color='green', linestyle='--')
    plt.plot([0, q1_star], [q2_star, q2_star], color='green', linestyle='--')
    plt.text(q1_star, q2_star + 0.5, f'({q1_star:.2f}, {q2_star:.2f})', horizontalalignment='center')
    plt.xlim(0, None)  # Setting x-axis limit to only show positive values
    plt.ylim(0, None)  # Setting y-axis limit to only show positive values
    plt.xlabel('q1')
    plt.ylabel('q2')
    plt.title('Best Response Functions and Nash Equilibrium')
    plt.legend()
    plt.grid(True)
    plt.show()

    return plt.gcf()

def plot_profit_duo(a, b, mc_duo):

    def tot_profit_duo(q, a, b, mc_duo):
        q1, q2 = q
        return profit_1_duo(a, b, q1, q2, mc_duo) + profit_2_duo(a, b, q1, q2, mc_duo)

    result_duo = optimize.root(lambda q: H_duo(q, a, b, mc_duo), [0.1, 0.1])
    q_star = result_duo.x
    q1_star, q2_star = q_star

    profit_star = tot_profit_duo(q_star, a, b, mc_duo)

    q1_range = np.linspace(0, 20, 100)
    q2_range = np.linspace(0, 20, 100)
    q1, q2 = np.meshgrid(q1_range, q2_range)
    profit = tot_profit_duo([q1, q2], a, b, mc_duo)

    profit[profit < 0] = np.nan

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(q1, q2, profit, cmap='viridis', alpha=0.7)
    ax.scatter(q1_star, q2_star, profit_star, color='red', label='Nash Equilibrium')
    ax.set_xlabel('q1')
    ax.set_ylabel('q2')
    ax.set_zlabel('Total Profit')
    ax.set_title('Cournot Model')


    ax.text(q1_star, q2_star, profit_star, f'({q1_star:.2f}, {q2_star:.2f}, {profit_star:.2f})', color='black')

    ax.legend()
    plt.show()

    return plt.gcf()

##CODE FOR THE COURNOT OLIGOPOL (FURTHER ANALYSIS)##

# We define the profit function and Best Response function for firm 1:
def profit_1_oligo(a,b,q1,q2,q3,mc):
    return (a-b*q1-b*q2-b*q3-mc)*q1

def BR1_oligo(q2,q3,a,b,mc):
    sol_BR1_oligo = optimize.minimize(lambda q: -profit_1_oligo(a,b,q,q2,q3,mc), x0=0.1, bounds=[(0, None)])
    return sol_BR1_oligo.x[0]  

# We define the profit function and Best Response function for firm 2:
def profit_2_oligo(a,b,q1,q2,q3,mc):
    return (a-b*q1-b*q2-b*q3-mc)*q2

def BR2_oligo(q1, q3, a, b, mc):
    sol_BR2_oligo = optimize.minimize(lambda q: -profit_2_oligo(a,b,q1,q,q3,mc), x0=0.1, bounds=[(0, None)])
    return sol_BR2_oligo.x[0]

# We define the profit function and Best Response function for firm 3 that makes it and oligopoly:
def profit_3_oligo(a,b,q1,q2,q3,mc):
    return (a-b*q1-b*q2-b*q3-mc)*q3

def BR3_oligo(q1,q2,a,b,mc):
    sol_BR3_oligo = optimize.minimize(lambda q: -profit_3_oligo(a,b,q1,q2,q,mc), x0=0.1, bounds=[(0, None)])
    return sol_BR3_oligo.x[0]

# We adjust the H-function to take into account Firm 3 as it is now returning err3 aswell:
def H_oligo(q,a,b,mc):
    q1, q2, q3 = q
    err1 = BR1_oligo(q2,q3,a,b,mc)-q1
    err2 = BR2_oligo(q1,q3,a,b,mc)-q2
    err3 = BR3_oligo(q1,q2,a,b,mc)-q3
    return [err1, err2, err3]

# We use optimize.root to solve for the equilibrium quantities and extract them 
def nash_eq_oligo(a, b, mc_oligo):
    result_oligo = optimize.root(lambda q: H_oligo(q,a,b,mc_oligo), [0.1, 0.1, 0.1])

    q_star_oligo = result_oligo.x
    q1_star_oligo, q2_star_oligo, q3_star_oligo = q_star_oligo
    total_output_oligo = q1_star_oligo + q2_star_oligo + q3_star_oligo

    print(f'Nash Equilibrium for mc1 = mc2 = mc3 = {mc_oligo:.0f}')
    print(f'q1* = {q1_star_oligo:1.3f}')
    print(f'q2* = {q2_star_oligo:1.3f}')
    print(f'q3* = {q3_star_oligo:1.3f}')
    print(f'For mc = {mc_oligo:.0f}, the total output in an oligopoly with three firms is: {total_output_oligo:1.3f}')

    return q1_star_oligo, q2_star_oligo, q3_star_oligo, total_output_oligo



##CODE FOR THE COURNOT OLIGOPOL WITH INDIVIDUAL MC (FURTHER ANALYSIS)##

#We define the functions for the firm individual marginal costs
def mc1(q1):
    return 1.1+(3*q1)

def mc2(q2):
    return 5+(3*q2)

def mc3(q3):
    return 3+(1.5*q3)


#We adapt the profit functions and Best Response functions for the three firms to account for the marginal cost functions depending on q
def profit_1_mc_var(a,b,q1,q2,q3,mc1):
    return (a-(q1+q2+q3)*b-mc1)*q1

def BR1_mc_var(q2,q3,a,b,mc1):
    sol_BR1_mc_var = optimize.minimize(lambda q: -profit_1_mc_var(a,b,q,q2,q3,mc1), x0=0.1, bounds=[(0, None)])
    return sol_BR1_mc_var.x[0]  

def profit_2_mc_var(a,b,q1,q2,q3,mc2):
    return (a-(q1+q2+q3)*b-mc2)*q2

def BR2_mc_var(q1,q3,a,b,mc2):
    sol_BR2_mc_var = optimize.minimize(lambda q: -profit_2_mc_var(a,b,q1,q,q3,mc2), x0=0.1, bounds=[(0, None)])
    return sol_BR2_mc_var.x[0]

def profit_3_mc_var(a,b,q1,q2,q3,mc3):
    return (a-(q1+q2+q3)*b-mc3)*q3

def BR3_mc_var(q1,q2,a,b,mc3):
    sol_BR3_mc_var = optimize.minimize(lambda q: -profit_3_mc_var(a,b,q1,q2,q,mc3), x0=0.1, bounds=[(0, None)])
    return sol_BR3_mc_var.x[0]


#We define the H-function to take into account the individual q
def H_mc_var(q,a,b,mc1,mc2,mc3):
    q1, q2, q3 = q
    err1 = BR1_mc_var(q2,q3,a,b,mc1)-q1
    err2 = BR2_mc_var(q1,q3,a,b,mc2)-q2
    err3 = BR3_mc_var(q1,q2,a,b,mc3)-q3
    return [err1, err2, err3]


def nash_eq_mc_var(a, b):
    # We formulate the Cournot equilibrium problem as a minimization problem using the optimize.minimize function. 
    # It minimizes the sum of squares of the errors in the system of equations, while we set the bounds to include only positive solutions.
    result_mc_var = optimize.minimize(lambda q: np.sum(np.square(H_mc_var(q,a,b, mc1(q[0]), mc2(q[1]), mc3(q[2])))),
                           x0=[0.1, 0.1, 0.1], bounds=[(0, None), (0, None), (0, None)], method='nelder-mead')

    #we tried solving with the optimize.root at first but were not able to make it work and instead we used optimize.minimize as above
    #result_mc_var = optimize.root(lambda q: H_mc_var(q, a, b, mc1, mc2, mc3), [0.1, 0.1, 0.1])

    # We extract the equilibrium quantities
    q_star_mc_var = result_mc_var.x
    q1_star_mc_var, q2_star_mc_var, q3_star_mc_var = q_star_mc_var
    total_output_mc_var = q1_star_mc_var + q2_star_mc_var + q3_star_mc_var

    print("Nash Equilibrium:")
    print(f'q1* = {q1_star_mc_var:1.3f}')
    print(f'q2* = {q2_star_mc_var:1.3f}')
    print(f'q3* = {q3_star_mc_var:1.3f}')
    print(f'In an oligopoly with firm individual and variable MC, the total output is: {total_output_mc_var:1.3f}')

    return q1_star_mc_var, q2_star_mc_var, q3_star_mc_var, total_output_mc_var


##FOR CONCLUSION##

def plot_results(a, b, mc_duo, mc_oligo):
# Calculate quantities for each case.
    q1_star, q2_star, total_output = nash_eq_duo(a, b, mc_duo)
    q1_star_oligo, q2_star_oligo, q3_star_oligo, total_output_oligo = nash_eq_oligo(a, b, mc_oligo)
    q1_star_mc_var, q2_star_mc_var, q3_star_mc_var, total_output_mc_var = nash_eq_mc_var(a, b)

    # Create a bar plot
    labels = ['Duopoly', 'Oligopoly\n(without MC function)', 'Oligopoly\n(with MC function)']
    outputs = [total_output, total_output_oligo, total_output_mc_var]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, outputs, color=['blue', 'orange', 'green'])
    plt.xlabel('Market Structure')
    plt.ylabel('Total Output')
    plt.title('Comparison of Total Output Across Market Structures')
    plt.show()

    return plt.gcf()