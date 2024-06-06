from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


class ExchangeEconomyClass():

    #INITIAL MODEL SETUP:
    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3
        par.p2 = 1
        par.N = 75

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

        # We add the constraints for the endowments
        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A

    def utility_A(self,x1A,x2A):
        
        par = self.par

        return x1A**par.alpha * x2A**(1-par.alpha)

    def utility_B(self,x1B,x2B):
        
        par = self.par

        return x1B**par.beta * x2B**(1-par.beta)

    def demand_A(self,p1):
        #since numeraire p2 = 1 we add that to the demand functions for consumer A.

        par = self.par

        x1A = par.alpha * ((p1*par.w1A + par.w2A)/p1)
        x2A = (1-par.alpha) * (p1*par.w1A + par.w2A)

        return x1A,x2A

    def demand_B(self,p1):
        #since numeraire p2 = 1 we add that to the demand functions for consumer B.

        par = self.par

        x1B = par.beta * ((p1*par.w1B + par.w2B)/p1)
        x2B = (1-par.beta) * (p1*par.w1B + par.w2B)

        return x1B,x2B
      
    
    #FOR QUESTION 1:
    def edgeworth_q1(self):
        par = self.par

        # Parameters according to the question
        N = par.N
        w1A = par.w1A
        w2A = par.w2A
        w1B = par.w1B
        w2B = par.w2B

        # We define a grids for each good between 0 and 1. N+1 to ensure inclusion of endpoints (0 and 1) while still having 75 points in between.
        grid_x1A = np.linspace(0, 1, N+1) 
        grid_x2A = np.linspace(0, 1, N+1)

        # We create lists to which we later can append the specific data points for each good.
        possible_x1A = []
        possible_x2A = []
        
        #We loop over the two goods in grids constructed above.
        for x1A in grid_x1A: 
            for x2A in grid_x2A:
                x1B = 1 - x1A # We make consumer B goods expressed by consumer A goods.
                x2B = 1 - x2A
                uA = self.utility_A(x1A, x2A) # We create the utilities for consumer A and B refering to functions above.
                uB = self.utility_B(x1B, x2B)

                if uA >= self.utility_A(w1A, w2A) and uB >= self.utility_B(w1B, w2B):
                    possible_x1A.append(x1A)
                    possible_x2A.append(x2A)

        return possible_x1A, possible_x2A
    
    def plot_edgeworth_q1(self, possible_x1A, possible_x2A):
        par = self.par

        # Total endowment to be considered
        w1bar = 1.0
        w2bar = 1.0

        # Figure set up
        fig = plt.figure(frameon=False, figsize=(6,6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        # We add the endowment (w1A = 0.8 and w2A = 0.3) as a red square and add the area of possible pareto efficient combinations
        ax_A.scatter(par.w1A, par.w2A, marker='s', color='red', label='endowment')
        ax_A.scatter(possible_x1A, possible_x2A, marker='s', color='lightblue', alpha=0.5, label='pareto combi')

        # Limits
        ax_A.plot([0, w1bar], [0, 0], lw=2, color='black')
        ax_A.plot([0, w1bar], [w2bar, w2bar], lw=2, color='black')
        ax_A.plot([0, 0], [0, w2bar], lw=2, color='black')
        ax_A.plot([w1bar, w1bar], [0, w2bar], lw=2, color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True, loc='lower left', bbox_to_anchor=(0.1, 0.1))

        plt.show()


    #FOR QUESTION 2:
    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2
    
    def market_clear_q2(self):
        par = self.par
        N = par.N

        #First we create a list of p1 according to the statement above:
        grid_p1 = np.linspace(0.5, 2.5, N+1)
        p1 = grid_p1.tolist()

        #Then we create empty lists of the error terms:
        err1 = []
        err2 = []

        #Using the check_market_clearing from our py-file, we append the eps1 and eps2 to respective error-lists above
        for i in p1:
            eps1, eps2 = self.check_market_clearing(i)
            err1.append(eps1)
            err2.append(eps2)

        # We check the values and the types
        print('First five values in err1 =', [f'{val:.5f}' for val in err1[0:5]])
        print('First five values in err2 =', [f'{val:.5f}' for val in err2[0:5]])
        print(f'Type of p1: {type(p1)}')
        print(f'Type of err1: {type(err1)}')
        print(f'Type of err2: {type(err2)}')

        # We can display in a figure using matplotlib imported earlier
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1) # We create the plot
        ax.plot(p1, err1, label='$\epsilon_{1}$', color='blue') #We add the error terms 1 and 2 as eps1 (blue) and eps 2 (green) respectively
        ax.plot(p1, err2, label='$\epsilon_{2}$', color='green')

        ax.set_title('Errors in the market clearing condition')
        ax.set_xlabel('$p_1$')
        ax.set_ylabel('$Errors$')
        ax.legend(loc='upper left')
        plt.show()


    #FOR QUESTION 3:

    def market_clearing_price_q3(self, p1_guess):
      
        # First we define a function of excess demand of good 1 given the price, p1. We could also have done the same for good 2.
        def excess_demand_good1(p1):
            eps1, _ = self.check_market_clearing(p1) #we use the function for market clearing defined for question 2.
            return eps1

        result_p1 = optimize.root(excess_demand_good1, p1_guess) #We use optimize.root to find a root (where eps1 is 0)

        if result_p1.success:
            
            market_clear_p1 = result_p1.x[0] #if it is a root we let the variable for the market clearing price, p1, take that value
            x1A, x2A = self.demand_A(market_clear_p1)
            x1B, x2B = self.demand_B(market_clear_p1)
                        
            return market_clear_p1, x1A, x2A, x1B, x2B
        else:
            raise ValueError("Market clearing price not found")


    #FOR QUESTION 4:

    def max_uA_q4a(self, grid_p1):
        par = self.par

        # We initialize the variables that we want to store later
        max_uA = float('-inf') #we create the variable to store utility higher than minus infinity
        max_uA_p1 = None
        best_x1A = None
        best_x2A = None

        # We evaluate utility for A at each point in P1 (grid_p1) by computing the quantity of each good and the corresponding utility
        for p1 in grid_p1:
            x1A = 1 - self.demand_B(p1)[0]
            x2A = 1 - self.demand_B(p1)[1]
            utilityA = self.utility_A(x1A, x2A)

            # If the calculated utility is greater than the current maximum utility (max_uA) it updates it and the corresponding price and quantities
            if utilityA > max_uA:
                max_uA = utilityA
                max_uA_p1 = p1
                best_x1A = x1A
                best_x2A = x2A

        return max_uA_p1, max_uA, best_x1A, best_x2A
    
    def max_uA_q4b(self):
        
        def obj_func(p1):
            x1B, x2B = self.demand_B(p1)
            x1A = 1 - x1B
            x2A = 1 - x2B
            return -self.utility_A(x1A, x2A)  # Negative for maximization problem

        result = optimize.minimize_scalar(obj_func, bounds=(1e-6, 100), method='bounded')

        if result.success:
            optimal_p1 = result.x
            x1B, x2B = self.demand_B(optimal_p1)
            x1A = 1 - x1B
            x2A = 1 - x2B
            max_utility_A = self.utility_A(x1A, x2A)
            return optimal_p1, max_utility_A, x1A, x2A
        else:
            raise ValueError("Optimal price not found")


    #FOR QUESTION 5:

    def marketmaker_q5a(self, possible_x1A, possible_x2A):
        
        par = self.par
        
        w1A = par.w1A
        w2A = par.w2A

        # We initialize the variables that we want to store later
        uA_max = float('-inf')  # We initialize with negative infinity for maximum utility
        best_x1A = None # 
        best_x2A = None

    # We iterate over possible combinations of x1A and x2A
        for x1 in possible_x1A:
            for x2 in possible_x2A:
                uA = self.utility_A(x1, x2)
                uB = self.utility_B(1 - x1, 1 - x2)  # we calculate utility for consumer B so B will not be worse of bellow

                # We check if current allocation (x1, x2) maximizes utility for A
                if uA >= uA_max and uB >= self.utility_B(1 - w1A, 1 - w2A):
                    uA_max = uA
                    best_x1A = x1
                    best_x2A = x2

        return best_x1A, best_x2A, uA_max

    def marketmaker_q5b(self,init_guess=[0.5, 0.5]):
        
        par = self.par

        def objective_function(x):
            x1A, x2A = x
            return -self.utility_A(x1A, x2A)

        # We define constraints
        constraints = ({'type': 'ineq', 'fun': lambda x: self.utility_A(x[0], x[1]) - self.utility_A(par.w1A, par.w2A)},
                       {'type': 'ineq', 'fun': lambda x: self.utility_B(1 - x[0], 1 - x[1]) - self.utility_B(1-par.w1A, 1-par.w2A)})
             
        # Minimize the negative utility function subject to constraints         
        result = optimize.minimize(objective_function, init_guess, bounds=([0,1],[0,1]), constraints=constraints, method='SLSQP')

        # Extract optimal allocation
        optimal_xA_q5b = result.x
        optimal_xB_q5b = [1 - result.x[0], 1 - result.x[1]]  # Here we calculate B's allocation from A's

        max_uA_q5b = self.utility_A(optimal_xA_q5b[0],optimal_xA_q5b[1])
        max_uB_q5b = self.utility_B(optimal_xB_q5b[0],optimal_xB_q5b[1])

        return max_uA_q5b, max_uB_q5b, optimal_xA_q5b, optimal_xB_q5b
    

    #FOR QUESTION 6:

    def socialplanner_q6a(self,init_guess=[0.5, 0.5]):
        
        par = self.par

        def aggr_utility_func(x):
            x1A, x2A = x
            return -(self.utility_A(x1A, x2A) + self.utility_B(1-x1A, 1-x2A))

        # We define the constraint given that we have to maximize utility subject to the supply available in the economy
        constraints = ({'type': 'ineq', 'fun': lambda x: x[0]-par.w1A+(1-x[0])-(1-par.w1A)})
        # We minimize the negative utility function subject to constraints
        result = optimize.minimize(aggr_utility_func, init_guess, constraints=constraints, bounds=[(0,1),(0,1)], method='SLSQP')

        # We extract the optimal allocation
        optimal_xA_6a = result.x
        optimal_xB_6a = [1 - result.x[0], 1 - result.x[1]]  # Here we calculate B's allocation from A's
        utility_A_6a = self.utility_A(result.x[0],result.x[1])
        utility_B_6a = self.utility_B(1-result.x[0],1-result.x[1])

        return optimal_xA_6a, optimal_xB_6a, utility_A_6a, utility_B_6a
    

    def plot_results_q6b(self,x1_q1, x2_q1, x1_q4a, x2_q4a, x1_q4b, x2_q4b, x1_q5a, x2_q5a, x1_q5b, x2_q5b,x1_q6a, x2_q6a):
        par = self.par
        
        # a. total endowment
        w1bar = 1.0
        w2bar = 1.0

        # b. figure set up
        fig = plt.figure(frameon=False,figsize=(6,6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        # A
        ax_A.scatter(par.w1A,par.w2A,marker='s',color='red',label='endowment')
        ax_A.scatter(x1_q1,x2_q1,marker='s',color='lightblue', alpha=0.5, label='pareto')
        ax_A.scatter(x1_q4a,x2_q4a,marker='s',color='yellow',label='solution 4a')
        ax_A.scatter(x1_q4b,x2_q4b,marker='x',color='green',label='solution 4b')
        ax_A.scatter(x1_q5a,x2_q5a,marker='x',color='magenta',label='solution 5a')
        ax_A.scatter(x1_q5b,x2_q5b,marker='s',color='blue',label='solution 5b')
        ax_A.scatter(x1_q6a,x2_q6a,marker='s',color='pink',label='solution 6a')

        # limits
        ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
        ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
        ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True,loc='lower left',bbox_to_anchor=(0,0));

        plt.show()