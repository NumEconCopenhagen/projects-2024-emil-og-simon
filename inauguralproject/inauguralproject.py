from types import SimpleNamespace
import numpy as np
from scipy import optimize



def square(x):
    """ square numpy array
    
    Args:
    
        x (ndarray): input array
        
    Returns:
    
        y (ndarray): output array
    
    """
    
    y = x**2
    return y


class ExchangeEconomyClass():

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
      
      
    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2
    
    #def find_optimal_5a(self, init_guess=[0.5, 0.5]):
        #par = self.par

        #def objective_function(x):
            #x1A, x2A = x
            #return -self.utility_A(x1A, x2A)

        # We tried to define the constraints for A, B and x1A and x2A rescted to C
        #def constraint1(x):
        #return self.utility_A(x[0], x[1]) - self.utility_A(par.w1A, par.w2A)
    
        #def constraint2(x):
            #return self.utility_B(1 - x[0], 1 - x[1]) - self.utility_B(1 - par.w1A, 1 - par.w2A)
    
        #def constraint3(x):
            #x1A, x2A = x
            #return some defining constraint so that x1A and x2A are in {0, 1/N, 2/N, ..., 1}, N=75. Unsuccesful.
    
        #constraints = ({'type': 'ineq', 'fun': constraint1},
                       #{'type': 'ineq', 'fun': constraint2},
                       #{'type': 'ineq', 'fun': constraint3})
             
    # Minimize the negative utility function subject to constraints         
        #result = optimize.minimize(objective_function, init_guess, bounds=([0, 1], [0, 1]), constraints=constraints, method='SLSQP')

    # We would extract optimal allocation
        #optimal_xA_5a = result.x
        #optimal_xB_5a = [1 - result.x[0], 1 - result.x[1]]  # Calculate B's allocation from A's

        #return optimal_xA_5a, optimal_xB_5a

    def find_optimal_5b(self,init_guess=[0.5, 0.5]):
        
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
        optimal_xA_5b = result.x
        optimal_xB_5b = [1 - result.x[0], 1 - result.x[1]]  # Here we calculate B's allocation from A's

        return optimal_xA_5b, optimal_xB_5b
    

    def find_optimal_6a(self,init_guess=[0.5, 0.5]):
        
        par = self.par

        def aggr_utility_func(x):
            x1A, x2A = x
            return -(self.utility_A(x1A, x2A) + self.utility_B(1-x1A, 1-x2A))

        # We define the constraint given that we have to maximize utility subject to the supply available in the economy
        constraints = ({'type': 'ineq', 'fun': lambda x: x[0]-par.w1A+(1-x[0])-(1-par.w1A)})
        # Minimize the negative utility function subject to constraints
        result = optimize.minimize(aggr_utility_func, init_guess, constraints=constraints, bounds=[(0,1),(0,1)], method='SLSQP')

        # Extract optimal allocation
        optimal_xA_6a = result.x
        optimal_xB_6a = [1 - result.x[0], 1 - result.x[1]]  # Here we calculate B's allocation from A's
        utility_A_6a = self.utility_A(result.x[0],result.x[1])
        utility_B_6a = self.utility_B(1-result.x[0],1-result.x[1])

        return optimal_xA_6a, optimal_xB_6a, utility_A_6a, utility_B_6a