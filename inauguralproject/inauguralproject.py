from types import SimpleNamespace



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
      
    def find_optimal_allocation(self):
        # Initialize variables to store optimal allocations
        optimal_allocations = []

        # Iterate over all possible combinations of goods for consumer A
        for x1A in range(self.par.N + 1):
            for x2A in range(self.par.N + 1):
                x1B = self.par.N - x1A
                x2B = self.par.N - x2A

                # Calculate utilities for both consumers
                uA = self.utility_A(x1A, x2A)
                uB = self.utility_B(x1B, x2B)

                # Check if allocations satisfy the conditions
                if uA >= self.utility_A(self.par.w1A, self.par.w2A) and uB >= self.utility_B(self.par.w1B, self.par.w2B):
                    optimal_allocations.append(((x1A/self.par.N, x2A/self.par.N), (x1B/self.par.N, x2B/self.par.N)))

        return optimal_allocations
      

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2