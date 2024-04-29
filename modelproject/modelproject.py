from scipy import optimize
from types import SimpleNamespace
import numpy as np

class CournotModelClass():
    
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

    
    #def solve_ss(alpha, c):
    
    #return result

def solve_ss(alpha, c):
    """ Example function. Solve for steady state k. 

    Args:
        c (float): costs
        alpha (float): parameter

    Returns:
        result (RootResults): the solution represented as a RootResults object.

    """ 
    
    # a. Objective function, depends on k (endogenous) and c (exogenous).
    f = lambda k: k**alpha - c
    obj = lambda kss: kss - f(kss)

    #. b. call root finder to find kss.
    result = optimize.root_scalar(obj,bracket=[0.1,100],method='bisect')
    
    return result