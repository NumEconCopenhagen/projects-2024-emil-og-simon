from scipy import optimize
from types import SimpleNamespace
import numpy as np
import sympy as sm

class CournotModelClass():
    
    def __init__(self):

        par = self.par = SimpleNamespace()

        par.a = 1
        par.b = 1
        par.mc = 1

        # a. preferences

    def sympy_solve():
        
        a = sm.symbols('a')
        b = sm.symbols('b')
        mc = sm.symbols('MC')
        q1 = sm.symbols('q1')
        q2 = sm.symbols('q2')

        p = a-b*q1-b*q2

        eq = sm.Eq((p-mc)*q1,(p-mc)*q2)
        sol_q1 = sm.solve(eq,q1)[0]
        sol_q2 = sm.solve(eq,q2)[0]
        return sol_q1, sol_q2
    
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