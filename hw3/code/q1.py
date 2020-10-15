# @author     Ingrid Navarro
# @andrewid   ingridn
# @date       Oct 9, 2020
#
# @brief      Hw3 - Q1 Uniform Approximation

#
# Includes ---------------------------------------------------------------------
import argparse
from math import cos
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad

#
# Problem implementation -------------------------------------------------------

#
# Helper methods ---------------------------------------------------------------
def f(x):
    return np.sin(x) - 0.5

def p_uniform(x):
    return -0.5 + 0.7246*x

def integrand_uniform(x):
    return (f(x) - p_uniform(x))**2

def p_leastsq(x):
    return -0.5 + 0.774*x

def integrand_leastsq(x):
    return (f(x) - p_leastsq(x))**2

#
# Q1.c
def equationsc(v):
    """
        System of equations obtained in 1(c)
    """
    x1, x2, a, b, c = v
    return (np.sin(x1) - 1 + b*np.pi/2 -b*x1 - c*x1**2 + c*(np.pi/2)**2, 
            np.cos(x1) - b - 2*c*x1,
            np.cos(x2) - b - 2*c*x2,
            2*a + c * (np.pi**2)/2 + 1,
            np.sin(x2) - 2*a - b*x2 - b*(np.pi/2) - c*x2**2 - c*(np.pi/2)**2
      ) 
    
def Plot(X, Y, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.plot(X, Y, color='r', label='f(x)')
    plt.legend(loc='upper left')
    plt.show()


#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":

    # Q1.B
    print(f"\n------------------------------------------------------------------")
    print(f"Question 1 (b) - Plot f(x) = sin(x) - 0.5 over [-pi/2, pi/2]")
    x = np.arange(start=-np.pi/2, stop=np.pi/2, step=0.001)
    y = f(x)
    Plot(x, y, "f(x) = sin(x) - 0.5 over [-pi/2, pi/2]")

    # Q1.C
    print(f"\n------------------------------------------------------------------")
    print(f"Question 1 (c) - Best Uniform Approximation by a quadratic")
    x = fsolve(equationsc, (0.,0.,0.,0.,0.))
    print(f"a: {x[2]} b: {x[3]} c: {x[4]}")
    I = quad(integrand_uniform, -np.pi/2, np.pi/2)
    print(f"integrand: {I}, L_2 norm {np.sqrt(np.sum(I))}")
    
    # Q1.C
    print(f"\n------------------------------------------------------------------")
    print(f"Question 1 (d) - Best Least Squares Approximation")
    I = quad(integrand_leastsq, -np.pi/2, np.pi/2)
    print(f"integrand: {I}, L_2 norm {np.sqrt(np.sum(I))}")
