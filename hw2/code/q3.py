# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 4, 2020
#
# @brief      Newton's method on finding roots
# @notes     

#
# Includes ---------------------------------------------------------------------
import argparse
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

class Method(Enum):
  NEWTON = 0
  BISECTION = 1

#
# Problem implementation -------------------------------------------------------
def BisectionMethod(f, a, b, eps=0.000001):
  """
    Bisection Method to find root of f(x) given a, b
    Inputs
    ------
      f        : Function that we want to find a root
      a        : Guess x_a
      b        : Guess x_b
      eps      : Closest value at which we can stop iterating
    Outputs
    -------
      x        : Calculated root value
  """
  fa = f(a)
  fb = f(b)
  assert fa * fb < 0, "f(a) and f(b) have equal signs"
  
  c = (a + b) / 2
  fc = f(c)
  
  while not np.abs(np.isclose(fc, 0.0, eps)):
    fa, fb = f(a), f(b)
    if fc * fa > 0:
      a = c
    else:
      b = c
      
    c = (a + b) / 2
    fc = f(c)
  return c

def NewtonsMethod(f, df, x, max_iter=10, eps=0.000001):
  """
    Newton's Method algorithm to find a root of f(x). If the method fails to 
    converge, it returns None
    
    Inputs
    ------
      f        : Function that we want to find a root
      df       : Derivative of the function f
      x        : Starting guess of the root x
      max_iter : Number of max iterations before using another approach
      eps      : Closest value at which we can stop iterating
    Outputs
    -------
      x        : Calculated root value or None (if fails)
  """
  n = 0
  solution_found = False
  
  # Check if starting point is bad
  df_x = df(x)
  if np.abs(np.isclose(df_x, 0.0)):
    print(f"Newton: df(x) is {df_x}. Solution cannot be found")
  
  while not solution_found and n < max_iter:
    
    # Check if solution has been found 
    if np.abs(np.isclose(f(x), eps)):
      solution_found = True
    else:
      # Make update
      n += 1
      x_ = x - f(x) / df(x)
      
      # Failure analysis
      df_x = df(x_)
      if np.isclose(df_x, 0.0):
        print(f"Newton: df(x) is {df_x}. Solution cannot be found")
    
      x = x_
    
  # Method did not converge
  if not solution_found:
    print(f"Newton: Max number of iterations reached: {max_iter}")
    return None
    
  return x
  
#
# Helper methods ---------------------------------------------------------------
def Tan(x):
  return np.tan(x)

def dTan(x):
  return 1. / (np.cos(x) ** 2)

def SelectX(f, df, x = np.arange(-100, 100), method=Method.NEWTON):
  plt.figure()
  plt.plot(x, f(x))
  plt.show()

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  
  x = np.pi / 2
  a = -0.1
  b = 0.1

  # SelectX(Tan, dTan)
  
  # x_root = NewtonsMethod(Tan, dTan, x)
  # if x_root == None:
  #   print("Could not find through Newton's. Using Bisection instead")
  #   x_root = BisectionMethod(Tan, a, b)
    
  # print(f"root at: {x_root}")

   
  
  