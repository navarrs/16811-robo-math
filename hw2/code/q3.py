# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 4, 2020
#
# @brief      Newton's method on finding roots
# @notes     

#
# Includes ---------------------------------------------------------------------
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

#
# Global parameters ------------------------------------------------------------
global x_guess
x_guess = 0

global y_truth 
y_truth = 0

global a
a = 0

global b
b = 0

#
# Problem implementation -------------------------------------------------------
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
    if np.abs(np.isclose(f(x), 0.0, eps)):
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

def FindRoot(f, df, low_val, high_val):
  # Find x_low
  x_guess = np.random.uniform(low=low_val, high=high_val)
  y_truth = f(x_guess)
  print(f"x_guess: {x_guess} with y_truth: {y_truth}")
  x_root = NewtonsMethod(f, df, x_guess)
  return x_root

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  n_tests = 1
  
  # Regions of convergence to start Newton's
  x_low_guess_range = [11.3, 13.8]
  x_high_guess_range = [14.3, 16.8]
  
  for n in range(n_tests):
    print(f"*** Test {n+1}/{n_tests}")  
    # Compute lower root < 15
    x_low = FindRoot(Tan, dTan, x_low_guess_range[0], x_low_guess_range[1])
    # Compute higher root > 15
    x_high = FindRoot(Tan, dTan, x_high_guess_range[0], x_high_guess_range[1])
    
    if x_low == None or x_high == None:
      print(f"Test {n+1} failed to converge")
      continue
    
    print(f"Interval [x_low, x_high] is [{x_low}, {x_high}]")