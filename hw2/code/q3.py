# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       October 15, 2020
#
# @brief      Newton's method on finding roots
# @notes     

#
# Includes ---------------------------------------------------------------------
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

#
# Problem implementation -------------------------------------------------------
def NewtonsMethod(f, df, x, max_iter=100, eps=0.00001):
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
  y = f(x)
  dy = df(x)
  
  for i in range(max_iter):
    
    # First, check if we have df(x) is not close to 0. If it is zero, we can't
    # find a solution.
    if np.isclose(df(x), 0.0, eps):
      print(f"Solution can't be found df(x) = {df(x)}")
      return None
    
    # Find new guess
    x_ = x - f(x) / df(x)
    
    # Check if we found a root 
    if np.abs(np.isclose(f(x), 0.0, eps)):
      return x_
    
    # If not, update 
    x = x_
  
  # print(f"Newton did not converge")
  return None
  
#
# Helper methods ---------------------------------------------------------------
def f(x):
  return x - np.tan(x)

def df(x):
  return 1 - 1. / (np.cos(x) ** 2)

def FindRoot(f, df, low_val, high_val):
  # Find x_low
  # x_guess = np.random.uniform(low=low_val, high=high_val)
  rang = np.arange(low_val, high_val, 0.05)
  for x_guess in rang:
    x_root = NewtonsMethod(f, df, x_guess)
    if x_root != None:
      print(f"Using x_guess: {x_guess}, found x_root: {x_root}")
      return x_root
  
  
#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  # Regions of convergence to start Newton's
  x_low_guess_range = [13.5, 14.5]
  x_high_guess_range = [16.5, 17.5]
  
  x_low = FindRoot(f, df, x_low_guess_range[0], x_low_guess_range[1])
  x_high = FindRoot(f, df, x_high_guess_range[0], x_high_guess_range[1])
  
  if x_low == None:
    print(f"Could not find x_low, used range: {x_low_guess_range}")
  elif x_high == None:
    print(f"Could not find x_low, used range: {x_high_guess_range}")
  
  print(f"Interval [x_low, x_high] is [{x_low}, {x_high}]")
    