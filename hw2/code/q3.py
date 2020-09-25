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

#
# Problem implementation -------------------------------------------------------
def Bisection():
  return None

def NewtonsMethod(f, df, x, max_iter=10, eps=0.000001):
  """
    Newton's Method algorithm to find a root of f(x). If the method fails to 
    converge, it will use Bisection method. 
    
    Inputs
    ------
      f        : Function that we want to find a root
      df       : Derivative of the function f
      x        : Starting guess of the root x
      max_iter : Number of max iterations before using another approach
      eps      : Closest value at which we can stop iterating
    Outputs
    -------
      x        : Calculated root value
  """
  n = 0
  solution_found = False
  
  # Check if starting point is bad
  df_x = df(x)
  if np.abs(np.isclose(df_x, 0.0)):
    print(f"Newton: df(x) is {df_x}. Solution cannot be found")
    x = Bisection()
  
  while not np.abs(np.isclose(f(x), 0.0)) and n < max_iter:
    n += 1
    x_ = x - f(x) / df(x)
    
    # Failure analysis
    df_x = df(x_)
    if np.isclose(df_x, 0.0):
      print(f"Newton: df(x) is {df_x}. Solution cannot be found")
      x = Bisection()
    
    # Continue
    x = x_
  
  # Method did not converge
  if n == max_iter and not solution_found:
    print(f"Newton: Max number of iterations reached: {max_iter}")
    x = Bisection()
    
  return x
  
#
# Helper methods ---------------------------------------------------------------
def Tan(x):
  return np.tan(x)

def dTan(x):
  return 1. / (np.cos(x) ** 2)

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  
  x = np.pi / 2
  print(f"root at: {NewtonsMethod(Tan, dTan, x)}")
   
  
  