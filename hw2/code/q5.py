# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 26, 2020
#
# @brief      Muller's method  

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
def MullerMethod(f, f2, X, max_iter=100, eps=0.0000001):
  """
    Mullers's Method algorithm to find a root of f(x). If the method fails to 
    converge, it returns None.
    
      a = f[x0, x1, x2]
      b = f[x1, x2] + f[x0, x1, x2] (x2 - x1)
      c = f[x2]
    
    Inputs
    ------
      f        : Function that we want to find a root
      X        : Starting guesses [x0, x1, x2]
      max_iter : Number of max iterations before using another approach
      eps      : Closest value at which we can stop iterating
    Outputs
    -------
      x        : Calculated root value or None (if fails)
  """
  solution_found = False
  n = 0
  
  # Calculate coefficients a, b, c
  def ComputeCoeffs(X):
    if f2 != None:
      fx0, den = f(X[0]), f2(X[0])
      fx0 /= den
      fx1, den = f(X[1]), f2(X[1])
      fx1 /= den
      fx2, den = f(X[2]), f2(X[2])
      fx2 /= den
    else:
      fx0, fx1, fx2 = f(X[0]), f(X[1]), f(X[2])
      
    dd_01 = (fx1 - fx0) / (X[1] - X[0])
    dd_12 = (fx2 - fx1) / (X[2] - X[1])
    
    a = (dd_12 - dd_01) / (X[2] - X[0])
    b = dd_12 + a * (X[2] - X[1])
    return a, b, fx2
  
  def ComputeRoot(a, b, c):
    d = np.sqrt(b ** 2 - 4 * a * c)
    # We want to maximize the denominator
    if np.abs(b + d) > np.abs(b - d):
      return 2 * c / (b + d)
    else:
      return 2 * c / (b - d)
      
  while not solution_found:
    n += 1
    
    a, b, c = ComputeCoeffs(X)
    x = ComputeRoot(a, b, c)
    x3 = X[2] - x
    fx3 = f(x3)
    if f2 != None:
      den = f2(x3)
      fx3 /= den
  
    if np.isclose(fx3, 0.0, eps):
      solution_found = True
  
    if np.isclose(x, 0.0, eps) or n >= max_iter:
      print(f"Solution cannot be found with {n} iterations")
      return None
    X[0], X[1], X[2] = X[1], X[2], x3
  
  if not solution_found:
    return None
    
  return x3
  
#
# Helper methods ---------------------------------------------------------------
def FindRoots(poly, X, n_roots, n_tries = 15):
  roots = np.zeros((n_roots), dtype=np.complex)
  X_ = X.copy()
  roots[0] = MullerMethod(poly, None, X_)
  
  poly_den = np.poly1d([1 - roots[0]])
  for i in range(1, n_roots):
    X_ = X.copy()
    root = MullerMethod(poly, poly_den, X_)
    
    # Will try to find the following roots 
    found_root = False
    ntry = 0
    while not found_root and ntry < n_tries:
      ntry += 1
      is_repeated = False
      
      for j in range(i):
        if np.isclose(abs(root - roots[j]), 0.0):
          is_repeated = True

      if not is_repeated:
        found_root = True
  
      if is_repeated:
        # perturb the point
        X_perturbed = X.copy() + 5.0*ntry
        root = MullerMethod(poly, poly_den, X_perturbed)
    
    if not found_root and ntry == n_tries:
      print(f"Could not found root {i}")
      roots[i] = np.nan
    else:
      roots[i] = root
    poly_den *= np.poly1d([1-roots[i]])
  
  return roots

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  X = np.array([1.0, 2.0, 3.0], dtype=np.complex)
  poly = np.poly1d([1, 0, 1, 1])
  x_roots = FindRoots(poly, X, 3)
  
  print(f"Roots:\n{x_roots}")
   
  
  