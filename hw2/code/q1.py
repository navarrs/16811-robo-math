# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 24, 2020
#
# @brief      Polynomial interpolation using divided differences approach.       

#
# Includes ---------------------------------------------------------------------
import argparse
from math import cos
import numpy as np
import matplotlib.pyplot as plt

#
# Problem implementation -------------------------------------------------------
def ComputeDivDiff(X, Y):
  """
    Computes the divided difference coefficients. 
    Inputs
    ------
      X: an (N, 1) array of values
      Y: an (N, 1) array of values
    Outputs
    -------
      dd: an (N, 1) the coefficients of the divided difference
  """
  assert X.shape == Y.shape, \
    f"Shapes of X:{X.shape} and Y:{Y.shape} don't match"
  N = X.shape[0]
  dd = Y.copy()
  # Compute the divided differences 
  for i in range(1, N):
    for j in range(N-1, i-1, -1):
      # print(i, j)
      dd[j] = (dd[j] - dd[j-1]) / (X[j] - X[j-i])
  # Full table
  # dd = np.zeros((N, N), dtype=np.float32)
  # dd[:, 0] = Y
  # for n in range(1, N):
  #   for i in range(N-n):
  #     dd[i, n] = (dd[i+1, n-1] - dd[i, n-1]) / (X[n+i] - X[i])
  # return dd[0]
  return dd

def InterpolateValue(X, Y, x):
  """
    Interpolates a value at x as:
      pn(x) = A0 + A1(x- x0) + A2(x - x0)(x - x1) + ... + An(x - x0)...(x - xn)
    using divided differences to obtain the coefficients Ai (i=0,...,n)
    Inputs
    ------
      X        : (N, 1) array with x values
      Y        : (N, 1) array with y values
      x_test   : Value at which we want to estimate f(x)
    Outputs
    -------
      y_interp : Estimated f(x) at x
  """
  dd = ComputeDivDiff(X, Y)
  N = dd.shape[0]
  y_interp = dd[0]
  # print(dd)
  for i in range(1, N):
    a = dd[i]
    for j in reversed(range(i)):
      a *= (x - X[j])
    y_interp += a
  return y_interp
    
#
# Helper methods ---------------------------------------------------------------
def CosPi(x, eps=0.0000001):
  """ f(x) from question 1 (b) """
  c = np.cos(np.pi * x)
  return c if c > eps else 0.0

def ComputeX(n):
  """ xi from question 1 (c) and (d) """
  return np.asarray([(i * 2 / n) - 1 for i in range(n+1)])

def ComputeF(x):
  """ f(x) from question 1 (c) and (d) """
  return 2 / (1 + 9*(x**2))

def Test(X, Y, x_test, y_truth = None, display_all=True):
  """
    Estimates f(x) at x_test given X, Y.
    Inputs
    ------
      X       : (N, 1) array with x values
      Y       : (N, 1) array with y values
      x_test  : Value at which we want to estimate f(x)
      y_truth : Actual value of f(x) 
    Outputs
    -------
      err     : Error value between f(x) and p(x) 
  """
  err = None
  print(f"X: {X}\nY: {Y}")
  
  y_interp = InterpolateValue(X, Y, x_test)
  
  # Compute error if y_truth was given
  if np.all(y_truth) != None:
    err = np.abs(y_truth - y_interp)
    
  if display_all:
    print(f"x:\n{x}")
    print(f"y_interp:\n{y_interp}")
    print(f"y_truth:\n{y_truth}\nerror: {err}")
  
  return err, y_interp
  

def Plot(X, Y, x, y, n):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title(f'Interpolation with {n} points')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  
  plt.plot(X, Y, color='r', label='actual')
  plt.plot(x, y, color='b', label='interpolated')
  
  plt.legend(loc='upper left')
  plt.show()
 
#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  
  # Q1.A
  print(f"\n------------------------------------------------------------------")
  print(f"Question 1 (a) (example) - Interpolating f(x) = (x - 1)^2")
  # Interpolating f(x) = (x - 1)^2
  X = np.array([0.0, 1.0, -1.0], dtype=np.float32)
  Y = np.array([1.0, 0.0, 4.0], dtype=np.float32)
  x = -3
  Test(X, Y, x)
  
  # Q1.B
  print(f"\n------------------------------------------------------------------")
  print(f"Question 1 (b) - Interpolating f(x) = cos(pi*x)")
  x = 3 / 10
  y_truth = CosPi(x)
  X = np.array([0, 1/8, 1/4, 3/8, 1/2], dtype=np.float32)
  Y = np.asarray([CosPi(x) for x in X], dtype=np.float32)
  Test(X, Y, x, y_truth)

  # Q1.C 
  print(f"\n------------------------------------------------------------------")
  print(f"Question 1 (c) - Interpolating f(x) = 2 / (1 + 9x^2)")
  x = 0.07
  y_truth = ComputeF(x)
  N = [2, 4, 40]
  for n in N: 
    X = ComputeX(n)
    Y = ComputeF(X)
    print(f"\n*** With polynomial of order n={n} and nodes={len(X)}")
    Test(X, Y, x, y_truth)
  
  # Q1.D 
  print(f"\n------------------------------------------------------------------")
  print(f"Question 1 (d) - Estimating max error of f(x) = 2 / (1 + 9x^2)")
  N = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 40]
  En = np.zeros((len(N), 1), dtype=np.float32)
  delta = 0.001
  x = np.arange(-1, 1, delta)
  for i, n in enumerate(N):
    X = ComputeX(n)
    Y = ComputeF(X)
    # y_truth
    y = ComputeF(x)
    print(f"\n*** With polynomial of order n={n} and nodes={len(X)}")
    err, y_interp = Test(X, Y, x, y, display_all=False)
    # Plot
    # X_ = X.copy()
    # Y_ = Y.copy()
    # x_test_ = x.copy()
    # Plot(X_, Y_, x_test_, y_interp, n)
    En[i] = err[np.argmax(err)]
    print(f"*** Max error={En[i]}")