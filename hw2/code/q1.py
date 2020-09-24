# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 4, 2020
#
# @brief      Polynomial interpolation using Newton's divided differences. 
# @notes     

#
# Includes ---------------------------------------------------------------------
import argparse
from math import cos
import numpy as np

#
# Problem implementation -------------------------------------------------------
def ComputeDivDiff(X, Y):
  """
    Computes the divided difference table.
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
  dd = Y
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
  dd = ComputeDivDiff(X, Y)
  N = dd.shape[0]
  value = dd[0]
  # print(dd)
  for i in range(1, N):
    a = dd[i]
    for j in reversed(range(i)):
      a *= (x - X[j])
    value += a
  return value
    
  
#
# Helper methods ---------------------------------------------------------------
def CosPi(x, eps=0.00001):
  c = cos(np.pi * x)
  return c if c > eps else 0.0

def ComputeX(n):
  return np.asarray([(i * 2 / n) - 1 for i in range(n)])

def ComputeF(x):
  return 2 / (1+ 9 * (x **2))
#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  # parser = argparse.ArgumentParser()
  # # Use it to verify PA == LDU and Ax == b
  # parser.add_argument("--do_assert", help="Asserts that PA == LDU and Ax == b", 
  #   action="store_true", default=False)
  # # If testing a sample from file, add --from-file --path /path/to/file
  # parser.add_argument("--random", help="Reading the input from a file", 
  #   action="store_true", default=False)
  # args = parser.parse_args()
  
  # (a)
  # X = np.array([5.0, 6.0, 9.0, 11.0], dtype=np.float32)
  # Y = np.array([12.0, 13.0, 14.0, 16.0], dtype=np.float32)
  # x = 7
  # X = np.array([0.0, 1.0, -1.0], dtype=np.float32)
  # Y = np.array([1.0, 0.0, 4.0], dtype=np.float32)
  
  # (b)
  # x = 3 / 10
  # X = np.array([0, 1/8, 1/4, 3/8, 1/2], dtype=np.float32)
  # Y = np.asarray([CosPi(x) for x in X])
  # print(f"X: {X}\nY: {Y}\nx:{x}\ny_interp: {InterpolateValue(X, Y, x)}\n")
  # print(CosPi(x))
  
  # (c)
  # x = 0.07
  # N = [2, 4, 40]
  # for n in N:
  #   X = ComputeX(n)
  #   Y = np.asarray([ComputeF(x) for x in X])
  #   print("----------------------------")
  #   print(f"X: {X}\nY: {Y}\nx: {x}\ny_interp: {InterpolateValue(X, Y, x)}\n")
  # print(f"Actual value when x: {x} is y: {ComputeF(x)}")
  
  # (d)
  N = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 40]
  x = 0.07
  fx_truth = ComputeF(x)
  En = np.zeros((len(N), 1), dtype=np.float32) 
  for i, n in enumerate(N):
    X = np.linspace(-1, 1, n)
    Y = np.asarray([ComputeF(x) for x in X])
    px = InterpolateValue(X, Y, x)
    En[i] = abs(fx_truth - px)
    print("----------------------------")
    print(f"n: {n}\nX: {X}\nY: {Y}\nfx: {fx_truth}\npx: {px}\nEn: {En[i]}")
  
  print(En)