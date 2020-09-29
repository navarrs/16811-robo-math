# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 28, 2020
#
# @brief      Bivariate polynomials

#
# Includes ---------------------------------------------------------------------
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

import q5

#
# Problem implementation -------------------------------------------------------
def DetQ():
  """
    Gets the coefficients of the resultant Q matrix
  """
  from sympy.matrices import Matrix
  x = sp.symbols('x')
  Q = np.array([[2, -4, (2*x**2 - 4*x + 3), 0], 
                [0, 2, -4, (2*x**2 - 4*x + 3)], 
                [1, 2*x - 3, (x**2 - 5*x + 4), 0], 
                [0, 1, 2*x - 3, (x**2 - 5*x + 4)]])
  detQ = sp.Matrix(Q).det()
  poly = sp.Poly(detQ)
  return np.asarray(poly.all_coeffs(), dtype=np.float)

def FindRealRoots(poly, X, n_roots, eps=0.000001):  
  # Find roots of the resultant 
  poly = np.poly1d(poly)
  x_roots = q5.FindRoots(poly, X, n_roots)
  print(f"Roots:\n{x_roots}")
  
  # Get only the real ones 
  x_roots = np.where(np.isclose(abs(x_roots.imag), 0.0, eps), x_roots, 0.0)
  return x_roots.real
    
#
# Helper methods ---------------------------------------------------------------
def p_(x, y):
  return 2*x**2 + 2*y**2 - 4*x - 4*y + 3

def q_(x, y):
  return x**2 + y**2 + 2*x*y - 5*x - 3*y + 4

def Plot(roots = None, x = np.arange(0, 2, 0.01), y = np.arange(-0.5, 2, 0.01)):
  X, Y = np.meshgrid(x, y)
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title('Zero Contours with Roots')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  
  if np.all(roots) != None:
    plt.scatter(roots[:, 0], roots[:, 1], color='b', label='roots')
    plt.vlines(roots[:, 0], 0, roots[:, 1], linestyle="dashed")
    for r in roots:
      ax.annotate(f"{np.round(r, 4)}", xy=(r[0], r[1]+0.05), fontsize=8)
  
  cp = plt.contour(X, Y, p_(X, Y), levels=[0], colors='r')
  cp.collections[0].set_label('p(x, y)')
  cq = plt.contour(X, Y, q_(X, Y), levels=[0], colors='g')
  cq.collections[0].set_label('q(x, y)')
  plt.axhline(0, color='black')
  
  plt.legend(loc='upper left')
  plt.show()

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  
  # Q7.B
  # Computes x roots (of the resultant)
  print(f"\n------------------------------------------------------------------")
  print(f"Question 7 (a) - Sketch p(x, y) and q(x, y)")
  # Plot()
  
  # Q7.B
  # Computes x roots (of the resultant)
  print(f"\n------------------------------------------------------------------")
  print(f"Question 7 (b) - Finding the resultant and roots of p(x, y), q(x, y)")
  detq = DetQ()
  print(f"Resultant detQ coefficients: {detq}")
  print(f"\n\n***Find x-roots of resultant")
  X = np.array([1.0, 2.0, 3.0], dtype=np.complex)
  x_roots = FindRealRoots(detq, X, 4)
  print(f"\nReal roots x: {x_roots}")
  
  roots = np.zeros((4, 2), dtype=np.float)
  roots[:, 0] = x_roots
  
  # Compute y roots for p 
  print(f"\n\n***Find y-roots of p")
  for root in x_roots:
    if not np.isclose(root, 0.0):  
      x, y = sp.symbols('x,y')
      p = 2*x**2 + 2*y**2 - 4*x - 4*y + 3
      poly = sp.Poly(p.subs(x, root))
      poly = np.asarray(poly.all_coeffs(), dtype=np.float)
      X = np.array([0, 0.5, 1.0], dtype=np.complex)
      y_roots = FindRealRoots(poly, X, 2)
      print(f"\nReal roots y: {y_roots} for x_root: {root}")
      
      # Append roots
      for y_root in y_roots:
        if not np.any(np.isclose(np.abs(y_root - x_roots), 0.0)):
          roots = np.append(roots, [[root, y_root]], axis=0)

  # Compute y roots of q
  print(f"\n\n***Find y-roots of q")
  for root in x_roots:
    if not np.isclose(root, 0.0):  
      x, y = sp.symbols('x,y')
      q = x**2 + y**2 + 2*x*y - 5*x - 3*y + 4
      poly = sp.Poly(p.subs(x, root))
      poly = np.asarray(poly.all_coeffs(), dtype=np.float)
      X = np.array([0, 0.5, 1.0], dtype=np.complex)
      y_roots = FindRealRoots(poly, X, 2)
      print(f"\nReal roots y: {y_roots} for x_root: {root}\n")
      
  # Q7.C
  # Mark the roots
  print(f"\n------------------------------------------------------------------")
  print(f"Question 7 (c) - Mark the shared roots")
  roots = roots[~np.all(np.isclose(roots, 0.0), axis=1)]
  Plot(roots)