# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 29, 2020
#
# @brief      Unicycle problem

#
# Includes ---------------------------------------------------------------------
import numpy as np
import sympy as sp
from enum import Enum
import matplotlib.pyplot as plt

class TypePath(Enum):
  UPPER = 0
  LOWER = 1

#
# Problem implementation -------------------------------------------------------
def ComputeBaricentricCoeffs(triangle, p):
  """
    Computes the baricentric coefficients. 
    Inputs
    ------
      triangle: set of 3 2D coordinates representing the triangle.
      p: Point to test if within the triangle. 
    Outputs
    -------
      bc: baricentric coeffs
  """
  pass

#
# Helper methods ---------------------------------------------------------------
def PickPaths(paths_all, n_paths=3, 
              u=np.array([5.0, 8.0]), l=np.array([8.0, 5.0])):    
  def GetPath(i):
    path = [
      paths_all[i].split(sep=' ')[:-1],
      paths_all[i+1].split(sep=' ')[:-1]
    ]
    return np.asarray(path, dtype=np.float32)

  def GetIndex():
    return np.random.randint(0, len(paths_all)/2) * 2
  
  def GetTypePath(p, n_points=10):
    mid = (p.shape[1]+1) / 2
    points = p[:, int(mid-n_points/2):int(mid+n_points//2)]
    du = np.linalg.norm(u.reshape(2, 1) - points)
    dl = np.linalg.norm(l.reshape(2, 1) - points)
    return TypePath.UPPER if du < dl else TypePath.LOWER
    
  pidx = []
  pidx.append(GetIndex())
  
  paths = []
  paths.append(GetPath(pidx[0]))
  type_path = GetTypePath(paths[0])
  
  for i in range(1, n_paths):
    # Get new index 
    found_index = False
    while not found_index:
      idx = GetIndex()
      
      if not idx in pidx:
        
        # Check if path in this index is valid 
        p = GetPath(idx)
        
        if type_path == GetTypePath(p):
          paths.append(p)
          pidx.append(idx)
          found_index = True
  
  return paths, type_path

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  
  # Q7.A
  
  
  # Q7.B
  # Computes x roots (of the resultant)
  print(f"\n------------------------------------------------------------------")
  print(f"Question 8 (b) - Algorithm to create unicycle path")
  ring_of_fire_center = [5, 5]
  f = open("paths.txt", 'r')
  paths_all = f.readlines()
  paths, type_path = PickPaths(paths_all)
  