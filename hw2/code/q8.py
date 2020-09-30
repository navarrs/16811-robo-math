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
# def ComputeWeights(p0, p1, p2, p):
#   A = np.matrix([[1.0, 1.0, 1.0],
#                  [p0[0], p1[0], p2[0]], 
#                  [p0[1], p1[1], p2[1]]])
#   b = np.array([1.0, p[0], p[1]])
#   return np.dot(np.linalg.inv(A), b)
  

def ConstructPath(paths, weights):
  """
    Constructs a new path following this formula:
      p_new = w1*p0 + w2*p1 + w3*p2
    Inputs
    ------
      paths: Set of paths p0, p1, p2
      weights: Weight values [w0, w1, w2] corresponding to each path
    Output
    ------
      new_path: The constructed path
  """
  assert len(weights) == len(paths), "Size of paths != size of weights"
  new_path = 0
  for i, path in enumerate(paths):
    new_path += weights[i] * path
  return new_path  

#
# Helper methods ---------------------------------------------------------------
def PickPaths(paths_all, n_paths=3, u=np.array([5.0, 8.0]), l=np.array([8.0, 5.0])):    
  """
    Selects n random paths from the list and checks all of them go to the same 
    direction.
    Input
    -----
      paths_all: List of all paths
      n_paths: How many paths to extact
      u: Upper bound direction
      l: ;ower bound direction
    Outputs
    -------
      paths: Selected paths
      type_path: i.e. Upper / Lower paths selcted
  """
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

def Plot(paths, rpath, ipath, ring_of_fire, destination):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title('Unicycle problem')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_xlim(0, 12)
  ax.set_ylim(0, 12)
  
  # Initial paths
  c = ['r', 'g', 'b']
  for i, path in enumerate(paths):
    plt.plot(path[0], path[1], color=c[i], label=f'path {i}', lw=1)
  
  # Constructed path 
  plt.plot(rpath[0], rpath[1], 'm', label='new path',lw=2)
  
  # Interpolated path
  plt.plot(ipath[:, 0], ipath[:, 1], 'c-', label='interpolated path',lw=2)
  
  # Plot ring of fire
  rf = plt.Circle(ring_of_fire["center"], ring_of_fire["radius"], 
                  color='r', label='ring of fire')
  ax.add_artist(rf)
  
  
  # Plot destination 
  plt.scatter(destination[0], destination[1], color='k', s=100, label='destination')
  
  plt.legend(loc='upper left')
  plt.show()

#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":
  
  # Q7.A
  
  
  # Q7.B
  # Computes x roots (of the resultant)
  
  start_location = [[0.8, 1.8], [2.2, 1.0], [2.7, 1.4]]
  
  print(f"\n------------------------------------------------------------------")
  print(f"Question 8 (b) - Algorithm to create unicycle path")
  ring_of_fire = {"center": (5.0, 5.0), "radius": 1.5}
  destination = [8, 8]
  f = open("paths.txt", 'r')
  paths_all = f.readlines()
  paths, type_path = PickPaths(paths_all)
  # print(paths)
  
  # Construct new path 
  weights = [0.3, 0.4, 0.3]
  new_path = ConstructPath(paths, weights)
  # Plot(paths, new_path, ring_of_fire, destination)
  
  # Interpolate values 
  t_steps = np.arange(0, new_path[0].shape[0]-5, 0.1)
  p_interp = np.zeros((t_steps.shape[0], 2), dtype=np.float)
  # print(new_path.shape)
  for i, t in enumerate(t_steps):
    p = int(t)
    guessp = new_path[:, p:p+2]
    t_ = t - p
    
    p_interp[i] = guessp[:, 0]*(1-t_)+guessp[:, 1]*(t_)
    print(p_interp[i], guessp[:, 0], guessp[:, 1])
  # print(p_interp.shape) 
  # print(interp_path.shape)
  Plot(paths, new_path, p_interp, ring_of_fire, destination)