# @author     Ingrid Navarro 
# @andrewid   ingridn
# @date       Sept 29, 2020
#
# @brief      Unicycle problem

#
# Includes ---------------------------------------------------------------------
import numpy as np
import sympy as sp
from scipy import linalg
from enum import Enum
import matplotlib.pyplot as plt

class TypePath(Enum):
  UPPER = 0
  LOWER = 1

#
# Problem implementation -------------------------------------------------------
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

def PickPaths(paths, p0, n_paths=3, u=np.array([5.0, 8.0]), l=np.array([8.0, 5.0])):    
  """
    Selects n random paths from the list and checks all of them go to the same 
    direction.
    Input
    -----
      paths: List of all paths
      p0: Starting location 
      n_paths: How many paths to extact
      u: Upper bound direction
      l: ;ower bound direction
    Outputs
    -------
      paths: Selected paths
      type_path: i.e. Upper / Lower paths selcted
  """
  def ComputeWeights(pi, pj, pk):
    """
      Computes the Barycentric weights.
    """
    A = np.matrix([[pi[0], pj[0], pk[0]], 
                   [pi[1], pj[1], pk[1]], 
                   [1.0,   1.0,   1.0]], dtype=np.float)
    b = np.array([p0[0], p0[1], 1.0], dtype=np.float)
    
    # Based on hw1-q3, solve for the weights using least squares
    m = A.shape[0]
    n = A.shape[1]
    U, s, V_T = linalg.svd(A)
    S = linalg.diagsvd(s, m, n)
    k = np.linalg.matrix_rank(S)
    U_T, V = np.transpose(U), np.transpose(V_T)
    S_inv = S
    for i in range(k):
      S_inv[i, i] = 1 / S_inv[i, i]
    S_inv = S_inv.transpose()
    # x = V 1/S U_T b
    x  = np.dot(np.matmul(np.matmul(V, S_inv), U_T), b)
    return x
  
  def GetTypePath(p, n=10):
    """
      Finds the direction of the path; i.e. either lower or upper path.
    """
    mid = (p.shape[1]+1) / 2
    points = p[:, int(mid-n/2):int(mid+n//2)]
    du = np.linalg.norm(u.reshape(2, 1) - points)
    dl = np.linalg.norm(l.reshape(2, 1) - points)
    return TypePath.UPPER if du < dl else TypePath.LOWER

  def FindLastPath(paths_ijk, idx):
    for i in range(2, len(idx)):
      next_path = paths[idx[i]:idx[i]+2, :]
      w = ComputeWeights(paths_ijk[0][:, 0], paths_ijk[1][:, 0], next_path[:, 0])
      if (w > 0.0).all() and np.isclose(np.sum(w), 1.0):
        return next_path, w
    return None, None
  
  # Find closest starting points in paths from the desired start point
  paths_p0 = paths[:, 0].reshape(int(paths.shape[0]/2), 2)
  dist = np.sum(abs(paths_p0-p0), axis=1)
  # dist = np.sum((paths_p0-p0)**2, axis=1)

  lidx = []
  uidx = []
  for i in np.argsort(dist, axis=0) * 2:
    if TypePath.LOWER == GetTypePath(paths[i:i+2, :]):
      lidx.append(i)
    else:
      uidx.append(i)
  
  # Check with lower paths first 
  paths_ijk = [paths[lidx[0]:lidx[0]+2, :], 
               paths[lidx[1]:lidx[1]+2, :]] 
  next_path, w = FindLastPath(paths_ijk, lidx)
  
  if np.any(next_path) == None:
    paths_ijk = [paths[uidx[0]:uidx[0]+2, :], 
                 paths[uidx[1]:uidx[1]+2, :]] 
    next_path, w = FindLastPath(paths_ijk, uidx)
    
  paths_ijk.append(next_path)

  # Compute weights 
  # print(paths_ijk[0][:, 0], paths_ijk[1][:, 0], paths_ijk[2][:, 0], w)
  
  return paths_ijk, w

def ComputeP(t, path):
  """
    Computes the interpolated value p(t) given the path
    Inputs
    ------
      t: time step at which to compute p(t)
      path: X.Y values of the path. 
  """
  total_t_steps = path.shape[1] - 2 # should be 48
  if t < 0.0:
    return path[:, 0]
  elif t > total_t_steps:
    return path[:, -1]
  
  t_ = t - int(t)
  guessp = path[:, int(t):int(t)+2]

  # Reference; https://en.wikipedia.org/wiki/Linear_interpolation
  return guessp[:, 0] * (1-t_) + guessp[:, 1] * t_

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
  # print(paths.shape)
  for i, path in enumerate(paths):
    plt.plot(path[0], path[1], color=c[i], label=f'path {i}', lw=1)
  
  # Constructed path 
  plt.plot(rpath[0], rpath[1], 'm', label='new path',lw=1)
  
  # Interpolated path
  plt.plot(ipath[0], ipath[1], 'c-', label='interpolated path',lw=2)
  
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
  
  # Q7.B - D
  # Read paths 
  paths = np.loadtxt('paths.txt', delimiter=' ')
  # p_start = np.array([[0.8, 1.8]], dtype=np.float)
  p_start = np.array([[0.8, 1.8], [2.2, 1.0], [2.7, 1.4]], dtype=np.float)
  p_dest = np.array([8.0, 8.0])
  ring_of_fire = {"center": [5.0, 5.0], "radius": 1.5}
  
  for p0 in p_start:
    # Pick starting paths and get corresponding weights
    start_paths, w = PickPaths(paths, p0)
    
    # Construct the new path
    new_path = ConstructPath(start_paths, w)
    # Plot(start_paths, new_path, ring_of_fire, p_dest)
    
    # Interpolate
    t_steps = np.arange(0, new_path.shape[1], 0.2)
    interp_path = np.zeros((2, t_steps.shape[0]), dtype=np.float)
    
    for i, ts in enumerate(t_steps):
      interp_path[:, i] = ComputeP(ts, new_path)
    
    Plot(start_paths, new_path, interp_path, ring_of_fire, p_dest)