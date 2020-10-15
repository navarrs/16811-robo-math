# @author     Ingrid Navarro
# @andrewid   ingridn
# @date       Oct 9, 2020
#
# @brief      Hw3 - Q4 Plane fitting

#
# Includes ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import q2
import random
from enum import Enum


class SortDirection(Enum):
    NONE = -1
    X_ASC = 0
    X_DES = 1
    Y_ASC = 2
    Y_DES = 3
    Z_ASC = 4
    Z_DES = 5


class Axis(Enum):
    X = 0
    Y = 1
    Z = 2

#
# Problem implementation -------------------------------------------------------


def EstimatePlane(P):
    """
      Estimates a plane using SVD by finding the eigenvector with the lowest
      values corresponding to the eigenvalue with the lowest value as well.
      The code estimates n = (a, b, c) and d from the following equation:
                            ax + by + cz = d
      Inputs
      ------
        P: np.array containing the points for which we want to calculate a plane
      Outputs
      ------
        n: normal vector corresponding to the plane
        d: plane distance value
        avg_d: average distance from points to plane
    """
    c = np.mean(P, axis=0, dtype=np.float)
    P_c = P - c
    Pc = P_c.T @ P_c
    U, _, _ = scipy.linalg.svd(Pc)
    n = U[:, -1]
    d = np.mean(np.dot(P, n))
    avg_d = np.mean(abs(np.dot(P, n) - d))  # want: np.t - d = 0
    # print(np.mean(abs(np.dot(P, n) - d)))
    return n, d, avg_d


def RANSAC(points, n_samples=5, n_iters=100, dthresh=0.005):
    """
      Uses RANSAC to find the best model of a plane that fits a set of points.
      Inputs
      ------
        points: np.array containing the points for which we want to calculate a plane
        n_samples: Number of samples used to estimate a plane
        n_iters: Number plane estimation iterations
        d_thresh: Minimum aceptable distance to consider a point part of a model
      Outputs
      -------
        best_n: normal vector corresponding to the best estimated plane
        best_d: best plane distance value
        best_avg_d: best average distance from points to plane
    """
    N = len(points)

    # Best model parameters
    best_ninliers = 0
    best_n = [0, 0, 0]
    best_d = 0
    best_avg_d = 0
    for i in range(n_iters):
        # Get sample points
        samples = random.sample(range(N), n_samples)
        testp = points[samples]

        # Estimate a plane from sample points
        n, d, avg_d = EstimatePlane(testp)
        # Count the number of inliers
        D = abs(np.dot(points, n) - d)
        inliers = np.sum(D <= dthresh)

        if inliers > best_ninliers:
            # print(n)
            best_ninliers = inliers
            best_n = n
            best_d = d
            best_avg_d = avg_d

    return best_n, best_d, best_avg_d

def MultiPlane(points, sort_dir=[], tol=0.003):
    N = len(points)
    labels = -np.ones(N)
    planes = []
    for i, sortd in enumerate(sort_dir):
        if sortd == SortDirection.Y_ASC:
            # Sort pointcloud ascending on Y
            points = points[np.argsort(points[:, Axis.Y.value])]
            l, h = 0, int(N/4)
        elif sortd == SortDirection.Y_DES:
            points = points[np.argsort(points[:, Axis.Y.value])]
            l, h = -int(N/4), -1
        elif sortd == SortDirection.X_ASC:
            points = points[np.argsort(points[:, Axis.X.value])]
            l, h = 0, int(N/4)
        elif sortd == SortDirection.X_DES:
            points = points[np.argsort(points[:, Axis.X.value])]
            l, h = -int(N/4), -1
        n, d, avd = RANSAC(points[l:h], dthresh=tol)
        print(f"PLANE {i} -> n: {n} d: {d} av. d: {avd}")
        planes.append([n, d, avd])
    return planes


def ExtractSeeds(points, axis, n_lpr=100, thresh=0.05, asc=True):
    mean = np.median(points[:n_lpr, axis.value])
    # print(points[:n_lpr, axis.value])
    if asc:
      return points[points[:, axis.value] < mean + thresh]
    return points[points[:, axis.value] > mean - thresh]

def EstimateRefine(points, axis, n_lpr=100, n_iter=5, dthresh=0.005, asc=True):
    seeds = ExtractSeeds(points, axis, n_lpr, asc=asc)
    n, d, avd = EstimatePlane(seeds)
    for i in range(n_iter):
      D = abs(np.dot(points, n) - d)
      refined = points[D <= dthresh]
      n, d, avd = EstimatePlane(seeds)
    return n, d, avd

def MultiPlaneClutter(points_, sort_dir=[], n_segments=1):
    N = len(points_)
    labels = -np.ones(N)
    planes = []
    points = points_.copy()

    for i, sortd in enumerate(sort_dir):
        if sortd == SortDirection.Y_ASC:
            points = points[np.argsort(points[:, Axis.Y.value])]
            n, d, avd = EstimateRefine(points, Axis.Y)
            # Plot(points, [[n, d, avg_d]])
            
        elif sortd == SortDirection.Y_DES:
            points = points[np.argsort(points[:, Axis.Y.value])][::-1]
            n, d, avd = EstimateRefine(points, Axis.Y, n_lpr=1000, asc=False)
            # Plot(points, [[n, d, avg_d]])
        
        elif sortd == SortDirection.X_ASC:
            points = points[np.argsort(points[:, Axis.X.value])]
            n, d, avd = EstimateRefine(points, Axis.X, n_lpr=200)
            # Plot(points, [[n, d, avg_d]])
        
        elif sortd == SortDirection.X_DES:
            points = points[np.argsort(points[:, Axis.X.value])][::-1]
            n, d, avd = EstimateRefine(points, Axis.X, n_lpr=200, asc=False)
            # Plot(points, [[n, d, avg_d]])
        print(f"PLANE {i} -> n: {n} d: {d} av. d: {avd}")
        planes.append([n, d, avd])
    return planes


#
# Helper methods ---------------------------------------------------------------

def Plot(points, plane=[], colors=['b'], axis_d=[Axis.Y], title="Point Cloud"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    minx, maxx = np.min(x), np.max(x)
    miny, maxy = np.min(y), np.max(y)
    minz, maxz = np.min(z), np.max(z)

    # PLOT POINT CLOUD
    # X - positive right
    # Y - positive down
    # Z - positive forward
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    ax.scatter(x, z, y, color='r', label="point cloud", marker='.')
    ax.set_xlim(minx, maxx)
    ax.set_ylim(minz, maxz)
    ax.set_zlim(maxy, miny)

    # PLOT PLANE
    for i, p in enumerate(plane):
        n, d, _ = p
        if axis_d[i] == Axis.Y:
            x = np.linspace(minx, maxx, 5)
            z = np.linspace(minz, maxz, 5)
            X, Z = np.meshgrid(x, z)
            Y = (d - n[0]*X - n[2]*Z) / n[1]
        elif axis_d[i] == Axis.Z:
            x = np.linspace(minx, maxx, 5)
            y = np.linspace(miny, maxy, 5)
            X, Y = np.meshgrid(x, y)
            Z = (d - n[0]*X - n[1]*Y) / n[2]
        elif axis_d[i] == Axis.X:
            y = np.linspace(miny, maxy, 5)
            z = np.linspace(minz, maxz, 5)
            Y, Z = np.meshgrid(y, z)
            X = (d - n[2]*Z - n[1]*Y) / n[0]

        ax.plot_surface(X, Z, Y, color=colors[i], alpha=0.7)
    ax.legend()
    plt.show()


#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":

    display = True

    # Q4.A
    print(f"\n----------------------------------------------------------------")
    print(f"Question 4(a) - clear_table.txt")
    clear_table = np.loadtxt("../data/clear_table.txt", dtype=np.float)
    n, d, avg_d = EstimatePlane(clear_table)
    print(f"Plane normal: {n} Distance: {d}, Av. dist: {avg_d}")
    if display:
        Plot(clear_table, plane=[[n, d, avg_d]],
             title="Clear Table Point Cloud")

    # Q4.B
    print(f"\n----------------------------------------------------------------")
    print(f"Question 4(b) - cluttered_table.txt")
    cluttered_table = np.loadtxt("../data/cluttered_table.txt", dtype=np.float)
    n, d, avg_d = EstimatePlane(cluttered_table)
    print(f"Plane normal: {n} Distance: {d}, Av. dist: {avg_d}")
    if display:
        Plot(cluttered_table, plane=[[n, d, avg_d]],
             title="Cluttered Table Point Cloud")

    # Q1.C
    print(f"\n----------------------------------------------------------------")
    print(f"Question 4(c) - cluttered_table.txt")
    n, d, avg_d = RANSAC(cluttered_table, 10, 10)
    print(f"Best model normal: {n} Distance: {d}, Av. dist: {avg_d}")
    if display:
        Plot(cluttered_table, plane=[[n, d, avg_d]],
             title="Cluttered Table Point Cloud")

    # Q1.D
    print(f"\n----------------------------------------------------------------")
    print(f"Question 4(d) - clean_hallway.txt")
    clean_hallway = np.loadtxt("../data/clean_hallway.txt", dtype=np.float)
    sort_dirs = [SortDirection.Y_ASC, SortDirection.Y_DES,
                 SortDirection.X_ASC, SortDirection.X_DES]
    colors = ['b', 'g', 'm', 'c']
    axis_dir = [Axis.Y, Axis.Y, Axis.X, Axis.X]
    planes = MultiPlane(clean_hallway, sort_dirs)
    if display:
        Plot(clean_hallway, plane=planes, colors=colors, axis_d=axis_dir,
             title="Clean Hallway Point Cloud")

    # Q1.D
    print(f"\n----------------------------------------------------------------")
    print(f"Question 4(e) - cluttered_hallway.txt")
    cluttered_hallway = np.loadtxt(
        "../data/cluttered_hallway.txt", dtype=np.float)
    sort_dirs = [SortDirection.Y_ASC, SortDirection.Y_DES,
                 SortDirection.X_ASC, SortDirection.X_DES]
    colors = ['b', 'g', 'm', 'c']
    axis_dir = [Axis.Y, Axis.Y, Axis.X, Axis.X]
    planes = MultiPlaneClutter(cluttered_hallway, sort_dirs, n_segments=4)
    if display:
        Plot(cluttered_hallway, plane=planes, colors=colors, axis_d=axis_dir,
            title="Cluttered Hallway Point Cloud")
