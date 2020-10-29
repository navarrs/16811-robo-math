# @author     Ingrid Navarro
# @andrewid   ingridn
# @date       Oct 27, 2020
#
# @brief      Hw4 - Q6 Path Optimization
# ------------------------------------------------------------------------------

#
# Includes ---------------------------------------------------------------------
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from enum import Enum

#
# Global parameters ------------------------------------------------------------
N = 101
waypoints = 300

class Cost(Enum):
    A = 0
    B = 1
    C = 2

#
# Problem implementation -------------------------------------------------------
# Q6.A
def Optimize6A(path0, obs_cost, t_step=0.1, u_optimum=0.1):

    gx, gy = np.gradient(obs_cost)

    path1 = path0.copy()
    # Do not optimize start / goal points
    for i in range(1, len(path1)-1):
        x, y = path1[i, 0], path1[i, 1]
        print(f"Optimizing point {i}: ({x},{y})")
        u = np.inf
        while u > u_optimum:
            x, y = int(path1[i, 0]), int(path1[i, 1])
            if x >= (N-1) or y >= (N-1):
                break

            grad_x = gx[x, y]
            grad_y = gy[x, y]
            # print(f"path point ({path[i, 0]}, {path[i, 1]})")
            # print(f"gradients ({grad_x}, {grad_y})")
            path1[i, 0] = path1[i, 0] - t_step * grad_x
            path1[i, 1] = path1[i, 1] - t_step * grad_y
            # print(f"path point updated ({path[i, 0]}, {path[i, 1]})")
            u = np.sqrt(grad_x**2 + grad_y**2)
            # print(f"u: {u}")

    tt = path1.shape[0]
    path1_values = np.zeros((tt, 1))
    for i in range(tt):
        path1_values[i] = obs_cost[int(np.floor(path1[i, 0])),
                                   int(np.floor(path1[i, 1]))]

    return path1, path1_values


def Optimize(path0, obs_cost, cost = Cost.A, iters=100, t_step=0.1):

    gx, gy = np.gradient(obs_cost)

    path1 = path0.copy()
    
    # Do not optimize start / goal points
    for k in range(iters):
        for i in range(1, len(path1)-1):
            x, y = path1[i, 0], path1[i, 1]
            # print(f"Optimizing point {i}: ({x},{y})")
           
            x, y = int(path1[i, 0]), int(path1[i, 1])
            if x >= (N-1) or y >= (N-1):
                break
            
            if cost == Cost.A:
                d = np.zeros(2)
            if cost == Cost.B:
                d = (path1[i] - path1[i-1])
            elif cost == Cost.C:
                d = (2 * path1[i] - path1[i-1] - path1[i+1])
                
            grad_x = 0.8 * gx[x, y] + 4 * d[0]
            grad_y = 0.8 * gy[x, y] + 4 * d[1]
            
            # print(f"path point ({path[i, 0]}, {path[i, 1]})")
            # print(f"gradients ({grad_x}, {grad_y})")
            path1[i, 0] = path1[i, 0] - t_step * grad_x
            path1[i, 1] = path1[i, 1] - t_step * grad_y
            # print(f"path point updated ({path[i, 0]}, {path[i, 1]})")

    tt = path1.shape[0]
    path1_values = np.zeros((tt, 1))
    for i in range(tt):
        path1_values[i] = obs_cost[int(np.floor(path1[i, 0])),
                                   int(np.floor(path1[i, 1]))]

    return path1, path1_values

#
# Helper Methods ---------------------------------------------------------------


def GenerateObstacleCostF(OBST, epsilon):
    obs_cost = np.zeros((N, N))
    for i in range(OBST.shape[0]):
        t = np.ones((N, N))
        t[OBST[i, 0], OBST[i, 1]] = 0
        t_cost = distance_transform_edt(t)
        t_cost[t_cost > epsilon[i]] = epsilon[i]
        t_cost = 1 / (2 * epsilon[i]) * (t_cost - epsilon[i])**2
        obs_cost = obs_cost + t_cost

    gx, gy = np.gradient(obs_cost)
    return obs_cost, gx, gy


def GenerateInitialPath(obs_cost, SX=10, SY=10, GX=90, GY=90):
    traj = np.zeros((2, waypoints))
    traj[0, 0] = SX
    traj[1, 0] = SY
    dist_x = GX-SX
    dist_y = GY-SY
    for i in range(1, waypoints):
        traj[0, i] = traj[0, i-1] + dist_x/(waypoints-1)
        traj[1, i] = traj[1, i-1] + dist_y/(waypoints-1)

    path_init = traj.T
    tt = path_init.shape[0]
    path_init_values = np.zeros((tt, 1))
    for i in range(tt):
        path_init_values[i] = obs_cost[int(np.floor(path_init[i, 0])),
                                       int(np.floor(path_init[i, 1]))]

    return path_init, path_init_values


def Plot(obs_cost, path0, path0_values, path1, path1_values,
         name='path'):
    # Plot 2D
    plt.imshow(obs_cost.T)
    plt.plot(path0[:, 0], path0[:, 1], 'ro', lw=0.5)
    plt.plot(path1[:, 0], path1[:, 1], 'go', lw=0.5)
    plt.savefig(f"../out/q6/plot2d_{name}.png")

    # Plot 3D
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    xx, yy = np.meshgrid(range(N), range(N))
    ax3d.plot_surface(xx, yy, obs_cost, cmap=plt.get_cmap('coolwarm'))
    ax3d.scatter(path0[:, 0], path0[:, 1], path0_values, s=20, c='r')
    ax3d.scatter(path1[:, 0], path1[:, 1], path1_values, s=20, c='g')
    plt.savefig(f"../out/q6/plot3d_{name}.png")
    plt.show()
    plt.close()


#
# Main program -----------------------------------------------------------------
if __name__ == "__main__":

    OBST = np.array([[20, 30], [60, 40], [70, 85]])
    epsilon = np.array([[25], [20], [30]])
    obs_cost, gx, gy = GenerateObstacleCostF(OBST, epsilon)

    path0, path0_values = GenerateInitialPath(obs_cost)
    # Plot(obs_cost, path0, path0_values, title='Initial Path')

    print(f"\n----------------------------------------------------------------")
    print("Question 6 (a) Optimize path")
    path1, path1_values = Optimize(path0, obs_cost, Cost.A, 1, t_step=0.8)
    Plot(obs_cost, path0, path0_values, path1, path1_values, name='path-a')
    
    path1, path1_values = Optimize6A(path0, obs_cost)
    # path1, path1_values = Optimize(path0, obs_cost, Cost.A, 5000, t_step=0.1)
    Plot(obs_cost, path0, path0_values, path1, path1_values, name='path-a_opt')

    # print(f"\n----------------------------------------------------------------")
    # print("Question 6 (b) Optimize path with additional cost iters = 100")
    # path1, path1_values = Optimize(path0, obs_cost, Cost.B, 100)
    # Plot(obs_cost, path0, path0_values, path1, path1_values, name='path-b_100')
    
    # print("Question 6 (b) Optimize path with additional cost iters = 100")
    # path1, path1_values = Optimize(path0, obs_cost, Cost.B, 200)
    # Plot(obs_cost, path0, path0_values, path1, path1_values, name='path-b_200')
    
    # print("Question 6 (b) Optimize path with additional cost iters = 500")
    # path1, path1_values = Optimize(path0, obs_cost, Cost.B, 500)
    # Plot(obs_cost, path0, path0_values, path1, path1_values, name='path-b_500')

    # print(f"\n----------------------------------------------------------------")
    # print("Question 6 (c) Optimize path with additional cost iters = 100")
    # path1, path1_values = Optimize(path0, obs_cost, Cost.C, 100)
    # Plot(obs_cost, path0, path0_values, path1, path1_values, name='path-c_100')
    
    # print("Question 6 (c) Optimize path with additional cost iters = 5000")
    # path1, path1_values = Optimize(path0, obs_cost, Cost.C, 5000)
    # Plot(obs_cost, path0, path0_values, path1, path1_values, name='path-c_5000')
