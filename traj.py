#! /usr/bin/env python3

'''
Sulla base di questo risultato non credo ci siano problemi sulla traiettoria.
Piuttosto bisogna capire quale può essere il problema che causa scatti
probabilmente rigurda la tf letta quindi bisogna usare la forward kinematics con i joint
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.special import comb
from sklearn.metrics import r2_score

def get_bezier_parameters(X, Y, degree=3):
    """ Least square qbezier fit using penrose pseudoinverse.

    Parameters:

    X: array of x data.
    Y: array of y data. Y[0] is the y point for X[0].
    degree: degree of the Bézier curve. 2 for quadratic, 3 for cubic.

    Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
    and probably on the 1998 thesis by Tim Andrew Pastva, "Bézier Curve Fitting".
    """
    if degree < 1:
        raise ValueError('degree must be 1 or greater.')

    if len(X) != len(Y):
        raise ValueError('X and Y must be of the same length.')

    if len(X) < degree + 1:
        raise ValueError(f'There must be at least {degree + 1} points to '
                         f'determine the parameters of a degree {degree} curve. '
                         f'Got only {len(X)} points.')

    def bpoly(n, t, k):
        """ Bernstein polynomial when a = 0 and b = 1. """
        return t ** k * (1 - t) ** (n - k) * comb(n, k)
        #return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    def bmatrix(T):
        """ Bernstein matrix for Bézier curves. """
        return np.matrix([[bpoly(degree, t, k) for k in range(degree + 1)] for t in T])

    def least_square_fit(points, M):
        M_ = np.linalg.pinv(M)
        return M_ * points

    T = np.linspace(0, 1, len(X))
    M = bmatrix(T)
    points = np.array(list(zip(X, Y)))
    
    final = least_square_fit(points, M).tolist()
    final[0] = [X[0], Y[0]]
    final[len(final)-1] = [X[len(X)-1], Y[len(Y)-1]]
    return final

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=50):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

# read a csv file
# df = pd.read_csv('/home/giorgio/ros/spline/trajectory.csv')
df = pd.read_csv('/home/lar-tesi01/ros/spline/trajectory.csv')


j1 = df['J1'].values
j2 = df['J2'].values
j3 = df['J3'].values
j4 = df['J4'].values
j5 = df['J5'].values
j6 = df['J6'].values
index = df.index.values


# from sklearn.metrics import mean_squared_error

# def find_best_degree(points, bounds=(1, 10)):
#     # separa i punti in coordinate x e y
#     x, y = points.T
#     best_r2 = 100
#     best_rmse = 100
#     best_mse = 100
#     best_degree = 10

#     # definisci la funzione da minimizzare
#     for degree in range(*bounds, 1):
        
#         # calcola i parametri di Bezier per il grado corrente
#         data = get_bezier_parameters(x, y, degree=degree)
        
#         # calcola la curva di Bezier per il grado corrente
#         xvals, yvals = bezier_curve(data, nTimes=len(x))
        
#         # calcola il valore di R2 per il grado corrente
#         r_squared = r2_score(y, yvals)
#         mse = mean_squared_error(y, yvals)
#         rmse = np.sqrt(mse)
        
#         # salva il valore di R2 più vicino a 1 e il grado corrispondente
#         if np.abs(r_squared -1) < np.abs(best_r2 - 1):
#             best_r2 = r_squared
        
#         if rmse < best_rmse:
#             best_rmse = rmse
#             best_degree = degree

#         if mse < best_mse:
#             best_mse = mse
            
#         print("Degree:", degree, "R2:", r_squared, "RMSE:", rmse, "MSE:", mse)
    
#     print("Best degree:", best_degree, "Best R2:", best_r2, "Best RMSE:", best_rmse, "Best MSE:", best_mse)

#     return best_degree, best_r2


# # esempio di utilizzo della funzione
# points = np.array(list(zip(index, j1)))
# # best_degree, best_r2 = find_best_degree(points)


# # calcola i parametri di Bezier per il grado corrente
# data = get_bezier_parameters(index, j1, degree=8)
# data = np.array(data)
# x_val, y_val = data.T

# # calcola la curva di Bezier per il grado corrente
# xvals, yvals = bezier_curve(data, nTimes=len(index))


# plt.plot(index, j1, "ro",label='Original Points')
# plt.plot(x_val,y_val,'k--o', label='Control Points')
# plt.plot(xvals, yvals, 'b-', label='B Curve')
# plt.xlim(-1, 16000)
# plt.ylim(-1.8, 0.8)
# plt.legend()
# plt.show()

# quit()


# compute velocity and acceleration
v1 = np.diff(j1)
v2 = np.diff(j2)
v3 = np.diff(j3)
v4 = np.diff(j4)
v5 = np.diff(j5)
v6 = np.diff(j6)

a1 = np.diff(v1)
a2 = np.diff(v2)
a3 = np.diff(v3)
a4 = np.diff(v4)
a5 = np.diff(v5)
a6 = np.diff(v6)

index_a1 = []

for i in range(0, len(a1)):
    # salvo gli indici dei punti in cui la velocità è maggiore di 0.1
    if np.abs(a1[i]) > 0.0005:
        index_a1.append(i)


primo = np.arange(4100, 5000, 1, dtype=int)
secondo = np.arange(6800, 8000, 1, dtype=int)
index_a1 = np.concatenate((primo, secondo), dtype=int)

j1_new = np.delete(j1, index_a1)
index_new = np.delete(index, index_a1)


# interpolo i punti rimasti con una cubic spline
cs = CubicSpline(index_new, j1_new)

j1 = cs(index)
v1 = np.diff(j1)
a1 = np.diff(v1)

# plt.scatter(index, j1, s=1, label='Original Points')

# plot position and velocity and acceleration of each joint
plt.figure(1)
plt.subplot(311)
plt.plot(index, j1, 'r')
plt.ylabel('J1')
plt.subplot(312)
plt.plot(index[1:], v1, 'r')
plt.ylabel('V1')
plt.subplot(313)
plt.plot(index[2:], a1, 'r')
plt.ylabel('A1')
plt.xlabel('Time')

# plt.figure(2)
# plt.subplot(311)
# plt.plot(index, j2, 'g')
# plt.ylabel('J2')
# plt.subplot(312)
# plt.plot(index[1:], v2, 'g')
# plt.ylabel('V2')
# plt.subplot(313)
# plt.plot(index[2:], a2, 'g')
# plt.ylabel('A2')
# plt.xlabel('Time')

# plt.figure(3)
# plt.subplot(311)
# plt.plot(index, j3, 'b')
# plt.ylabel('J3')
# plt.subplot(312)
# plt.plot(index[1:], v3, 'b')
# plt.ylabel('V3')
# plt.subplot(313)
# plt.plot(index[2:], a3, 'b')
# plt.ylabel('A3')
# plt.xlabel('Time')

# plt.figure(4)
# plt.subplot(311)
# plt.plot(index, j4, 'y')
# plt.ylabel('J4')
# plt.subplot(312)
# plt.plot(index[1:], v4, 'y')
# plt.ylabel('V4')
# plt.subplot(313)
# plt.plot(index[2:], a4, 'y')
# plt.ylabel('A4')
# plt.xlabel('Time')

# plt.figure(5)
# plt.subplot(311)
# plt.plot(index, j5, 'c')
# plt.ylabel('J5')
# plt.subplot(312)
# plt.plot(index[1:], v5, 'c')
# plt.ylabel('V5')
# plt.subplot(313)
# plt.plot(index[2:], a5, 'c')
# plt.ylabel('A5')
# plt.xlabel('Time')

# plt.figure(6)
# plt.subplot(311)
# plt.plot(index, j6, 'm')
# plt.ylabel('J6')
# plt.subplot(312)
# plt.plot(index[1:], v6, 'm')
# plt.ylabel('V6')
# plt.subplot(313)
# plt.plot(index[2:], a6, 'm')
# plt.ylabel('A6')
# plt.xlabel('Time')

plt.show(block=False)

input("Press Enter to close all the figure...")
plt.close('all')