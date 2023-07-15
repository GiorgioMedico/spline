#! /usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.special import comb


# read a csv file
df = pd.read_csv('/home/giorgio/ros/spline/trajectory.csv')
# df = pd.read_csv('/home/lar-tesi01/ros/spline/trajectory.csv')


j1 = df['J1'].values
j2 = df['J2'].values
j3 = df['J3'].values
j4 = df['J4'].values
j5 = df['J5'].values
j6 = df['J6'].values
index = df.index.values

# se j5 = 0,pi,-pi or j3 = 0,pi,-pi, or j2=0andj3=0andj4=0
# the trajectory is not feasible print a message and exit
tol = 1e-1
if np.allclose(j5, 0,atol=tol) or np.allclose(j5, np.pi,atol=tol) or np.allclose(j5, -np.pi,atol=tol) or np.allclose(j3, 0,atol=tol) or np.allclose(j3, np.pi,atol=tol) or np.allclose(j3, -np.pi,atol=tol) or np.allclose(j2, 0,atol=tol) and np.allclose(j3, 0,atol=tol) and np.allclose(j4, 0,atol=tol):
    print('Trajectory not feasible')
    quit()

quit()

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

plt.figure(2)
plt.subplot(311)
plt.plot(index, j2, 'g')
plt.ylabel('J2')
plt.subplot(312)
plt.plot(index[1:], v2, 'g')
plt.ylabel('V2')
plt.subplot(313)
plt.plot(index[2:], a2, 'g')
plt.ylabel('A2')
plt.xlabel('Time')

plt.figure(3)
plt.subplot(311)
plt.plot(index, j3, 'b')
plt.ylabel('J3')
plt.subplot(312)
plt.plot(index[1:], v3, 'b')
plt.ylabel('V3')
plt.subplot(313)
plt.plot(index[2:], a3, 'b')
plt.ylabel('A3')
plt.xlabel('Time')

plt.figure(4)
plt.subplot(311)
plt.plot(index, j4, 'y')
plt.ylabel('J4')
plt.subplot(312)
plt.plot(index[1:], v4, 'y')
plt.ylabel('V4')
plt.subplot(313)
plt.plot(index[2:], a4, 'y')
plt.ylabel('A4')
plt.xlabel('Time')

plt.figure(5)
plt.subplot(311)
plt.plot(index, j5, 'c')
plt.ylabel('J5')
plt.subplot(312)
plt.plot(index[1:], v5, 'c')
plt.ylabel('V5')
plt.subplot(313)
plt.plot(index[2:], a5, 'c')
plt.ylabel('A5')
plt.xlabel('Time')

plt.figure(6)
plt.subplot(311)
plt.plot(index, j6, 'm')
plt.ylabel('J6')
plt.subplot(312)
plt.plot(index[1:], v6, 'm')
plt.ylabel('V6')
plt.subplot(313)
plt.plot(index[2:], a6, 'm')
plt.ylabel('A6')
plt.xlabel('Time')

plt.show(block=False)

input("Press Enter to close all the figure...")
plt.close('all')