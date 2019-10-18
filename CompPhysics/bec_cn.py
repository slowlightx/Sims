#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @TIME       : 2019/1/5  21:53
# @AUTHOR     : Yi Xu
# @FILE       : .py

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def u0(x):
    return np.exp(-x**2/4)


def potential(u, x, N):
    # constant, the product of N and g, where g = 8*pi*(a/lambda) (before normalization, 4*pi*hbar^2*a/m)
    Ng = N * 8 * np.pi * 0.001
    # potential of the external field
    #V = x**2/4
    V = 0
    # effective potential of the BEC
    Veff = 1/1j*(V + Ng * np.abs(u)**2)*u
    return Veff


def create_A_matrix(J, gamma):
    """Create a tridiagonal matrix in the format used in the CN algorithm for the left hand side"""
    A = np.zeros((3, J), dtype=complex)
    A[0, 1] = 0
    A[0, 2:] = -gamma
    A[1, 0] = 1
    A[1, :] = 1 + 2 * gamma
    A[2, :-2] = -gamma
    A[2, -2] = 0
    return A


def create_B_matrix(J, gamma):
    """create a tridiagonal matrix in the format used in the CN algorithm for the right hand side"""
    dia_u = gamma*np.ones(J-1, dtype=complex)
    dia_u[0] = 0
    dia_l = gamma*np.ones(J-1, dtype=complex)
    dia_l[J-2] = 0
    dia = (1-2*gamma)*np.ones(J, dtype=complex)
    dia[0] = 1
    dia[-1] = 1
    return diags([dia_u, dia, dia_l], [1, 0, -1], (J, J), format='csr')


def crank_nicolson(xmin, xmax, J, dtau, T, D, N):
    # Initialize
    # grid
    x = np.linspace(xmin, xmax, J)
    dx = (xmax-xmin)/J
    dt = 1*dtau
    # matrix
    gamma = D*dt/dx**2
    A = create_A_matrix(J, gamma)
    B = create_B_matrix(J, gamma)
    # initial condition
    u1 = u0(x-10)
    u2 = u0(x+10)
    Norm1 = (np.sum(np.abs(u1) ** 2) * dx) ** 0.5
    u1 /= Norm1
    Norm2 = (np.sum(np.abs(u2) ** 2) * dx) ** 0.5
    u2 /= Norm2
    old_u1 = u1
    old_u2 = u2
    # Iteration
    t = np.arange(0, T, dtau)
    U = np.zeros((len(t), J), dtype=complex)
    for i in range(len(t)):
        U[i, :] = u1 + u2
        # for the first step, we calculate only with the previous state
        if i == 0:
            F1 = potential(u1, x, N)
            F2 = potential(u2, x, N)
        # else, we calculate with the previous step and the one before to keep the second order accuracy
        else:
            F1 = 3 / 2 * potential(u1, x, N) - 1 / 2 * potential(old_u1, x, N)
            F2 = 3 / 2 * potential(u2, x, N) - 1 / 2 * potential(old_u2, x, N)
        old_u1 = np.copy(u1)
        old_u2 = np.copy(u2)
        C1 = B.dot(u1) + dt * F1
        C2 = B.dot(u2) + dt * F2
        C1[0] = 0
        C1[-1] = 0
        C2[0] = 0
        C2[-1] = 0
        u1 = solve_banded((1, 1), A, C1)
        u2 = solve_banded((1, 1), A, C2)
        # normalize
        u1 /= Norm1
        u2 /= Norm2
    return U


xmin = -50
xmax = 50
J = 100001
dtau = 0.01
T = 5.01
D = 1j
N = 1000

u_1 = crank_nicolson(xmin, xmax, J, dtau, T, D, N)
rou = np.abs(u_1)**2


'''plot'''
fig1 = plt.figure()
ax = Axes3D(fig1)

T_Set = np.arange(0, T, dtau)
X_Set = np.linspace(xmin, xmax, J)

X_Grid, T_Grid = np.meshgrid(X_Set, T_Set)
surf = ax.plot_surface(X_Grid, T_Grid, rou, label=r"1", alpha=0.8)
# fig1.colorbar(surf, shrink=0.5, aspect=10)

ax.view_init(elev=25, azim=140) # 改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
# ax.plot(X_Set, np.zeros(len(T_Set)), np.abs(u0(X_Set)/(np.sum(np.abs(u0(X_Set)) ** 2) * 0.04) ** 0.5)**2)
plt.title(r"GP Equation Simulation(CN Method)")
ax.set_xlabel(r"x")
ax.set_ylabel(r"t")
ax.set_zlabel(r"$|\Psi|^2$")
plt.savefig('0_1.png')
#plt.show()


for i in range(6):
    plt.figure()
    labelname = 't='+str(i)
    plt.plot(np.linspace(xmin, xmax, J), rou[i*100, :], label=labelname)
    plt.xlabel(r'x')
    plt.ylabel(r'$|\Psi|^2$')
    plt.legend()
    filename = '0_t='+str(i)+'.png'
    plt.savefig(filename)
    #plt.show()
