#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @TIME       : 2019/1/5  10:44
# @AUTHOR     : Yi Xu
# @FILE       : .py


import numpy as np
import matplotlib.pyplot as plt
import copy

def metropolis_hastings(S, T, num_of_iter):
    max_flip_angle = np.pi/2
    kb = 1

    for times in range(num_of_iter):
        i = np.random.randint(0, S.shape[0])
        j = np.random.randint(0, S.shape[0])

        delta = (2*np.random.random()-1)*max_flip_angle
        energy_before = get_local_energy(i, j, S)
        energy_after = get_local_energy(i, j, S, angle=delta)
        if energy_before < energy_after:
            prob = np.exp(-(energy_after-energy_before)/(kb*T))
            if np.random.random() < prob:
                S[i][j] = S[i][j] + delta - np.floor((S[i][j] + delta)/(2*np.pi))
        else:
            S[i][j] = S[i][j] + delta - np.floor((S[i][j] + delta)/(2*np.pi))
    return S


def get_local_energy(i, j, S, angle=None):
    width = S.shape[0]
    height = S.shape[1]
    bottom = max(0, i-1)
    top = min(height-1, i+1)
    left = max(0, j-1)
    right = min(width-1, j+1)
    environment = [[bottom, j], [top, j], [i, left], [i, right]]
    energy = 0
    if angle == None:
        for t in range(4):
            energy += -np.cos(S[i][j] - S[environment[t][0]][environment[t][1]])
    else:
        for t in range(4):
            energy += -np.cos(S[i][j] + angle - S[environment[t][0]][environment[t][1]])
    return energy


def get_total_energy(S):
    energy = 0
    for i in range(len(S)):
        for j in range(len(S[0])):
            energy += get_local_energy(i, j, S)
    average_energy = energy/(len(S[0])*len(S))
    return average_energy/2


def plot_mag_moment(S, i=0):
    X, Y = np.meshgrid(np.arange(0, S.shape[0]), np.arange(0,S.shape[1]))

    U = np.cos(S)
    V = np.sin(S)

    fig = plt.figure()
    Q = plt.quiver(X, Y, U, V, units='inches')
    #im = plt.imshow(S)
    #fig.colorbar(im, shrink=1, aspect=15)
    #im.set_clim(vmin=0, vmax=2*np.pi)
    #plt.show()
    save_path = "C:/Users/xy/PycharmProjects/computational_physics/venv/figure_xy/"+str(i)+".png"
    plt.savefig(save_path, format='png')
    plt.close(fig)


def get_mag_moementum(S):
    mag_momentum = np.zeros([1, 2])
    for i in range(len(S)):
        for j in range(len(S[0])):
            mag_momentum[0, 0] += np.cos(S[i][j])
            mag_momentum[0, 1] += np.sin(S[i][j])
    return mag_momentum/(len(S)*len(S[0]))


def xy_model(size_of_sample, temperature):
    S = 2 * np.pi * np.random.random((size_of_sample, size_of_sample))
    initial_energy = get_total_energy(S)
    #print('系统的初始平均能量:', initial_energy)
    newS = np.array(copy.deepcopy(S))
    for nseeps in range(100):
        newS = metropolis_hastings(newS, temperature, size_of_sample ** 2)
    #plot_mag_moment(newS, nseeps)
    ave_mag_momentum = get_mag_moementum(newS)
    ave_energy = get_total_energy(newS)
    # plot_mag_moment(newS)
    #print('系统的平均能量:', ave_energy)
    # reshaped = np.reshape(newS, (1, size_of_sample ** 2))
    return newS, ave_mag_momentum


#res, ave_mag_m = xy_model(40, 5)
#print(ave_mag_m)


'''
接下来我们尝试来作出平均磁矩和温度的关系，借此来寻找临界温度.
由于单次误差会较大，于是对每个温度下的磁矩都作多次模拟并取平均.
'''
t_set = np.linspace(0.01, 5, 20)
amm = np.zeros(len(t_set))
for i in range(len(t_set)):
    for j in range(1):
        res1, res2 = xy_model(40, t_set[i])
        amm[i] += np.sqrt(res2[0, 0]**2+res2[0, 1]**2)
    amm[i] = amm[i]/1

plt.figure()
plt.plot(t_set, amm, label=r'mean magnetic momentum')
plt.xlabel(r'temperature')
plt.ylabel(r'mean magnetic momentum')
plt.title(r'phase transition of xy model')
#plt.show()
plt.savefig("C:/Users/xy/PycharmProjects/computational_physics/venv/figure_xy/phase_trans.png")