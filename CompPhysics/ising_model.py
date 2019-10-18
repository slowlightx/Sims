#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @TIME       : 2019/1/5  13:11
# @AUTHOR     : Yi Xu
# @FILE       : .py


import numpy as np
import matplotlib.pyplot as plt
import copy


def metropolis_hastings(S, T, num_of_iter):
    kb = 1 # boltzmann constant

    for times in range(num_of_iter):
        i = np.random.randint(0, S.shape[0])
        j = np.random.randint(0, S.shape[0])

        energy_before = get_local_energy(i, j, S)
        energy_after = get_local_energy(i, j, S)
        if energy_before < energy_after:
            prob = np.exp(-(energy_after-energy_before)/(kb*T))
            if np.random.random() < prob:
                S[i][j] = -S[i][j]
        else:
            S[i][j] = -S[i][j]
    return S


def get_local_energy(i, j, S):
    width = S.shape[0]
    height = S.shape[1]
    bottom = max(0, i-1)
    top = min(height-1, i+1)
    left = max(0, j-1)
    right = min(width-1, j+1)
    environment = [[bottom, j], [top, j], [i, left], [i, right]]
    energy = 0
    for t in range(4):
        energy += - S[i][j] * S[environment[t][0]][environment[t][1]]
    return energy


def get_total_energy(S):
    energy = 0
    for i in range(len(S)):
        for j in range(len(S[0])):
            energy += get_local_energy(i, j, S)
    average_energy = energy/(len(S[0])*len(S))
    return average_energy/2


def plot_mag_moment(S, i=0):
    # X, Y = np.meshgrid(np.arange(0, S.shape[0]),np.arange(0,S.shape[1]))

    fig = plt.figure()
    im = plt.imshow(S)
    fig.colorbar(im, shrink=0.5, aspect=10)
    plt.show()
    # save_path = "C:/Users/xy/PycharmProjects/computational_physics/venv/figure_heisenberg/"+str(i)+".png"
    # plt.savefig(save_path, format='png')
    # plt.close(fig)


def get_mag_moementum(S):
    mag_momentum = 0
    for i in range(len(S)):
        for j in range(len(S[0])):
            mag_momentum += S[i][j]
    return mag_momentum/(len(S)*len(S[0]))


def ising_model(size_of_sample, temperature):
    S = 2 * np.random.randint(2, size=(size_of_sample, size_of_sample)) - 1
    initial_energy = get_total_energy(S)
    print('系统的初始平均能量:', initial_energy)
    newS = np.array(copy.deepcopy(S))
    for nseeps in range(100):
        newS = metropolis_hastings(newS, temperature, 100 * size_of_sample ** 2)
    plot_mag_moment(newS, nseeps)
    ave_mag_momentum = get_mag_moementum(newS)/(size_of_sample**2)
    ave_energy = get_total_energy(newS)
    # plot_mag_moment(newS)
    print('系统的平均能量:', ave_energy)
    # reshaped = np.reshape(newS, (1, size_of_sample ** 2))
    return newS, ave_mag_momentum


res, ave_mag_m = ising_model(40, 0.0005)
print(ave_mag_m)