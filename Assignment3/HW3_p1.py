# coding=utf-8

"""
Created on 2020/2/5 13:34
Author: Xinyu Guo
Email: xyguo@bu.edu
IDE: PyCharm
"""

from FFT import *
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


if __name__ == '__main__':
    ######## problem a #########
    sigma = 0.2
    v0 = 0.08
    kappa = 0.7
    rho = -0.4
    theta = 0.1

    S0 = 250
    K = 250
    r = 0.02
    T = 0.5

    n = 11
    B = 250 * 2.7

    a = FFT(sigma, v0, kappa, rho, theta, S0, r, T)
    # print(a.Heston_fft(1.5, n, B, K)[0])

    # (a) i
    alpha_list = np.linspace(0,30,100)
    price_alpha = [a.Heston_fft(alpha, n, B, K)[0] for alpha in alpha_list]
    plt.plot(alpha_list, price_alpha)
    plt.title(r'Option price v.s alpha')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Option Price')

    # (a) ii
    Bs = np.linspace(K*2.5,K*2.7,100)
    ns = np.array([7,8,9,10,11,12,13,14])
    alpha = 1.5
    print('When K is 250:')
    a.NB_plot(ns, Bs, alpha, K, 21.27)


    # (a) iii
    K2 = 260
    alpha_list = [0.01, 0.02, 0.25, 0.5, 0.8, 1 ,1.05, 1.5, 1.75,10,30,40]
    b = {alpha:a.Heston_fft(alpha,n,B,K2)[0] for alpha in alpha_list}
    print('When K2 is 260, option price of different alpha:\n',b)
    # price is 16.73

    alpha = 1.5
    Bs = np.linspace(K2*2.5,K2*2.7,100)
    ns = np.array([7,8,9,10,11,12,13,14])
    a.NB_plot(ns, Bs, alpha, K2, 16.73)

    plt.show()