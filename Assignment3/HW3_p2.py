# coding=utf-8

"""
Created on 2020/2/8 14:57
Author: Xinyu Guo
Email: xyguo@bu.edu
IDE: PyCharm
"""

from Option_Pricing import *
from FFT import *
from scipy.optimize import root
import matplotlib.pyplot as plt
import numpy as np


def K_vol_analysis(b, alpha, n, B, K, plot=True, ax=None):

    *_, [Ks, Calls] = b.Heston_fft(alpha, n, B, K)
    K_C_dict = {Ks[i]: Calls[i]
                for i in range(len(Ks)) if K - 70 < Ks[i] < K + 100}
    K_vol_dict = {}

    for KK in K_C_dict:
        result = root(
            lambda x: Euro_option(
                KK,
                S0,
                r,
                x,
                T).BSM()['Call'] -
            K_C_dict[KK],
            0.3)
        K_vol_dict[KK] = result.x

    if plot:
        ax.plot(list(K_vol_dict.keys()), list(K_vol_dict.values()))
        ax.set_title('Implied Volatility v.s. Strike Price', size=15)
        ax.set_xlabel('Strike Price K', size=12)
        ax.set_ylabel('Implied Volatility', size=12)


def T_vol_analysis(b, alpha, n, B, K, plot=True, ax=None):
    T_list = list(np.linspace(0.05, 1.8, 100))
    vol_list = []
    for TT in T_list:
        b.T = TT
        price = b.Heston_fft(alpha, n, B, K)[0]
        result = root(
            lambda x: Euro_option(
                K,
                S0,
                r,
                x,
                TT).BSM()['Call'] -
            price,
            0.3)
        vol_list.append(result.x)

    if plot:
        ax.plot(T_list, vol_list)
        ax.set_title('Implied Volatility v.s. Expiry', size=15)
        ax.set_xlabel('Expiry T', size=12)
        ax.set_ylabel('Implied Volatility', size=12)


def K_T_vol_analysis(b, alpha, ax, variable):
    K_vol_analysis(b, alpha, n=10, B=405, K=150, ax=ax[0])
    T_vol_analysis(b, alpha, n=10, B=405, K=150, ax=ax[1])
    ax[0].legend(variable)
    ax[1].legend(variable)


def Skew_Term_plot(sigma, v0, kappa, rho, theta):
    parameters = [sigma, v0, kappa, rho, theta]
    param_name = ['sigma', 'v0', 'kappa', 'rho', 'theta']
    chg_var = [parameters.index(v)
               for v in parameters if isinstance(v, list)][0]

    title = 'Impact of ' + param_name[chg_var]
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.suptitle(title, size=20)

    if chg_var == 0:
        variable = sigma
        for var in variable:
            b = FFT(var, v0, kappa, rho, theta, S0=150, r=0.025, T=0.25)
            K_T_vol_analysis(b, alpha, ax, variable)
    elif chg_var == 1:
        variable = v0
        for var in variable:
            b = FFT(sigma, var, kappa, rho, theta, S0=150, r=0.025, T=0.25)
            K_T_vol_analysis(b, alpha, ax, variable)
    elif chg_var == 2:
        variable = kappa
        for var in variable:
            b = FFT(sigma, v0, var, rho, theta, S0=150, r=0.025, T=0.25)
            K_T_vol_analysis(b, alpha, ax, variable)
    elif chg_var == 3:
        variable = rho
        for var in variable:
            b = FFT(sigma, v0, kappa, var, theta, S0=150, r=0.025, T=0.25)
            K_T_vol_analysis(b, alpha, ax, variable)
    elif chg_var == 4:
        variable = theta
        for var in variable:
            b = FFT(sigma, v0, kappa, rho, var, S0=150, r=0.025, T=0.25)
            K_T_vol_analysis(b, alpha, ax, variable)


if __name__ == '__main__':
    ######## problem b #########
    sigma = 0.4
    v0 = 0.09
    kappa = 0.5
    rho = 0.25
    theta = 0.12

    S0 = 150
    K = 150
    r = 0.025
    T = 0.25

    n = 10
    B = K * 2.7
    alpha = 1.5

    # (b) i
    b = FFT(sigma, v0, kappa, rho, theta, S0, r, T)
    # K_vol_analysis(b, alpha, n, B, K)

    # (b) ii
    T_vol_analysis(b, alpha, n, B, K)

    # # (b) iii
    # Impact of Sigma
    Skew_Term_plot([0.5, 0.6, 0.7], 0.09, 0.5, 0.25, 0.12)
    # Impact of V0
    Skew_Term_plot(0.4, [0.1, 0.12, 0.14], 0.5, 0.25, 0.12)
    # Impact of k
    Skew_Term_plot(0.4, 0.09, [0.6, 0.8, 1.0], 0.25, 0.12)
    # Impact of rho
    Skew_Term_plot(0.4, 0.09, 0.5, [0.35, 0.45, 0.55], 0.12)
    # # Impact of theta
    Skew_Term_plot(0.4, 0.09, 0.5, 0.25, [0.14, 0.16, 0.18])

    plt.show()
