"""
Created on 2020/2/26 16:01
Author: Xinyu Guo
Email: xyguo@bu.edu
IDE: PyCharm
"""
from Option_Pricing import *
import numpy as np
from scipy.stats import norm
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def calK_byDelta(vol, T, delta, type):
    N_inv = norm.ppf(delta)
    if type == 'c':
        K = 100 * np.exp(0.5 * vol ** 2 * T - vol * np.sqrt(T) * N_inv)
    else:
        K = 100 * np.exp(0.5 * vol ** 2 * T + vol * np.sqrt(T) * N_inv)
    return K

def cal_2ndDeriv(K, S0, r, sigma, T, h):
    C = Euro_option(K, S0, r, sigma, T).BSM()['Call']
    C_down = Euro_option(K-h, S0, r, sigma, T).BSM()['Call']
    C_up = Euro_option(K+h, S0, r, sigma, T).BSM()['Call']
    deriv = (C_down - 2*C + C_up)/h**2
    return deriv

def RN_Dnsty(vols, Ks, S0, r, T, h):
    pdf = []
    for i, sigma in enumerate(vols):
        prob = np.exp(r*T)*cal_2ndDeriv(Ks[i], S0, r, sigma, T, h)
        pdf.append(prob)
    plt.plot(Ks, pdf)
    plt.xlabel('Strike Price', size=15)
    plt.ylabel('Density', size=15)
    plt.title('Risk Neutral Density', size=20)
    return pdf

def RN_Dnsty_ConstVol(S0, r, T, h, vol):
    Ks = np.linspace(70, 130, 150)
    pdf = []
    for i, K in enumerate(Ks):
        prob = np.exp(r*T)*cal_2ndDeriv(K, S0, r, vol, T, h)
        pdf.append(prob)
    plt.plot(Ks, pdf)
    plt.xlabel('Strike Price', size=15)
    plt.ylabel('Density', size=15)
    plt.title('Risk Neutral Density with constant vol', size=20)
    return pdf

def dig_pay(S, K, opt_type):
    if opt_type == 'c':
        v = 1 if S>=K else 0
    else:
        v = 1 if S<=K else 0
    return v

def dig_price(density, S, K, opt_type):
    price = 0
    for i in range(0, len(S) - 2):
        price += density[i] * dig_pay(S[i], K, opt_type) * 0.1
    return price

def euro_price(density, S, K):
    price = 0
    for i in range(0, len(S) - 2):
        price += density[i] * max(0, S[i] - K) * 0.1
    return price


if __name__ == '__main__':
    S0 = 100
    r = 0
    ### problem a ###
    opt_name = ['10P','25P','40P','50P','40C','25C','10C']
    vol_data = [[32.25, 28.36],[24.73,21.78],[20.21,18.18],[18.24,16.45],[15.74,14.62],[13.7,12.56],[11.48,10.94]]
    Vol_dict = dict(zip(opt_name, vol_data))
    Vol_df = pd.DataFrame.from_dict(Vol_dict, orient='index', columns = ['1M','3M'])
    K_dict = {}

    for key in Vol_dict:
        delta = int(key[:2])/100
        opt_type = key[-1].lower()
        K_1m = calK_byDelta(Vol_dict[key][0]/100, 1/12, delta, opt_type)
        K_3m = calK_byDelta(Vol_dict[key][1]/100, 1/4, delta, opt_type)
        K_dict[key] = [K_1m, K_3m]

    K_df = pd.DataFrame.from_dict(K_dict, orient='index', columns = ['1M','3M'])
    print(K_df)

    ### problem b ###
    K_list = np.linspace(75, 110, 100)

    fit_1m = np.polyfit(K_df['1M'], Vol_df['1M']/100, 2)
    fit_3m = np.polyfit(K_df['3M'], Vol_df['3M']/100, 2)
    vol_func_1m = np.poly1d(fit_1m)(K_list)
    vol_func_3m = np.poly1d(fit_3m)(K_list)

    plt.plot(K_list, vol_func_1m, K_list, vol_func_3m)
    plt.legend(['1M','3M'])
    plt.xlabel('Strike Price', size=15)
    plt.ylabel('Vol', size=15)
    plt.title('Volatility v.s. Strike Price', size=20)

    ### problem c ###
    plt.figure()
    pdf_1 = RN_Dnsty(vol_func_1m, K_list, S0, r, T=1/12, h=0.1)
    pdf_3 = RN_Dnsty(vol_func_3m, K_list, S0, r, T=1/4, h=0.1)
    plt.legend(['1M','3M'])

    ### problem d ###
    plt.figure()
    RN_Dnsty_ConstVol(S0, r, T=1/12, h=0.1, vol=0.1824)
    RN_Dnsty_ConstVol(S0, r, T=1/4, h=0.1, vol=0.1645)
    plt.legend(['1M','3M'])

    ### problem e ###
    S = np.linspace(75,112.5,len(pdf_1))
    price1 = dig_price(pdf_1, S, K=110, opt_type = 'p')
    price2 = dig_price(pdf_3, S, K=105, opt_type = 'c')

    vol_2 = (vol_func_1m + vol_func_3m)/2
    pdf_2 = RN_Dnsty(vol_2, K_list, S0, r, T=1/6, h=0.1)
    price3 = euro_price(pdf_2, S, K=100)

    print('Price of 1M European Digital Put Option with Strike 110:', price1)
    print('Price of 3M European Digital Call Option with Strike 105:', price2)
    print('Price of 2M European Call Option with Strike 100:', price3)

    plt.show()



