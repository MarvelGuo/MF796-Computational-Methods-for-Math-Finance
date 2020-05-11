"""
Created on 2020/2/14 10:46
Author: Xinyu Guo
Email: xyguo@bu.edu
IDE: PyCharm
"""
import yfinance as yf
from Get_Ticker import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.optimize import minimize
plt.style.use('seaborn')

def Get_stock_data(local=True):
    if local:
        price=pd.read_csv('Price_data.csv', index_col=0)
    else:
        save_sp500_tickers()
        with open('sp500tickers.pickle', 'rb') as f:
            data = pickle.load(f)
        ticker_list = ''.join(data[:123]).replace('\n', ' ')

        data = yf.download(ticker_list, start='2015-01-01', end='2020-01-01')
        price = data['Adj Close']
        price.to_csv('Price_data.csv')
    return price

def Eigen_Decomp(matrix, print_out=True):
    m = np.array(matrix)
    eigen_value = LA.eig(m)[0]
    eigen_vector = LA.eig(m)[1]
    N = len(eigen_value)

    pos = len(eigen_value[eigen_value > 0]) / N
    neg = len(eigen_value[eigen_value < 0]) / N

    if print_out:
        print('\n%d%% of eigenvalues are positive.' % (pos * 100))
        print('%d%% of eigenvalues are negative.' % (neg * 100))

    return eigen_value, eigen_vector

if __name__ == '__main__':

    ######### problem 1 ###########
    ## 1 ##
    price = Get_stock_data()
    price.dropna(axis=1, how='all', inplace=True)
    price.fillna(method='bfill',inplace=True)

    ## 2 ##
    r_df = np.log(price.pct_change().dropna() + 1)
    print(r_df.head())

    ## 3 ##
    cov_df = r_df.cov()
    eig_value, eigen_vector = Eigen_Decomp(cov_df)

    ## 4 ##
    total_var = np.sum(eig_value)
    portion = np.cumsum(eig_value) / total_var
    perc_50 = len(portion[portion < 0.5]) + 1
    perc_90 = len(portion[portion < 0.9]) + 1
    print('\n{} eigenvalues are requied to account for 50% of variance'.format(perc_50))
    print('{} eigenvalues are requied to account for 90% of variance'.format(perc_90))

    ## 5 ##
    W = eigen_vector[:, :perc_90]
    residual = r_df - np.dot(r_df.values.dot(W), W.T)

    fig, ax = plt.subplots()
    r_df.plot(ax=ax)
    ax.legend_.remove()
    ax.set_xlabel('Date', size=18)
    ax.set_ylabel('Return', size=18)
    ax.set_title('Original Return', size=22)

    fig, ax = plt.subplots()
    residual.plot(ax=ax)
    ax.legend_.remove()
    ax.set_xlabel('Date', size=18)
    ax.set_ylabel('Return', size=18)
    ax.set_title('Return Stream of Residual', size=22)

    ######### problem 2 ###########
    N = len(r_df.columns)
    G = np.zeros([2, N])
    G[0, :18] = 1
    G[1, :] = 1

    C = cov_df.values
    R = r_df.mean(axis=0).values
    a = 1
    c = np.array([1, 0.1])

    C_inv = LA.inv(C)
    lmbd = LA.inv(np.dot(G, C_inv.dot(G.T))).dot(G.dot(C_inv).dot(R) - 2 * a * c)
    w = 1 / 2 / a * C_inv.dot((R - G.T.dot(lmbd)))

    plt.figure()
    plt.plot(range(len(w)), w)
    plt.xlabel('Assets', size=18)
    plt.ylabel('Weights', size=18)
    plt.title('Portfolio Allocation', size=22)

    ## Optimizer Solution ##
    # fun = lambda w: -(R.dot(w) - a * np.dot(w, C.dot(w)))
    # G = np.zeros(N)
    # G[:18] = 1
    # cons = ({'type': 'eq', 'fun': lambda w: G.dot(w) - 0.1},
    #         {'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    # w0 = np.random.random(N)
    # res = minimize(fun, w0, method='SLSQP', constraints=cons)
    #
    # print('Minimal Value：', res.fun)
    # print('Optimal solution：', res.x)
    # print('Success of iteration or not：', res.success)
    # print('Reason for the stop pf iteration：', res.message)
    #
    # w = res.x
    # plt.plot(range(len(w)), w)


    plt.show()



