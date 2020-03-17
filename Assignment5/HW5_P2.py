"""
Created on 2020/2/27 12:38
Author: Xinyu Guo
Email: xyguo@bu.edu
IDE: PyCharm
"""
import pandas as pd
import numpy as np
from FFT import *
from scipy.optimize import minimize

### problem a ###


def read_data():
    file = r'C:\Users\PC\Desktop\2020spring\MF796\HW\HW5\mf796-hw5-opt-data.xlsx'
    data = pd.read_excel(file)

    data['call_mid'] = (data.call_bid + data.call_ask) / 2
    data['put_mid'] = (data.put_bid + data.put_ask) / 2

    data_call = data[['expDays', 'expT', 'K',
                      'call_mid', 'call_ask', 'call_bid']]
    data_put = data[['expDays', 'expT', 'K', 'put_mid', 'put_ask', 'put_bid']]
    return data_call, data_put, data


def check_mono(df, opt_type):
    '''
    Check whether the option price is monotonically changing
    '''
    mid_col = df.columns[df.columns.str.contains(
        'mid')][0]  # find the column containing 'mid'
    if opt_type == 'c':
        return any(df[mid_col].pct_change().dropna() >= 0)
    else:
        return any(df[mid_col].pct_change().dropna() <= 0)


def check_delta(df, opt_type):
    '''
    Check the delta of option price to strike price
    '''
    mid_col = df.columns[df.columns.str.contains('mid')][0]
    df['delta'] = (df[mid_col] - df[mid_col].shift(1)) / (df.K - df.K.shift(1))
    if opt_type == 'c':
        return any(df.delta >= 0) or any(df.delta < -1)
    else:
        return any(df.delta > 1) or any(df.delta <= 0)


def check_convex(df):
    '''
    Check the convexity of option price
    '''
    mid_col = df.columns[df.columns.str.contains('mid')][0]
    df['convex'] = df[mid_col] - 2 * \
        df[mid_col].shift(1) + df[mid_col].shift(2)
    return any(df.convex < 0)


def ArbitrageCheck(df, opt_type):
    r1 = check_mono(df, opt_type)
    r2 = check_delta(df, opt_type)
    r3 = check_convex(df)
    return pd.Series([r1, r2, r3], index=['Monotonic', 'Delta', 'Convexity'])


### problem b ###
def cal_Heston_price(K, Ks, T, alpha, params):
    prices = FFT(params, T).Heston_fft(alpha, K, B=K * 2.5, Ks=Ks)
    return prices


def sqr_sum(data, alpha, params, weighted=False):
    opt = data.columns[3].split('_')[0]
    obj = 0
    if not weighted:
        for T in data.expT.unique():
            temp = data[data.expT == T]
            Ks = temp.K.values
            prices = cal_Heston_price(np.mean(Ks), Ks, T, alpha, params)
            obj += np.sum((prices - temp[opt + '_mid'].values)**2)
    else:
        for T in data.expT.unique():
            temp = data[data.expT == T]
            Ks = temp.K.values
            w = 1 / (temp[opt + '_ask'] - temp[opt + '_bid'])
            w = w.values
            prices = cal_Heston_price(np.mean(Ks), Ks, T, alpha, params)
            obj += w.dot((prices - temp[opt + '_mid'].values)**2)
    return obj


def obj_func(params, alpha, data_call, data_put, weighted=False):
    '''
    For call, alpha>0; For put, alpha<0
    '''
    call_sum = sqr_sum(data_call, alpha, params, weighted)
    put_sum = sqr_sum(data_put, -alpha, params, weighted)
    return call_sum + put_sum


def callbackF1(Xi):
    global times
    if times % 5 == 0:
        print('{}: {}'.format(times, obj_func(Xi, alpha, data_call, data_put)))
        print(Xi)
    times += 1

def callbackF2(Xi):
    global times
    print('{}: {}'.format(times, obj_func(Xi, alpha, data_call, data_put, True)))
    print(Xi)
    times += 1


if __name__ == '__main__':

    ### problem a ###
    data_call, data_put, data = read_data()
    Call_check = data_call.groupby('expDays').apply(
        ArbitrageCheck, opt_type='c')
    Put_check = data_put.groupby('expDays').apply(
        ArbitrageCheck, opt_type='p')
    print('Arbitrage check of Call:\n', Call_check)
    print('Arbitrage check of Put:\n', Put_check)

    ### problem b ###
    # parameters: kappa, theta, sigma, rho, v0

    start_params = [0, 0.2, 0.2, 0, 0.2]
    lower = [0.01, 0.01, 0.0, -1, 0.0]
    upper = [2.5, 1, 1, 0.5, 0.5]
    bounds = tuple(zip(lower, upper))
    alpha = 1.5

    times = 1
    args = (alpha, data_call, data_put)
    result = minimize(
        obj_func,
        np.array(start_params),
        args=args,
        method='SLSQP',
        bounds=bounds,
        callback=callbackF1)

    print(result.success)
    print(result.x)
    print(result.fun)

    ### problem d ###
    start_params = [4.14,0.06,1.67,-0.81,0.04]
    lower = [0.01,0.01,0,-1,0]
    upper = [2.5, 1, 1, 0.5, 0.5]

    bounds = tuple(zip(lower, upper))
    alpha = 1.5

    times = 1
    args = (alpha, data_call, data_put, True)
    result = minimize(
        obj_func,
        np.array(start_params),
        args=args,
        method='SLSQP',
        bounds=bounds,
        callback=callbackF2)

    print(result.success)
    print(result.x)
    print(result.fun)

