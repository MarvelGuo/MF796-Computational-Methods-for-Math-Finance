"""
Created on 2020/4/2 13:31
Author: Xinyu Guo
Email: xyguo@bu.edu
IDE: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
from FFT import *
plt.style.use('seaborn')

def Simu_Heston_Path(steps, N, Option_params, Heston_params, seed=None):
    '''
    steps: total steps for one path
    N: number of simulated paths
    '''
    assert (len(Option_params) == 4) & (len(Heston_params) == 5), 'Wrong Input Parameters length'

    Kappa, theta, sigma, rho, v0 = Heston_params
    S0, T, r, q = Option_params
    dt = T/steps

    if seed is not None:
        np.random.seed(seed)

    mean = np.array([0, 0])
    cov = np.array([[dt, dt * rho], [dt * rho, dt]])
    dW = np.random.multivariate_normal(mean, cov, steps*N)
    dW1 = dW[:,0].reshape(N, steps)
    dW2 = dW[:,1].reshape(N, steps)

    S = np.zeros([N, steps])
    v = np.zeros([N, steps])
    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(1, steps):
        dvt = Kappa*(theta-np.maximum(v[:,t-1],0))*dt + sigma*np.maximum(v[:,t-1], 0)**0.5*dW2[:,t-1]
        v[:,t] = v[:, t - 1] + dvt

        dSt = (r-q)*S[:,t-1]*dt+np.maximum(v[:,t-1],0)**0.5*S[:,t-1]*dW1[:,t-1]
        S[:, t] = S[:, t - 1] + dSt

    return S

def cal_Euro(S, r, T, K, opt_type='c'):
    if opt_type == 'c':
        payoff = np.maximum(S[:,-1]-K, 0)
    else:
        payoff = np.maximum(K-S[:,-1], 0)
    opt_price = np.mean(payoff)*np.exp(-r*T)
    return opt_price

def cal_UpAndOut(S, r, T, K1, K2, opt_type='c'):
    if opt_type == 'c':
        max_S = np.max(S,axis=1)
        indicator = np.where(max_S<K2, 1, 0)
        payoff = np.maximum(S[:,-1]-K1, 0)*indicator
    else:
        max_S = np.max(S, axis=1)
        indicator = np.where(max_S < K2, 1, 0)
        payoff = np.maximum(K1 - S[:, -1], 0) * indicator
    opt_price = np.mean(payoff)*np.exp(-r*T)
    return opt_price

def cal_UpAndOut_vr(S, r, T, K1, K2, euro):
    pay_euro = np.maximum(S[:, -1] - K1, 0)

    max_S = np.max(S, axis=1)
    indicator = np.where(max_S < K2, 1, 0)
    pay_UAO = np.maximum(S[:, -1] - K1, 0) * indicator

    cov_mat = np.cov(pay_euro, pay_UAO)
    c = - cov_mat[0][1]/cov_mat[0][0]

    payoff = np.mean(pay_UAO) + c * (pay_euro - euro)

    opt_price = np.mean(payoff)*np.exp(-r*T)
    return opt_price


if __name__ == '__main__':
    Kappa = 3.52
    theta = 0.052
    sigma = 1.18
    rho = -0.77
    v0 = 0.034
    Heston_params = [Kappa, theta, sigma, rho, v0]

    S0 = 282
    r = 0.015
    q = 0.0177
    T = 1
    Option_params = [S0, T, r, q]

    ### problem 2 ###
    steps = 250
    N = 100000

    ### problem 3 ###
    K = 285
    simu_S = Simu_Heston_Path(steps, N, Option_params, Heston_params)
    euro_simu = cal_Euro(simu_S, r, T, K, opt_type='c')
    euro_FFT = FFT(Heston_params, T, S0, r, q).Heston_fft(alpha=1.5, K=K, B=K * 2.5)
    print('European Call price through Simulation is:', euro_simu)
    print('European Call price through FFT is:', euro_FFT)

    ### problem 4 ###
    K1 = 285
    K2 = 315
    simu_S = Simu_Heston_Path(steps, N, Option_params, Heston_params)
    UAO_price = cal_UpAndOut(simu_S, r, T, K1, K2)
    print('Up-And-Out Call price through simulation is:', UAO_price)


    Ns = np.logspace(1,4.9,400)
    UpAndOut = []
    for sample_N in Ns:
        sample_index = np.random.choice(N, int(sample_N), replace=False)
        sample_path = simu_S[sample_index, :]
        price = cal_UpAndOut(sample_path, r, T, K1, K2)
        UpAndOut.append(price)

    Error = [abs(p-UAO_price) for p in UpAndOut]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_title('Option Price vs Number of simulated paths')
    ax.set_ylabel('Option Price')
    ax.set_xlabel('Number of simulated paths (N)')
    ax.plot(Ns, UpAndOut, "-")


    ### problem 5 ###
    simu_S = Simu_Heston_Path(steps, N, Option_params, Heston_params)
    true_euro = cal_Euro(simu_S, r, T, K, opt_type='c')
    Ns = np.logspace(1,4.9,50)
    UpAndOut=[]
    UpAndOut_cv = []
    for sample_N in Ns:
        sample_index = np.random.choice(N, int(sample_N), replace=False)
        sample_path = simu_S[sample_index, :]
        price = cal_UpAndOut(sample_path, r, T, K1, K2, opt_type='c')
        price_cv = cal_UpAndOut_vr(sample_path, r, T, K1, K2, true_euro)
        UpAndOut.append(price)
        UpAndOut_cv.append(price_cv)

    R_Error = [abs(p-UAO_price) for p in UpAndOut]
    R_Error_vc = [abs(p-UAO_price) for p in UpAndOut_cv]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_title('Error vs Number of simulated paths')
    ax.set_ylabel('Error')
    ax.set_xlabel('Number of simulated paths (N)')
    ax.plot(Ns, R_Error, Ns, R_Error_vc)
    ax.legend(['Simulation Error without vc','Simulation Error with vc'])

    plt.show()


