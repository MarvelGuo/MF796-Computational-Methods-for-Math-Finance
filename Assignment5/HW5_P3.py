"""
Created on 2020/2/28 23:52
Author: Xinyu Guo
Email: xyguo@bu.edu
IDE: PyCharm
"""

from FFT import *
from Option_Pricing import *
from scipy.optimize import root


def cal_deltas(h, params):

    K = 275; S0 =267.15; r=0.015; T = 1/4

    ## FFT delta ##
    C = FFT(params, T, S0=S0).Heston_fft(alpha=1.5, K=275, B=275*2.5, n=11)
    C_up = FFT(params, T, S0=S0+h).Heston_fft(alpha=1.5, K=275, B=275*2.5, n=11)
    C_down = FFT(params, T, S0=S0-h).Heston_fft(alpha=1.5, K=275, B=275*2.5, n=11)
    delta_FFT = (C_up-C_down)/2/h

    ## BSM delta ##
    iv = root(lambda x: Euro_option(K, S0, r, x, T).BSM()['Call'] - C, 0.2).x
    delta_BSM = Euro_option(K, S0, r, iv, T).delta()

    print('Delta calculated by FFT is:', delta_FFT)
    print('Delta calculated by BSM is:', delta_BSM)

def cal_vegas(h, params):

    K = 275; S0 =267.15; r=0.015; T = 1/4

    ## FFT delta ##
    C = FFT(params, T, S0=S0).Heston_fft(alpha=1.5, K=275, B=275*2.5, n=11)
    params_up = params + np.array([0, h, 0, 0, h])
    # params_up[1] += h
    # params_up[4] += h
    C_up = FFT(params_up, T, S0=S0).Heston_fft(alpha=1.5, K=275, B=275*2.5, n=11)
    params_down = params - np.array([0, h, 0, 0, h])
    # params_down[1] -= h
    # params_down[4] -= h
    C_down = FFT(params_down, T, S0=S0).Heston_fft(alpha=1.5, K=275, B=275*2.5, n=11)
    vega_FFT = (C_up-C_down)/2/h

    ## BSM delta ##
    iv = root(lambda x: Euro_option(K, S0, r, x, T).BSM()['Call'] - C, 0.2).x
    vega_BSM = Euro_option(K, S0, r, iv, T).vega()

    print('Vega calculated by FFT is:', vega_FFT)
    print('Vega calculated by BSM is:', vega_BSM)



if __name__ == '__main__':

    ### problem a ###
    # parameters: kappa, theta, sigma, rho, v0
    params = np.array([3.51,0.052,1.17,-0.77,0.034])

    cal_deltas(0.01, params)
    cal_vegas(0.05*0.034, params)

