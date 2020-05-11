"""
Created on 2020/3/9 17:50
Author: Xinyu Guo
Email: xyguo@bu.edu
IDE: PyCharm
"""
from Option_Pricing import *
from scipy.optimize import root
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
plt.style.use('ggplot')


class Explicit_Euler:
    def __init__(self, opt_params, smax, M, N):
        self.S0 = opt_params['S0']
        self.K1 = opt_params['K'][0]
        self.K2 = opt_params['K'][1]
        self.rf = opt_params['Rf']
        self.T = opt_params['T']
        self.sigma = opt_params['sigma']

        self.smax = smax
        self.M = M
        self.N = N
        self.hs = smax/M
        self.ht = self.T/N

        self.S = np.arange(self.hs, smax+self.hs, self.hs) # size: M
        self.A = self.Matrix_A()

    def stability_check(self):
        # check stability condition for ht
        # ht_check = (self.ht < (self.hs/self.sigma/self.smax)**2)

        # check the eigenvalues of matrix A
        abs_eig = np.array(sorted(abs(LA.eig(self.A)[0]), reverse=True))
        A_check = any(abs_eig > 1)
        plt.plot(range(len(abs_eig)), abs_eig)
        plt.plot(range(self.M-1), [1]*(self.M-1), color='black', linewidth=1.5, linestyle='--')
        plt.title('Eigen values of matrix A',size=20)

        if A_check:
            return False
        else:
            return True

    def pay_call_spread(self, ST):
        long = ST-K1
        long[long<=0] = 0
        short = ST-K2
        short[short<=0] = 0
        pay = long - short
        return pay

    def Matrix_A(self):
        hs = self.hs
        ht = self.ht
        S = self.S[:-1] # size: M-1

        self.a = 1 - self.sigma**2*S**2*ht/hs**2 - self.rf*ht
        self.l = self.sigma**2*S**2/2*ht/hs**2 - self.rf*S*ht/2/hs
        self.u = self.sigma**2*S**2/2*ht/hs**2 + self.rf*S*ht/2/hs

        A = np.diag(self.a)
        l = self.l[1:]
        u = self.u[:-1]
        for i in range(M-2):
            A[i][i+1] = u[i]
            A[i+1][i] = l[i]

        return A

    def Cal_Call_Price(self, earlyEx=False):

        if self.stability_check() == False:
            print('Not satisfying Stability Condition')
            return None

        C = np.zeros([self.M-1, self.N])
        CT = self.pay_call_spread(self.S[:-1])
        C[:,-1] = CT
        for j in range(self.N, 1, -1):
            tj = self.ht*j
            bj = self.u[-1]*(self.K2-self.K1)*np.exp(-self.rf*(self.T-tj))  # formula 10
            C[:,j-2] = self.A.dot(C[:,j-1])
            C[-1,j-2] = C[-1,j-2] + bj
            if earlyEx:
                C[:,j-2] = np.max([C[:,j-2], CT], axis=0)

        return np.interp(self.S0, self.S[:-1], C[:,0])

if __name__ == '__main__':

    rf = 0.72/100
    K1 = 315
    K2 = 320
    T = 7/12
    S0 = 312.86 # price on March 4th

    # mkt price of calls on March 6th
    C1 = 14.5
    C2 = 12.2

    ### 3 ###
    iv1 = root(lambda x: Euro_option(K1, S0, rf, x, T).BSM()['Call'] - C1, 0.2).x[0]
    iv2 = root(lambda x: Euro_option(K2, S0, rf, x, T).BSM()['Call'] - C2, 0.2).x[0]
    vol = (iv1+iv2)/2
    print('The Volatility is:', round(vol,2))

    ### 4 ###
    smax = 600
    M = 250
    N = 1000
    opt_params = dict(zip(['S0','K','Rf','T','sigma'],[S0, [K1, K2], rf, T, vol]))

    ### 6 ###
    SpreadPrice = Explicit_Euler(opt_params, smax, M, N).Cal_Call_Price()
    print('Price of the call spread without the right of early exercise on Mar 4th is:', round(SpreadPrice,2))

    ### 7 ###
    SpreadPrice1 = Explicit_Euler(opt_params, smax, M, N).Cal_Call_Price(earlyEx=True)
    print('Price of the call spread with the right of early exercise on Mar 4th is:', round(SpreadPrice1,2))

    ### 8 ###
    Premium = SpreadPrice1 - SpreadPrice
    print('Premium of the call spread with early exercise right is:', round(Premium,2))

    plt.show()

