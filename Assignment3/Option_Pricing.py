#!/usr/bin/env python
# coding: utf-8
import sys
from os import path
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.style.use('ggplot')


class Simulation():
    def __init__(self, steps, simu_times):
        self.steps = steps
        self.simu_times = simu_times
        self.dt = 1 / self.steps

    def generate_simu_paths(self, model, S0, sigma, r, beta, plot, seed):

        if model == 'BSM':
            S = self.Black_Scholes_path(S0, sigma, r, seed)
        elif model == 'Bachelier':
            S = self.Bachelier_path(S0, sigma, r, seed)
        elif model == 'CEV':
            S = self.CEV_path(S0, sigma, r, beta, seed)
        else:
            raise ValueError('No such Simulated Model!!')

        if plot:
            self.plot_path(S, model)
            # plt.show()

        mean_terminal = S.mean(axis=0)[-1]
        var_terminal = S.var(axis=0)[-1]

        return S[:, -1], mean_terminal, var_terminal

    def Black_Scholes_path(self, S0, sigma, r, seed=None):
        if not seed:
            np.random.seed(seed)
        random_num = np.random.normal(
            0, 1 / 252 ** 0.5, [self.simu_times, self.steps])
        move = random_num * sigma + r * 1 / 252

        S = np.zeros([self.simu_times, self.steps])
        S[:, 0] = S0
        for t in range(1, steps):
            S[:, t] = S[:, t - 1] + S[:, t - 1] * move[:, t - 1]

        return S

    def Bachelier_path(self, S0, sigma, r, seed=None):
        if seed is not None:
            np.random.seed(seed)

        random_num = np.random.normal(
            0, 1 / 252 ** 0.5, [self.simu_times, self.steps])
        move = random_num * sigma + r * 1 / 252

        S = np.zeros([self.simu_times, self.steps])
        S[:, 0] = S0
        for t in range(1, self.steps):
            S[:, t] = S[:, t - 1] + move[:, t - 1]

        return S

    def CEV_path(self, S0, sigma, r, beta=None, seed=None):
        if beta is None:
            raise ValueError('beta should not be NULL')
        if seed is not None:
            np.random.seed(seed)

        dWt = np.random.normal(
            0, self.dt ** 0.5, [self.simu_times, self.steps])

        S = np.zeros([self.simu_times, self.steps])
        S[:, 0] = S0

        for t in range(1, self.steps):
            dSt = S[:, t - 1] * r * 1 / 252 + \
                S[:, t - 1]**beta * dWt[:, t - 1] * sigma
            S[:, t] = S[:, t - 1] + dSt

        return S

    def plot_path(self, paths, path_name):
        plt.figure(figsize=(8, 4), dpi=120)
        plt.plot(paths.T, linewidth=1)
        plt.title('Simulated Paths of Security Price (%s)' % path_name)
        plt.xlabel('steps')
        plt.ylabel('Price')


class Option():
    def __init__(self, K, S0, r, sigma, T):
        self.K = K
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T

    def simulated_paths(self, steps, simu_times, model, beta, seed, plot):
        simulation = Simulation(steps, simu_times)
        ST, *_ = simulation.generate_simu_paths(
            model, self.S0, self.sigma, self.r, beta, plot, seed)
        return ST

    def discount(self, x):
        return x * np.exp(- self.r * self.T)

class Euro_option(Option):
    def __init__(self, K, S0, r, sigma, T):
        super().__init__(K, S0, r, sigma, T)
        # Option.__init__(self, K, S0, r, sigma)

    def price_simu(self, steps, simu_times, model,
                   beta=None, seed=None, plot=False):
        ST = self.simulated_paths(steps, simu_times, model, beta, seed, plot)
        pay_put, pay_call = self.payoff_simu(ST)
        price_put = self.discount(np.mean(pay_put))
        price_call = self.discount(np.mean(pay_call))

        payoff = {'Put': np.mean(pay_put), 'Call': np.mean(pay_call)}
        price = {'Put': price_put, 'Call': price_call}
        return price, payoff, ST

    def payoff_simu(self, ST):

        temp_pay_put = self.K - ST
        pay_put = map(lambda x: max(x, 0), temp_pay_put)
        pay_put = np.array(list(pay_put))

        temp_pay_call = ST - self.K
        pay_call = map(lambda x: max(x, 0), temp_pay_call)
        pay_call = np.array(list(pay_call))

        return pay_put, pay_call

    def BSM(self):
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma ** 2 / 2)
              * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)

        C = self.S0 * N_d1 - self.K * np.exp(-self.r * self.T) * N_d2
        P = C + self.K * np.exp(-self.r * self.T) - self.S0

        price = {'Put': P, 'Call': C}
        self.delta = N_d1

        return price


class Lookback_option(Option):
    def __init__(self, K, S0, r, sigma, T):
        super().__init__(K, S0, r, sigma, T)

    def payoff_simu(self, ST, plot=True):

        temp_pay_put = self.K - ST.min(axis=1)
        pay_put = map(lambda x: max(x, 0), temp_pay_put)
        pay_put = np.array(list(pay_put))

        temp_pay_call = ST.max(axis=1) - self.K
        pay_call = map(lambda x: max(x, 0), temp_pay_call)
        pay_call = np.array(list(pay_call))

        return pay_put, pay_call

    # def payoff_hist(self):
    #     plt.figure(figsize=[6, 3], dpi=120)
    #     plt.hist(self.payoff, rwidth=0.8)
    #     plt.title('Distribution of Payoff of Lookback option')
    #     plt.xlabel('Payoff')
    #     plt.ylabel('Frequency')


if __name__ == '__main__':

    K = 100
    S0 = 100
    r = 0
    sigma = 0.25

    steps = 250
    simu_times = 1000

    normal_mean = 0

    print(Euro_option(K, S0, r, sigma, 20 / 250).BSM()[0])
