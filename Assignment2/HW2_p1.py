from Option_Pricing import *
from Quadratures import *
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


if __name__ == '__main__':

    ##### Problem 1 #####

    K = 12
    S0 = 10
    r = 0.04
    sigma = 20 / 100
    T = 1 / 4

    # 1
    option1 = Euro_option(K, S0, r, sigma, T)
    Call_BSM = option1.BSM()['Call']
    print('Call price of BSM is {}'.format(Call_BSM))

    # 2
    error_dict = defaultdict(list)
    option1_quadra = Quadra_pricing(S0, K, r, T, sigma)
    for N in [5, 10, 50, 100]:
        error_dict[N].append(abs(Call_BSM - option1_quadra.Left_Method(N, 5)))
        error_dict[N].append(abs(Call_BSM - option1_quadra.Mid_Method(N, 5)))
        error_dict[N].append(
            abs(Call_BSM - option1_quadra.Gaussian_Legendre_Method(N, 5)))

    error = pd.DataFrame.from_dict(
        error_dict, orient='index', columns=[
            'Left', 'Middle', 'Gaussian'])
    print(round(error, 4))

    # 3
    error_dict = defaultdict(list)
    option1_quadra = Quadra_pricing(S0, K, r, T, sigma)
    for N in range(1, 30):
        error_dict[N].append(abs(Call_BSM - option1_quadra.Left_Method(N, 5)))
        error_dict[N].append(abs(Call_BSM - option1_quadra.Mid_Method(N, 5)))
        error_dict[N].append(
            abs(Call_BSM - option1_quadra.Gaussian_Legendre_Method(N, 5)))

    error = pd.DataFrame.from_dict(
        error_dict, orient='index', columns=[
            'Left', 'Middle', 'Gaussian'])

    x_n = np.array(range(1, 11))

    error['Left'].plot()
    plt.plot(x_n, 1 / x_n**1)
    plt.plot(x_n, 1 / x_n**2)
    plt.plot(x_n, 1 / x_n**3)
    plt.legend(['$Left$', '$O(N^{-1})$', '$O(N^{-2})$', '$O(N^{-3})$'])
    plt.title('Fig 1: Left_Riemann vs Different Orders')

    plt.figure()
    error['Middle'].plot()
    plt.plot(x_n, 1 / x_n**1)
    plt.plot(x_n, 1 / x_n**2)
    plt.plot(x_n, 1 / x_n**3)
    plt.legend(['$Mid$', '$O(N^{-1})$', '$O(N^{-2})$', '$O(N^{-3})$'])
    plt.title('Fig 2: Middle_Riemann vs Different Orders')

    plt.figure()
    x_n = x_n[:11]
    error['Gaussian'].iloc[:11].plot()
    plt.plot(x_n, 1 / x_n ** x_n)
    plt.plot(x_n, 1 / x_n ** (2 * x_n))
    plt.legend(['$Gaussian$', '$O(N^{-N})$', '$O(N^{-2N})$'])
    plt.title('Fig 3: Gaussian vs Different Orders')

    plt.show()




