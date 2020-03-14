from Quadratures import *


if __name__ == '__main__':

    ##### Problem 2 #####

    vol1 = 20
    vol2 = 15
    rho = 0.95
    SPY_0 = 321.73
    r = 0
    T1 = 1
    T2 = 0.5

    N = 1000

    ### 1
    K1 = 370
    option2_1 = Quadra_pricing_normal_S(SPY_0, K1, r, T1, vol1)
    print('\nVanilla Option Price: {}'.format(option2_1.Mid_Method(N, 5)))

    ### 2
    K2 = 365

    option2_2 = Quadra_pricing_Contingent(SPY_0, r, K1, K2, vol1, vol2, T1, T2, rho)
    print('\nContingent Option Price: {}\n'.format(option2_2.Mid_Method(N, 5)))

    ### 3
    for rho in [0.8, 0.5, 0.2]:
        option2_3 = Quadra_pricing_Contingent(SPY_0, r, K1, K2, vol1, vol2, T1, T2, rho)
        print('When rho is {}, Contingent Option Priceis : {}'.format(rho, option2_3.Mid_Method(N, 5)))

    ### 5
    rho = 0.95
    for K2 in [360, 350, 340]:
        option2_5 = Quadra_pricing_Contingent(SPY_0, r, K1, K2, vol1, vol2, T1, T2, rho)
        print('When K2 is {}, Contingent Option Priceis : {}'.format(K2, option2_5.Mid_Method(N, 5)))

