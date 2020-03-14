from Option_Pricing import *

if __name__ == '__main__':

    K = 100
    S0 = 100
    r = 0
    sigma = 0.25
    beta = 1
    T = 1

    steps = 250
    simu_times = 1000

    # b
    option_1 = Euro_option(K, S0, r, sigma, T)
    simu_price, simu_payoff, simu_ST = option_1.price_simu(
        steps, simu_times, 'CEV', beta, seed=1, plot=True)
    print('b) Price through simulation is: {}'.format(simu_price['Call']))

    # c
    BSM_price, delta = option_1.BSM()
    print('\nc) Price through BSM is:{}'.format(BSM_price['Call']))

    # d
    print('\nd) Delta is:{}'.format(delta))

    # e
    hdg_shares = - delta

    # f
    simu_portfolio_pay = np.mean(
        simu_payoff['Call'] + hdg_shares * (simu_ST - S0))
    print('\nf) Payoff of Delta Neutral portfolio is:{}'.format(
        simu_portfolio_pay))

    # g
    _, simu_payoff_1, simu_ST_1 = option_1.price_simu(
        steps, simu_times, 'CEV', beta=0.5, seed=1)
    simu_portfolio_pay_1 = np.mean(
        simu_payoff_1['Call'] + hdg_shares * (simu_ST_1 - S0))
    print('\ng) When beta is 0.5, payoff of Delta Neutral portfolio is:{}'.format(
        simu_portfolio_pay_1))

    # h
    sigma_2 = 0.4
    option_2 = Euro_option(K, S0, r, sigma_2, T)
    _, simu_payoff_2, simu_ST_2 = option_2.price_simu(
        steps, simu_times, 'CEV', beta, seed=1)
    _, delta_2 = option_2.BSM()

    hdg_shares_2 = - delta_2
    simu_portfolio_pay_2 = np.mean(
        simu_payoff_2['Call'] + hdg_shares_2 * (simu_ST_2 - S0))
    print('\nh) When sigma is 0.4, new delta is:{}, payoff of Delta Neutral portfolio is:{}'.format(
        delta_2, simu_portfolio_pay_2))

    plt.show()
