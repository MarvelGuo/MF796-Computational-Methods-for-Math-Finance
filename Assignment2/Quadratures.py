import numpy as np

class Quadra_pricing():
    def __init__(self, S0, K, r, T, sigma):
        self.K = K
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.norm_miu = T * (r - 0.5 * sigma**2) + np.log(S0)
        self.norm_sigma = sigma * np.sqrt(T)

    def Norm_pdf(self, x):
        # input x here is actually log(x)
        denominator = 1 / np.sqrt(2 * np.pi) / self.norm_sigma
        exp_inside = -0.5 * ((x - self.norm_miu) / self.norm_sigma)**2
        return np.exp(exp_inside) * denominator

    def Standard_Norm_pdf(self, x):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2)

    def Call_pricing(self, x):
        return (np.exp(x) - self.K) * self.Norm_pdf(x)

    def Left_Method(self, N, n_dev):
        lower = np.log(self.K)
        upper = self.norm_miu + n_dev * self.norm_sigma
        Payoff = Quadratures.Left_Riemann(self.Call_pricing, lower, upper, N)
        C = np.exp(-self.r * self.T) * Payoff
        return C

    def Mid_Method(self, N, n_dev):
        lower = np.log(self.K)
        upper = self.norm_miu + n_dev * self.norm_sigma
        Payoff = Quadratures.Mid_Point_Riemann(
            self.Call_pricing, lower, upper, N)
        C = np.exp(-self.r * self.T) * Payoff
        return C

    def Gaussian_Legendre_Method(self, N, n_dev):
        lower = np.log(self.K)
        upper = self.norm_miu + n_dev * self.norm_sigma
        Payoff = Quadratures.Gaussian_Legendre(
            self.Call_pricing, lower, upper, N)
        C = np.exp(-self.r * self.T) * Payoff
        return C

    # def Left_Method(self, N):
    #     N_d1 = Quadratures.Left_Riemann(self.Norm_pdf, -10, self.d1, N)
    #     N_d2 = Quadratures.Left_Riemann(self.Norm_pdf, -10, self.d2, N)
    #     C = self.S0 * N_d1 - self.K * np.exp(-self.r * self.T) * N_d2
    #     return C

    # def Mid_Method(self, N):
    #     N_d1 = Quadratures.Mid_Point_Riemann(self.Norm_pdf, -10, self.d1, N)
    #     N_d2 = Quadratures.Mid_Point_Riemann(self.Norm_pdf, -10, self.d2, N)
    #     C = self.S0 * N_d1 - self.K * np.exp(-self.r * self.T) * N_d2
    #     return C

class Quadra_pricing_normal_S(Quadra_pricing):
    '''
    This class is used to pricing the option under assumption that S is normal distribution
    '''
    def __init__(self, S0, K, r, T, sigma):
        super().__init__(S0, K, r, sigma, T)
        self.norm_miu = S0
        self.norm_sigma = sigma

    def Call_pricing(self, x):
        return (x - self.K) * self.Norm_pdf(x)

    def Mid_Method(self, N, n_dev):
        lower = self.K
        upper = self.norm_miu + n_dev * self.norm_sigma
        Payoff = Quadratures.Mid_Point_Riemann(
            self.Call_pricing, lower, upper, N)
        C = np.exp(-self.r * self.T) * Payoff
        return C

class Quadra_pricing_Contingent():
    '''
    The comment parts is used to calculate the case when ln(s) is normal distribution
    '''

    def __init__(self, S0, r, K1, K2, sigma1, sigma2, T1, T2, rho):
        self.S0 = S0
        self.r = r
        self.K1 = K1
        self.K2 = K2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.T1 = T1
        self.T2 = T2
        self.rho = rho
        self.miu1 = S0
        self.miu2 = S0
        # self.norm_miu1 = T1 * (r - 0.5 * sigma1**2) + np.log(S0)
        # self.norm_sigma1 = sigma1 * np.sqrt(T1)
        # self.norm_miu2 = T2 * (r - 0.5 * sigma2**2) + np.log(S0)
        # self.norm_sigma2 = sigma2 * np.sqrt(T2)

    def Bivariate_Norm_pdf(self, x1, x2):
        coef = 1 / (2 * np.pi * self.sigma1 *
                    self.sigma2 * np.sqrt(1 - self.rho**2))

        stand_1 = ((x1 - self.miu1) / self.sigma1) ** 2
        stand_2 = ((x2 - self.miu2) / self.sigma2) ** 2
        stand_common = 2 * self.rho * \
            (x1 - self.miu1) * (x2 - self.miu2) / \
            (self.sigma1 * self.sigma2)

        exp_inside = -1 / 2 / (1 - self.rho**2) * \
            (stand_1 + stand_2 - stand_common)

        return coef * np.exp(exp_inside)

    def Contingent_Call_Pricing(self, x1, x2):
        return (x1 - self.K1) * self.Bivariate_Norm_pdf(x1, x2)

    def Mid_Method(self, N, n_dev):
        lower1 = self.K1
        upper1 = self.miu1 + n_dev * self.sigma1
        lower2 = self.miu2 - n_dev * self.sigma2
        upper2 = self.K2

        payoff = 0
        logS1_points = Quadratures.get_Mid_Points(lower1, upper1, N)
        logS2_points = Quadratures.get_Mid_Points(lower2, upper2, N)

        for x2 in logS2_points:
            integrals = self.Contingent_Call_Pricing(logS1_points, x2)
            payoff += np.sum((upper1 - lower1) / N *
                             integrals) * (upper2 - lower2) / N

        C = np.exp(-self.r * self.T1) * payoff
        return C

class Quadratures:
    def Left_Riemann(f, a, b, N):
        x1 = a
        xN = a + (b - a) / N * (N - 1)
        x = np.linspace(x1, xN, N)
        return Quadratures.Riemann_Rule(f, x, a, b, N)

    def get_Mid_Points(a, b, N):
        x1 = a + (b - a) / N * 1 / 2
        xN = a + (b - a) / N * (N - 1 / 2)
        x = np.linspace(x1, xN, N)
        return x

    def Mid_Point_Riemann(f, a, b, N):
        x = Quadratures.get_Mid_Points(a, b, N)
        return Quadratures.Riemann_Rule(f, x, a, b, N)

    def Right_Riemann(f, a, b, N):
        x1 = a + (b - a)
        xN = b
        x = np.linspace(x1, xN, N)
        return Quadratures.Riemann_Rule(f, x, a, b, N)

    def Riemann_Rule(f, x, a, b, N):
        w = (b - a) / N
        return np.sum(w * f(x))

    def get_Gaussian_Legendre(a, b, N):
        x, w = np.polynomial.legendre.leggauss(N)
        y = 0.5 * x * (b - a) + (a + b) / 2
        return y, w

    def Gaussian_Legendre(f, a, b, N):
        x, w = Quadratures.get_Gaussian_Legendre(a, b, N)
        return np.sum(w * f(x)) * 0.5 * (b - a)


if __name__ == '__main__':
    a = -10
    b = 0

    for N in range(5, 20):
        error = abs(Quadratures.Mid_Point_Riemann(Norm_pdf, a, b, N) - 0.5)
        print(round(error * 10**5, 2))
