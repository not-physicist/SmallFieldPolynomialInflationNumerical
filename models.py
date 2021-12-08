################################################################
# class for inflaton potential
################################################################
import numpy as np

# in reduced planck unit


class SFPInf:
    "SFPI model with 2, 3, 4 order monomial"
    def __init__(self, phi0):
        # here these parameters are in m_pl
        if phi0 > 1 or phi0 < 0:
            print("Error: phi0 out of range!")
            return ValueError
        self.__phi0 = phi0
        self.__d = phi0**2 * 6.61e-16
        self.__beta = phi0**4 * 9.73e-7
        self.__A = -8.0/3.0 * phi0
        self.__c = -8*self.__d*self.__phi0/3
        self.__b = 9.0/32*self.__c**2/self.__d

        # H0 as new unit
        V0 = self.__d*(phi0**4 + self.__A*(1-self.__beta)*phi0**3
                       + 9/32.0*self.__A**2*phi0**2)
        self.__H0 = np.sqrt(V0/3)

        # rescaled parameters
        self.b_re = self.__b / self.__H0**2
        self.c_re = self.__c * phi0 / self.__H0**2
        self.d_re = self.__d * phi0**2 / self.__H0**2

        # by default ODE parameters are not set
        self.__ODE_para_set = False

    def print(self):
        # print parameter info
        if self.__ODE_para_set:
            print(
'''
Current inflation model:
    Small Field Polynomial Inflation
with the parameter:
    d = %2.2e, beta = %2.2e, phi0 = %2.2e /M_pl
    phi_i=%e, t_max=%e, N_t=%e
'''
                %
                (self.__d, self.__beta, self.__phi0, self.__phi_i,
                self.__t_max/self.get_H_inf(), self.__N_t))
        else:
            print(
'''
Current inflation model:
    Small Field Polynomial Inflation
with the parameter:
    d = %2.2e, beta = %2.2e, phi0 = %2.2e /M_pl
    ODE parameters not set
'''
                %
                (self.__d, self.__beta, self.__phi0, self.__phi_i))

    def print_orig(self):
        print("Original parameters: b=%2.2e M_pl^2, c=%2.2e M_pl, d=%2.2e"
              % (self.__b, self.__c, self.__d))

    '''
    def get_b(self):
        return self.__b

    def get_c(self):
        return self.__c

    def get_d(self):
        return self.__d
    '''
    #######################################################################
    # in reduced planck unit
    def get_phi0(self):
        return self.__phi0

    def get_H_inf(self):
        # get scale of inflation
        return self.__H0

    def get_V(self, phi):
        # return inflaton potential
        V = self.__b_re * phi**2 + self.__c_re * phi**3 + self.__d_re * phi**4
        return V
    #######################################################################

    #######################################################################
    # in phi0, H0 units
    def get_V_p(self, phi):
        V_p = 2 * self.__b_re * phi \
            + 3 * self.__c_re * phi**2 \
            + 4 * self.__d_re * phi**3
        #  print("%e" % V_p)
        return V_p

    def get_V_pp(self, phi):
        V_pp = 2 * self.__b_re \
            + 6 * self.__c_re * phi \
            + 12 * self.__d_re * phi**2
        return V_pp

    def get_phi_i(self):
        # get phi at end of slow-roll inflation
        # as initial field value
        return (1-self.__phi0**2/24.0)

    def set_ODE(self, t_max, N_t):
        # set ODE parameters
        # phi_i in units of phi0
        # t_max in units of H0
        self.__phi_i = self.get_phi_i()
        self.__t_max = t_max / self.get_H_inf()
        self.__N_t = int(N_t)
        self.__ODE_para_set = True

    def get_ODE(self):
        if self.__ODE_para_set:
            return [self.__phi_i, self.__t_max, self.__N_t]
        else:
            print("Error: ODE parameters not set!")
            return ValueError

    def get_t_max(self):
        if self.__ODE_para_set:
            return self.__t_max
        else:
            print("Error: ODE parameters not set!")
            return ValueError

    def get_N_t(self):
        if self.__ODE_para_set:
            return self.__N_t
        else:
            print("Error: ODE parameters not set!")
            return ValueError

    def get_t_step(self):
        if self.__ODE_para_set:
            return self.__t_max / self.__N_t
        else:
            print("Error: ODE parameters not set!")
            return ValueError

    '''
    def get_phi(self, delta):
        # get phi from delta
        return self.phi0*(1-delta)

    def get_delta(self, phi):
        # get delta from phi
        return 1-phi/self.phi0

    def get_coeff(self):
        # return coefficents of polynomial
        return self.b, self.c, self.d
    '''
    def get_quad(self, phi):
        # get only quadratic contribution
        return self.__b_re * phi**2

    def get_cub(self, phi):
        return self.__c_re * phi**3

    def get_quar(self, phi):
        return self.__d_re*phi**4

    def get_quad_perc(self, phi):
        # get percentage of contribution of terms
        return (self.get_quad(phi),
                self.get_cub(phi),
                self.get_quar(phi))/self.get_V(phi)
    #######################################################################
