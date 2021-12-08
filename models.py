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
        self.__c = -8*self.__d*self.__phi0/3
        self.__b = 9.0/32*self.__c**2/self.__d
        self.__A = -8.0/3.0 * phi0

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
            print("Error: ODE parameters not set!")
            return ValueError

    def print_orig(self):
        print("Original parameters: b=%2.2e M_pl^2, c=%2.2e M_pl, d=%2.2e"
              % (self.__b, self.__c, self.__d))

    def get_b(self):
        return self.__b

    def get_c(self):
        return self.__c

    def get_d(self):
        return self.__d

    def get_phi0(self):
        return self.__phi0

    def get_V(self, phi):
        # return inflaton potential at given field value with given parameters
        V = self.__d*(phi**4 + self.__A*(1-self.__beta)*phi**3
                      + 9/32.0*self.__A**2*phi**2)
        return V

    def get_V_p(self, phi):
        V_p = self.__d*(4*phi**3 + 3*self.__A*(1-self.__beta)*phi**2
                        + 9/16.0*self.__A**2*phi)
        #  print("%e" % V_p)
        return V_p

    def get_V_pp(self, phi):
        return self.__d*(12*phi**2 + 6*self.__A*(1-self.__beta)*phi
                         + 9/16.0*self.__A**2)

    def get_phi_end(self):
        # get phi at end of slow-roll inflation
        return (1-self.__phi0**2/24.0)*self.__phi0

    def set_ODE(self, t_max, N_t):
        # set ODE parameters
        # t_max in units of H0
        self.__phi_i = self.get_phi_end()
        self.__t_max = t_max / self.get_H_inf()
        self.__N_t = int(N_t)
        self.__ODE_para_set = True

    def get_ODE(self):
        if self.__ODE_para_set:
            return [self.__phi_i, self.__t_max, self.__N_t]
        else:
            print("Error: ODE parameters not set!")
            return ValueError

    def get_phi_i(self):
        return self.__phi_i

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
        return self.__d * 9.0/32.0 * self.__A**2 * phi**2

    def get_cub(self, phi):
        return self.__d * self.__A * (1-self.__beta) * phi**3

    def get_quar(self, phi):
        return self.__d*phi**4

    def get_quad_perc(self, phi):
        # get percentage of contribution of terms
        return (self.get_quad(phi),
                self.get_cub(phi),
                self.get_quar(phi))/self.get_V(phi)

    def get_H_inf(self):
        # get scale of inflation
        return np.sqrt(self.get_V(self.__phi0)/3)
