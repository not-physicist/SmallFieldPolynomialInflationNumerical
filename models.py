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

        # omegaStar as new unit for time
        V0 = self.__d*(phi0**4 + self.__A*(1-self.__beta)*phi0**3
                       + 9/32.0*self.__A**2*phi0**2)
        self.__H0 = np.sqrt(V0/3)

        # correction factor for H0
        #  self.__alpha = np.sqrt(3) / phi0**2 \
            #  * np.sqrt(phi0**4
                      #  + self.__A*(1-self.__beta)*phi0**3
                      #  + 9/32 * self.__A**2 * phi0**2)
        self.__alpha = 1
        self.__H0 = 8.6e-9 * phi0**3 * self.__alpha
        self.__omegaStar = self.__H0 / phi0
        #  print("omegaStar = %e m_pl" % (self.__omegaStar))

        # rescaled parameters
        # c_re contains beta !!!
        self.__b_re = self.__d * 9/32 * self.__A**2 / self.__omegaStar**2
        self.__c_re = self.__d * self.__A \
            * (1 - self.__beta) * phi0 / self.__omegaStar**2
        self.__d_re = self.__d * phi0**2 / self.__omegaStar**2
        #  print("b_re = %.10f, c_re = %.10f, d_re = %.10f"
              #  % (self.__b_re, self.__c_re, self.__d_re))

        # by default ODE parameters are not set
        self.__ODE_para_set = False

    def print(self):
        # print parameter info
        if self.__ODE_para_set:
            print(
'''Current inflation model:
    Small Field Polynomial Inflation
with the parameter:
    d = %2.2e, beta = %2.2e, phi0 = %2.2e /M_pl
    phi_i=%e, t_max=%e, N_t=%e'''
                %
                (self.__d, self.__beta, self.__phi0, self.__phi_i,
                self.__t_max/self.get_H_inf(), self.__N_t))
        else:
            print(
'''Current inflation model:
    Small Field Polynomial Inflation
with the parameter:
    d = %2.2e, beta = %2.2e, phi0 = %2.2e /M_pl
    ODE parameters not set'''
                %
                (self.__d, self.__beta, self.__phi0, self.__phi_i))

    def print_orig(self):
        print("Original parameters: b=%2.2e M_pl^2, c=%2.2e M_pl, d=%2.2e"
              % (self.__b, self.__c, self.__d))

    #######################################################################
    # in reduced planck unit
    def get_b(self):
        return self.__b

    def get_c(self):
        return self.__c

    def get_d(self):
        return self.__d

    def get_phi0(self):
        return self.__phi0

    def get_H_inf(self):
        # get scale of inflation
        return self.__H0

    def get_omegaStar(self):
        return self.__omegaStar

    def get_alpha(self):
        return self.__alpha

    #######################################################################

    #######################################################################
    # in phi0, omegaStar units
    def get_V(self, phi):
        # return inflaton potential
        V = self.__b_re * phi**2 \
            + self.__c_re * phi**3 \
            + self.__d_re * phi**4
        return V

    def get_V_p(self, phi):
        V_p = 2 * self.__b_re * phi \
            + 3 * self.__c_re * phi**2 \
            + 4 * self.__d_re * phi**3
        return V_p

    def get_V_pp(self, phi):
        V_pp = 2 * self.__b_re \
            + 6 * self.__c_re * phi \
            + 12 * self.__d_re * phi**2
        return V_pp

    def get_phi_end(self):
        return (1-self.__phi0**2/24.0)

    def get_phi_i(self):
        phi_end = (1-self.__phi0**2/24.0)  # end of slow roll
        delta = 1 - phi_end  # difference between phi0 and phi_end
        phi_i = phi_end + delta/2  # go back in time a bit
        return phi_i

    def set_ODE(self, t_max, N_t):
        # set ODE parameters
        # phi_i in units of phi0
        # t_max in units of omegaStar
        self.__phi_i = self.get_phi_i()
        self.__t_max = t_max
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
