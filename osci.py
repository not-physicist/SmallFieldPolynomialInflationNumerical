import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def find_phi_i(model):
    # find phi after oscillation; energy lose due to Hubble friction
    # deprecated, use numerical instead...
    b, c, d = model.get_coeff()
    V0 = model.get_V(model.phi0)
    a = -V0*(1 + 2*np.sqrt(6)*model.phi0)

    poly = np.polynomial.Polynomial((a, 0, b, c, d))
    roots = poly.roots()
    for i in roots:
        if i < 0 and np.iscomplex(i) is False:
            phi_1 = np.real(i)
            print("phi_1 = %f2.2" % (phi_1))


def find_period(t, phi):
    '''
    find period of inflaton oscllation
    '''
    # take absolute value <=> find peaks and troughs
    abs_phi = np.fabs(phi)
    peaks, _ = find_peaks(abs_phi)

    #  plt.scatter(t[peaks], phi[peaks], color="red")
    #  plt.plot(t, phi)
    #  plt.show()

    T = 2 * np.diff(t[peaks])
    # peak to trough is only half period
    print("Periods of inflaton oscillation are", T)
    print("Relative variance in periods", np.var(T)/np.mean(T)**2)
