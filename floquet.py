import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import newton
from scipy.integrate import quad
from scipy.integrate import solve_ivp
import os

import ODE
import models
import globals as gl

'''
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
'''


def find_period(t, phi):
    '''
    find period of inflaton oscllation
    '''
    # take absolute value <=> find peaks and troughs
    abs_phi = np.fabs(phi)
    peaks, _ = find_peaks(abs_phi, height=[0, 1])

    T = 2 * np.diff(t[peaks])
    # peak to trough is only half period
    print("Periods of inflaton oscillation are", np.mean(T))
    print("Relative variance in periods", np.var(T)/np.mean(T)**2)

'''
def find_phi_min(inf_model):
    # find the minimal field value from the potential
    phi0 = inf_model.get_phi0()
    V0 = inf_model.get_V(phi0)
    phi_min = newton(lambda phi: V0 - inf_model.get_V(phi), -0.3*phi0)
    return phi_min


def find_period_integral(inf_model):
    phi_max = inf_model.get_phi0()
    phi_min = find_phi_min(inf_model)
    V0 = inf_model.get_V(phi_min)

    T = quad(lambda phi: np.sqrt(2/V0 - inf_model.get_V(phi)), phi_min, phi_max)
    return T
'''


def solve_fund_matrix(k, inf_model, t0, t1, get_phi):
    def solve_fund_matrix_aux(t, y, k):
        fld = y[0]
        pi = y[1]

        dydt = [pi, (-k**2 - inf_model.get_V_pp(get_phi(t)))*fld]
        #  print(t)
        return dydt

    T = t1 - t0  # in omega* unit, period is roughly 1
    sol1 = solve_ivp(lambda t, y: solve_fund_matrix_aux(t, y, k), [t0, t1], [1, 0])
    #  fld1 = sol1.y[0, -1]
    #  pi1 = sol1.y[1, -1]
    sol2 = solve_ivp(lambda t, y: solve_fund_matrix_aux(t, y, k), [t0, t1], [0, 1])
    #  fld2 = sol2.y[0, -1]
    #  pi2 = sol2.y[1, -1]
    #  print(sol1, sol2)
    O = np.array([[sol1.y[0, -1], sol2.y[0, -1]],
                  [sol1.y[1,-1], sol2.y[1, -1]]])
    #  print(O)
    o = np.linalg.eigvals(O)
    #  print(o)

    #  o_p = (fld1 + pi2)/2 + np.sqrt((fld1 - pi2)**2 + 4*fld2*pi1)/2
    #  o_m = (fld1 + pi2)/2 - np.sqrt((fld1 - pi2)**2 + 4*fld2*pi1)/2
    #  print(o_p, o_m, np.log(np.abs([o_p, o_m])/T))

    flo_coef = np.log(np.abs(o))/T
    return flo_coef


def compute_flo(inf_model):
    # get phi(t)
    t_range = [0, 50]
    popt = np.genfromtxt("./data/1st_peaks_fit.dat")
    phi_i = gl.power_law(inf_model.get_phi0(), *popt)
    sol = solve_ivp(lambda t, y: [y[1], -inf_model.get_V_p(y[0])],
                    t_range, [phi_i, 0], max_step=0.01)
    t = sol.t
    phi = sol.y[0]

    def get_phi(x):
        return np.interp(x, t, phi)

    # find first and second trough
    abs_phi = np.fabs(phi)
    peaks, _ = find_peaks(abs_phi, height=[0, 1])
    t0 = t[peaks[0]]
    t1 = t[peaks[2]]
    #  print(phi[peaks[0]], phi[peaks[2]])
    #  print(t[peaks[0]], t[peaks[2]])
    #  find_period(t, phi)
    '''
    plt.plot(t, phi)
    plt.scatter(t[peaks], phi[peaks], color="red")
    plt.show()
    '''
    m_min = np.sqrt(np.abs(inf_model.get_V_pp(2/3)))
    #  k = np.logspace(-1, 1, num=100, base=10)
    k = np.linspace(0, 3, num=100)
    R_mu = np.zeros(k.shape[0])
    for i in range(0, k.shape[0]):
        k_i = k[i]
        coeffs = solve_fund_matrix(k_i*m_min, inf_model, t0, t1, get_phi)
        print(k_i, coeffs)

        if coeffs[0] * coeffs[1] < 0:  # different sign
            if coeffs[0] != - coeffs[1]:
                pass
                #  print("Inconsistency encountered in floquent coefficients!")
                # TODO: don't know why this happens a lot
                #  return ValueError
            R_mu[i] = coeffs[0] if coeffs[0] > 0 else coeffs[1]
        else:
            R_mu[i] = 0
        #  plt.scatter(k_i, R_mu[i], color="black")
    #  plt.ylabel("$\Re{\mu}/\omega_*$")
    #  plt.xlabel("$k/m$")
    #  plt.show()
    return k, R_mu


def save_flo():
    fn = "./data/floquent.dat"
    if os.path.exists(fn):
        os.remove(fn)
    f = open(fn, 'ab')

    phi0_array = np.logspace(-4, 0, 500, base=10)
    for i in range(0, phi0_array.shape[0]):
        print(i, phi0_array.shape[0])
        phi0 = phi0_array[i]
        SFPI = models.SFPInf(phi0)
        k, flo = compute_flo(SFPI)
        np.savetxt(f, [flo])
    f.close()

    # k should be the same for all
    fn_k = "./data/floquent_k.dat"
    np.savetxt(fn_k, k)

    fn_phi0 = "./data/floquent_phi0.dat"
    np.savetxt(fn_phi0, phi0_array)


def plot_flo():
    phi0_array = np.genfromtxt("./data/floquent_phi0.dat")
    k_array = np.genfromtxt("./data/floquent_k.dat")
    flo_big_array = np.genfromtxt("./data/floquent.dat")

    if flo_big_array.shape[0] != phi0_array.shape[0]:
        print("Something is wrong!")
        print(flo_big_array.shape, phi0_array.shape)
        return ValueError

    heatmap = flo_big_array
    extent = [k_array[0], k_array[-1], phi0_array[0], phi0_array[-1]]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plt.clf()
    im = plt.imshow(heatmap, extent=extent)
    plt.yscale("log")

    ax.set_xlabel(r"k/m")
    ax.set_ylabel(r"$\phi_0/m_{pl}$")

    # colorbar
    cax = plt.axes([0.75, 0.1, 0.04, 0.8])  # [left, bottom, width, height]
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(r"$\Re(\mu)/\omega_*$")

    plt.savefig("./figs/floquent_heatmap.pdf", bbox_inches="tight")
