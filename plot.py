import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

import globals as gl
import parameters as para

def plot_potential(inf_model, phi0):
    """
    Plot potential
    """
    print("Plotting potentials...")
    phi = np.linspace(-phi0, 2*phi0, 1000)
    V = inf_model.get_V(phi)
    plt.plot(phi, V, color="black", label="$V$")
    plt.plot(phi, inf_model.get_quad(phi), label=r"$\sim \phi^2$")
    plt.plot(phi, np.fabs(inf_model.get_cub(phi)), label=r"$\sim |\phi^3|$")
    plt.plot(phi, inf_model.get_quar(phi), label=r"$\sim \phi^4$")
    #  plt.plot([-phi0, 2*phi0], [0, 0], color="black")
    #  plt.plot(phi, SFPI.get_V_p(phi))
    plt.xlabel(r"$\phi/m_{pl}$")
    plt.ylabel(r"$V(\phi)/m_{pl}^4$")
    plt.legend()
    plt.savefig("./figs/potential-phi0=" + str(phi0) + ".pdf",
                bbox_inches="tight")
    plt.close()


def draw_phi_find_N_tachy(inf_model, t, phi, xlims=False):
    '''
    Plotting phi and dots over phi0/3
    returns number of times return to tachyonic region
    '''
    print("Plotting phi...")

    H0 = inf_model.get_H_inf()
    phi0 = inf_model.get_phi0()
    # rescaled variables, more readable
    t_re = t * H0
    phi_re = phi / phi0

    plt.plot(t_re, phi_re)
    plt.plot([t_re[0], t_re[-1]], [1/3, 1/3], linestyle="--", color="grey")
    # _ is throwaway; find_peaks returns indices
    peaks, _ = find_peaks(phi_re, height=1/3)
    #  print(peaks.shape[0])
    plt.scatter(t_re[peaks], phi_re[peaks], color="red")

    plt.ylabel(r"$\langle \phi \rangle / \phi_0$")
    plt.xlabel(r"$t \cdot H_0$")
    plt.ylim(-0.5, 1.1)
    if xlims is not False:
        plt.xlim(*xlims)
    plt.savefig("./figs/phi-phi0=" + str(phi0) + ".pdf", bbox_inches="tight")
    plt.close()
    return peaks.shape[0]


def draw_para(inf_model, t, phi, phi_dot, xlims=False):
    '''
    Plotting parameters
    '''
    print("Plotting parameters...")
    phi0 = inf_model.get_phi0()
    phi_ddot = np.diff(phi_dot)/np.diff(t)

    # rescaled variables, more readable
    H0 = inf_model.get_H_inf()
    t_re = t * H0

    # slow roll parameters
    eta = para.get_SR_eta(phi, inf_model)
    eps = para.get_SR_epsilon(phi, inf_model)

    # 1st hubble SR para
    hubble = gl.get_Hubble(phi, phi_dot, inf_model.get_V(phi))
    hubble_dot = np.diff(hubble)/np.diff(t)
    hubble_SR = para.get_Hubble_SR_para(hubble[:-1], hubble_dot)

    # aux parameter for slow roll conditions
    a1 = phi_ddot/((3*hubble*phi_dot)[:-1])
    a2 = phi_dot**2/2/inf_model.get_V(phi)

    plt.plot(t_re, np.fabs(eta), label=r"$|\eta_{SR}|$")
    plt.plot(t_re, eta, label=r"$\eta_{SR}$")
    plt.plot(t_re, eps, label=r"$\varepsilon_{SR}$", color="red")
    plt.plot(t_re[:-1], hubble_SR, label=r"$\varepsilon_{H}$", color="black", linestyle="--")
    #  plt.plot(t_re[:-1], a1, label=r"$\ddot{\phi}/3H\dot{\phi}$")
    #  plt.plot(t_re, a2, label=r"$\dot{\phi}^2/2/V$")

    plt.legend()
    plt.xlabel(r"$t \cdot H_0$")
    plt.ylabel(r"parameters")
    plt.ylim(-2, 2)
    if xlims is not False:
        plt.xlim(*xlims)
    plt.savefig("./figs/para-phi0=" + str(phi0) + ".pdf", bbox_inches="tight")
    plt.close()


def power_law(x, a, b):
    return a * x**b


def plot_N_tachy():
    '''
    plot number of times the background field enters the tachyonic region
    '''
    # read phi0 and N_tachy
    phi0_list = np.genfromtxt("./data/N_tachy.dat").T[0]
    N_tachy_list = np.genfromtxt("./data/N_tachy.dat").T[1]

    # only take positive N_tachy
    phi0_list = phi0_list[N_tachy_list > 0]
    N_tachy_list = N_tachy_list[N_tachy_list > 0]

    # plot and fit N_tachy
    plt.scatter(phi0_list, N_tachy_list, label="from ODEs' solutions")
    popt, perr = curve_fit(power_law, phi0_list, N_tachy_list)
    fit_label = r"fit function $\log N = %.2f \log (\phi_0/m_{pl}) + (%.2f) $" % (popt[1], popt[0])

    plt.plot(phi0_list, power_law(phi0_list, *popt), 'r-',
             color="black", label=fit_label)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\phi_0/m_{pl}$")
    plt.ylabel(r"$N$")
    plt.legend()
    plt.savefig("./figs/N_tachy.pdf", bbox_inches="tight")
    plt.close()
