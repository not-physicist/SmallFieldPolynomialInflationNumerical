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
    phi = np.linspace(-1, 2, 1000)
    V = inf_model.get_V(phi)
    plt.plot(phi, V, color="black", label=r"$\tilde{V}$")
    plt.plot(phi, inf_model.get_quad(phi), label=r"$\sim \phi^2$")
    plt.plot(phi, np.fabs(inf_model.get_cub(phi)), label=r"$\sim |\phi^3|$")
    plt.plot(phi, inf_model.get_quar(phi), label=r"$\sim \phi^4$")
    #  plt.plot([-phi0, 2*phi0], [0, 0], color="black")
    #  plt.plot(phi, SFPI.get_V_p(phi))
    plt.xlabel(r"$\phi/\phi_0$")
    plt.ylabel(r"$\tilde{V}$")
    plt.legend()
    plt.savefig("./figs/potential-phi0=" + str(phi0) + ".pdf",
                bbox_inches="tight")
    plt.close()


def draw_phi_tachy_points(inf_model, t, phi, xlims=False):
    '''
    Plotting phi and dots over phi0/3
    returns number of times return to tachyonic region
    '''
    print("Plotting phi...")
    phi0 = inf_model.get_phi0()
    #  print(t, phi)
    # using mask instead of setting xlims
    # computationally cheaper
    if xlims is not False:
        mask = np.logical_and(t > xlims[0], t < xlims[1])
        t_masked = t[mask]
        phi_masked = phi[mask]
    else:
        t_masked = t
        phi_masked = phi

    plt.plot(t_masked, phi_masked)
    plt.plot([t_masked[0], t_masked[-1]], [1/3, 1/3],
             linestyle="--", color="grey")
    peaks_i = find_tachy_indices(t_masked, phi_masked)
    plt.scatter(t_masked[peaks_i], phi_masked[peaks_i], color="red")

    plt.ylabel(r"$\phi / \phi_0$")
    plt.xlabel(r"$t \cdot \omega_0$")
    plt.ylim(-0.5, 1.1)
    #  plt.ylim(0.999, 1.001)
    plt.savefig("./figs/phi-phi0=" + str(phi0) + ".pdf", bbox_inches="tight")
    plt.close()


def find_tachy_indices(t, phi):
    '''
    find indices of peaks in tachyonic region
    '''
    # _ is throwaway; find_peaks returns indices
    # due to machine precision, sometimes with small phi0,
    #  the field can go up in the beginning
    # To counter this: maximal height set to 1
    peaks, _ = find_peaks(phi, height=[1/3, 1])

    # get rid of last element, may not be a peak/trough
    return peaks[:-1]


def find_N_tachy(t, phi):
    return find_tachy_indices(t, phi).shape[0]


def draw_para(inf_model, t, phi, phi_dot, xlims=False):
    '''
    Plotting parameters
    '''
    print("Plotting parameters...")
    phi0 = inf_model.get_phi0()
    #  H0 = inf_model.get_H_inf()

    # slow roll parameters
    eta = para.get_SR_eta(phi, inf_model)
    eps = para.get_SR_epsilon(phi, inf_model)

    # 1st hubble SR para
    hubble = gl.get_Hubble(phi, phi_dot, inf_model)
    hubble_dot = np.diff(hubble)/np.diff(t)
    hubble_SR = para.get_Hubble_SR_para(hubble[:-1], hubble_dot)

    plt.plot(t, np.fabs(eta), label=r"$|\eta_{SR}|$")
    plt.plot(t, eta, label=r"$\eta_{SR}$")
    plt.plot(t, eps, label=r"$\varepsilon_{SR}$", color="red")
    plt.plot(t[:-1], hubble_SR, label=r"$\varepsilon_{H}$", color="black", linestyle="--")

    '''
    Probably not good idea, keep anyway
    # aux parameter for slow roll conditions
    phi_ddot = np.diff(phi_dot)/np.diff(t)
    a1 = phi_ddot/((3*hubble*phi_dot)[:-1])
    a2 = phi_dot**2/2/inf_model.get_V(phi)
    plt.plot(t[:-1], a1, label=r"$\ddot{\phi}/3H\dot{\phi}$")
    plt.plot(t, a2, label=r"$\dot{\phi}^2/2/V$")
    '''

    plt.legend()
    plt.xlabel(r"$t \cdot \omega_0$")
    plt.ylabel(r"parameters")
    plt.ylim(-2, 2)
    if xlims is not False:
        plt.xlim(*xlims)
    plt.savefig("./figs/para-phi0=" + str(phi0) + ".pdf", bbox_inches="tight")
    plt.close()


def plot_N_tachy():
    '''
    plot number of times the background field enters the tachyonic region
    '''
    # read phi0 and N_tachy
    data = np.genfromtxt("./data/N_tachy.dat").T
    phi0_list = data[0]
    N_tachy_list = data[1]

    # only take positive N_tachy, exclude zeros
    phi0_list = phi0_list[N_tachy_list > 0]
    N_tachy_list = N_tachy_list[N_tachy_list > 0]

    # plot and fit N_tachy
    plt.scatter(phi0_list, N_tachy_list, label="from ODEs' solutions")
    popt, perr = curve_fit(gl.power_law, phi0_list, N_tachy_list)
    fit_label = r"fit function $\log N = %.2f \log (\phi_0/m_{pl}) + (%.2f) $" % (popt[1], popt[0])

    plt.plot(phi0_list, gl.power_law(phi0_list, *popt), 'r-',
             color="black", label=fit_label)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\phi_0/m_{pl}$")
    plt.ylabel(r"$N$")
    plt.legend()
    plt.savefig("./figs/N_tachy.pdf", bbox_inches="tight")
    plt.close()


def plot_eff_mass(t, phi, inf_model):
    eff_mass = inf_model.get_V_pp(phi)
    plt.plot(t, eff_mass)
    plt.xlabel(r"$t * \omega_*$")
    plt.ylabel(r"$V'' / \omega_*$")
    plt.show()
