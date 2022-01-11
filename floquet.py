import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
#  from scipy.optimize import newton
#  from scipy.integrate import quad
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import os

#  import ODE
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


def find_1st_peak(t, phi):
    """
    return height of 1st peak in phi
    """
    peaks, _ = find_peaks(phi, height=[0, 1])
    i = peaks[0]
    return phi[i]


def plot_1st_peaks():
    data = np.genfromtxt("./data/peak_height.dat").T
    phi0_array = data[0]
    #  log_phi0_array = np.log10(data[0])
    #  peak_height_array = data[1]
    loss = 1 - data[1]

    plt.scatter(phi0_array, loss, color="black")

    popt, perr = curve_fit(gl.power_law, phi0_array, loss)
    plt.plot(phi0_array, gl.power_law(phi0_array, *popt),
             '--', label="fit", color="black")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\phi_0/m_{pl}$")
    plt.ylabel(r"$1-\phi_{max}/\phi_0$")
    plt.legend()

    plt.savefig("./figs/1st_peaks.pdf", bbox_inches="tight")
    np.savetxt("./data/1st_peaks_fit.dat", popt)


def solve_fund_matrix(k, inf_model, t0, t1, get_phi):
    def solve_fund_matrix_aux(t, y, k):
        fld = y[0]
        pi = y[1]

        dydt = [pi, (-k**2 - inf_model.get_V_pp(get_phi(t)))*fld]
        #  dydt = [pi, (-k**2)*fld]
        #  print(t)
        return dydt

    T = t1 - t0  # in omega* unit, should be roughly 1
    sol1 = solve_ivp(lambda t, y: solve_fund_matrix_aux(t, y, k),
                     [t0, t1], [1, 0], max_step=T/50000, method="DOP853")
    sol2 = solve_ivp(lambda t, y: solve_fund_matrix_aux(t, y, k),
                     [t0, t1], [0, 1], max_step=T/50000, method="DOP853")
    #  print(sol1, sol2)
    fund_matrix = np.array([[sol1.y[0, -1], sol2.y[0, -1]],
                            [sol1.y[1, -1], sol2.y[1, -1]]])
    #  print(O)
    eig = np.linalg.eigvals(fund_matrix)
    #  print(f"Eigenvalues at {k} are {eig}")
    det = np.linalg.det(fund_matrix)
    tolerance = 1e-3
    if np.abs(1 - det) > tolerance:
        print(f"Exceeding tolerance! Dets at {k} are {det}.")
    else:
        print(f"Dets at {k} are {det}")


    flo_coef = np.log(np.abs(eig))/T
    return flo_coef, det


def compute_flo(inf_model, k_num):
    # get phi(t)
    t_range = [0, 20]
    popt = np.genfromtxt("./data/1st_peaks_fit.dat")
    phi_i = 1 - gl.power_law(inf_model.get_phi0(), *popt)
    sol = solve_ivp(lambda t, y: [y[1], -inf_model.get_V_p(y[0])],
                    t_range, [phi_i, 0], max_step=t_range[1]/50000,
                    method="DOP853")
    t = sol.t
    phi = sol.y[0]

    def get_phi(x):
        return np.interp(x, t, phi)

    #  plt.plot(t, get_phi(t), color="black")
    #  plt.plot(t, phi, linestyle="--", color="red")
    #  plt.show()

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
    #  print(m_min)
    k = np.linspace(0, 2, num=k_num)
    R_mu = np.zeros(k.shape[0])
    det = np.zeros(k.shape[0])
    for i, k_i in enumerate(k):
        coeffs, dets = solve_fund_matrix(k_i*m_min, inf_model, t0, t1, get_phi)
        #  print(f"Floquet coefficients at {k_i} are {coeffs}")
        #  print(f"Difference in Floquet coeff is {np.abs(coeffs[0] + coeffs[1])}")

        R_mu[i] = np.amax(coeffs)
        det[i] = dets

        #  plt.scatter(k_i, R_mu[i], color="black")
    #  plt.ylabel("$\Re{\mu}/\omega_*$")
    #  plt.xlabel("$k/m$")
    #  plt.show()
    #  print(k, R_mu)
    return k, R_mu, det


def save_flo(phi0_num, k_num):
    fn = "./data/floquent.dat"
    if os.path.exists(fn):
        os.remove(fn)

    fn_det = "./data/floquent_dets.dat"
    if os.path.exists(fn_det):
        os.remove(fn_det)

    with open(fn, 'ab') as f, open(fn_det, 'ab') as f_dets:
        phi0_array = np.logspace(0, -4, phi0_num, base=10)
        for i, phi0 in enumerate(phi0_array):
            print(f"phi0: {i+1} / {phi0_array.shape[0]}")
            SFPI = models.SFPInf(phi0)
            k, flo, det = compute_flo(SFPI, k_num)
            np.savetxt(f, [flo])
            np.savetxt(f_dets, [det])

    # k should be the same for all
    fn_k = "./data/floquent_k.dat"
    np.savetxt(fn_k, k)

    fn_phi0 = "./data/floquent_phi0.dat"
    np.savetxt(fn_phi0, phi0_array)


def plot_flo(unit="omega"):
    phi0_array = np.genfromtxt("./data/floquent_phi0.dat")
    log_phi0_array = np.log10(phi0_array)
    k_array = np.genfromtxt("./data/floquent_k.dat")
    flo_big_array = np.genfromtxt("./data/floquent.dat")

    if flo_big_array.shape[0] != phi0_array.shape[0]:
        print("Something is wrong!")
        print(flo_big_array.shape, phi0_array.shape)
        return ValueError

    # unit for mu
    if unit == "omega":
        heatmap = flo_big_array
        fn = "./figs/floquent_heatmap_omega.pdf"  # plot file to be saved
        cb_label = r"$\Re(\mu)/\omega_*$"
    elif unit == "H0":
        fn = "./figs/floquent_heatmap_H0.pdf"  # plot file to be saved
        heatmap = np.zeros(flo_big_array.shape)
        cb_label = r"$\Re(\mu)/H_0$"
        for i, n in enumerate(flo_big_array):
            heatmap[i] = n / phi0_array[i]
    else:
        return ValueError
    # left, right, bottom, top
    extent = [k_array[0], k_array[-1],
              log_phi0_array[-1], log_phi0_array[0]]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plt.clf()
    im = plt.imshow(heatmap, extent=extent)
    #  plt.yscale("log")

    #  ax.set_xlabel(r"k/m")
    #  ax.set_ylabel(r"$\phi_0/m_{pl}$")
    plt.xlabel(r"$k/m(\phi_0)$")
    plt.ylabel(r"$\log_{10}(\phi_0/m_{pl})$")

    # colorbar
    cax = plt.axes([0.65, 0.1, 0.04, 0.8])  # [left, bottom, width, height]
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(cb_label)

    plt.savefig(fn, bbox_inches="tight")
