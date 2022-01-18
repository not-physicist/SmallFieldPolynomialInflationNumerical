import numpy as np
#  import matplotlib.pyplot as plt
#  from numba import jit
import sys
import time

import globals as gl
import models
import floquet as flo
import tachy
import parameters as para
import ODE
import plot

# TODO: write a general class for potential

# change hardcoded limit of # of point
# https://stackoverflow.com/questions/37470734/matplotlib-giving-error-overflowerror-in-draw-path-exceeded-cell-block-limit
#  import matplotlib as mpl
#  mpl.rcParams['agg.path.chunksize'] = 10000

'''
UNITs for physical quantities:
    - field: phi0
    - time: omega0
    - momentum: m2_eff, min

except for the section for parameters of CosmoLattice
'''


def compute_ODE_para(phi0):
    '''
    compute parameters for ODE based on past experience
    '''
    t_max = 50 * phi0**(-2/3)
    N_t = 100 / phi0 + 10000
    return t_max, N_t


def do_ODE_everything(flag=False):
    #  phi0_array = np.array([1e-4])
    phi0_array = np.array([1e0, 5e-1, 1e-1, 5e-2, 1e-2,
                           5e-3, 1e-3, 5e-4, 1e-4, 3e-5])
    xlims_array = np.zeros((phi0_array.shape[0], 2))
    xlims_array[-1] = [0, 0.002]

    N_tachy_array = np.zeros(phi0_array.shape[0])
    peak_height_array = np.zeros(phi0_array.shape[0])
    for i in range(0, phi0_array.shape[0]):
        print("-----------------------------------------------------")
        phi0 = phi0_array[i]
        xlim = xlims_array[i]

        SFPI = models.SFPInf(phi0)
        #  SFPI.set_ODE(500/omega0, 1e5)
        SFPI.set_ODE(*compute_ODE_para(phi0))
        SFPI.print()

        fn = "./data/phi-phi0=" + str(phi0) + ".dat"
        if flag:
            ODE.solve_EOM_save2file(SFPI, fn)
        t, phi, phi_dot = ODE.read_sol(fn)

        para.print_pr_para(SFPI, phi, phi_dot)

        #  if xlim[0] == xlim[1]:  # in case both are zero
            #  plot.draw_phi_tachy_points(SFPI, t, phi)
        #  else:
            #  plot.draw_phi_tachy_points(SFPI, t, phi)
        #  plot.draw_para(SFPI, t, phi, phi_dot, xlims=(0, 3))
        N_tachy_array[i] = plot.find_N_tachy(t, phi)

        flo.find_period(t, phi)

        peak_height_array[i] = flo.find_1st_peak(t, phi)
        print("height of first peak", peak_height_array[i])

        print("-----------------------------------------------------")
        print("")

    np.savetxt("./data/N_tachy.dat", np.array([phi0_array, N_tachy_array]).T)
    plot.plot_N_tachy()
    np.savetxt("./data/peak_height.dat", np.array([phi0_array, peak_height_array]).T)
    flo.plot_1st_peaks()


if __name__ == "__main__":

    #  do_ODE_everything(flag=False)

    '''
    phi0 = 1e-3
    SFPI = models.SFPInf(phi0)

    fn = "./data/phi-phi0=" + str(phi0) + ".dat"
    t, phi, _ = ODE.read_sol(fn)
    plot.plot_eff_mass(t, phi, SFPI)
    '''

    try:
        n_proc = int(sys.argv[1])
        n_step = float(sys.argv[2])
        print(f"Using {n_proc} CPU cores with {n_step} steps")
        flo.save_flo(100, 128, n_proc, n_step)
    except Exception:
        print("Error!")

    #  flo.plot_flo("H0")
    #  flo.plot_flo()

    '''
    start_t = time.perf_counter()
    phi0 = 2.1049e-4
    SFPI = models.SFPInf(phi0)
    flo.compute_flo(SFPI, 0, 2, 20, 5e4)
    end_t = time.perf_counter()
    print(f"\nExecution time: {end_t - start_t:0.3f} s")
    '''
