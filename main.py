import numpy as np
import matplotlib.pyplot as plt
#  from numba import jit

import globals as gl
import models
import osci
import tachy
import parameters as para
import ODE
import plot

# TODO: change units; rescale equations
# TODO: write a general class for potential
# TODO: find period for oscillations
# TODO: allow git sync in ownCloud

# change hardcoded limit of # of point
# https://stackoverflow.com/questions/37470734/matplotlib-giving-error-overflowerror-in-draw-path-exceeded-cell-block-limit
#  import matplotlib as mpl
#  mpl.rcParams['agg.path.chunksize'] = 10000

'''
UNITs for physical quantities:
    - field: phi0
    - time: H0
    - momentum: m2_eff, min

except for the section for parameters of CosmoLattice
'''


def comp_ODE_para(phi0):
    t_max = 10
    N_t = 100 / phi0 + 1000
    return t_max, N_t


if __name__ == "__main__":
    '''
    phi0_list = np.array([1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3,
                          5e-4, 1e-4, 3e-5])
    N_tachy_list = np.zeros(phi0_list.shape[0])
    # iterate over phi0's
    for i in range(0, phi0_list.shape[0]):
        phi0 = phi0_list[i]
        #  ODE_para = ODE_para_list[i]
        ODE_para = np.float128(get_ODE_para(phi0))
        # !!! ODE_para[0] is phi_i in units of phi_0

        # Intializing model
        SFPI = models.SFPInf(phi0, ODE_para)
        draw_para(phi0, t, phi, phi_dot)
        SFPI.print()
        print_pr_para(SFPI, ODE_para, phi0)

        fn = "./data/phi-phi0=" + str(phi0) + ".dat"
        solve_ODE(phi0, fn)
        t, phi, phi_dot = read_sol(fn)

        plot_potential(phi0)
        draw_para(phi0, t, phi, phi_dot)

        N_tachy_list[i] = draw_phi_find_N_tachy(phi0, t, phi)

        print("")

    np.savetxt("./data/N_tachy.dat", np.array([phi0_list, N_tachy_list]).T)

    plot_N_tachy()
    '''
    # 1e0, 10, 1e3
    # 1e-1, 10, 1e4
    # 1e-2, 10, 1e5
    # 1e-3, 5, 1e5
    # 1e-4,
    phi0 = 1.0e-4

    SFPI = models.SFPInf(phi0)
    #  SFPI.set_ODE(*comp_ODE_para(phi0))
    SFPI.set_ODE(5, 1e7)
    SFPI.print()
    para.print_pr_para(SFPI)

    fn = "./data/phi-phi0=" + str(phi0) + ".dat"
    ODE.solve_EOM_save2file(SFPI, fn)
    t, phi, phi_dot = ODE.read_sol(fn)
    plot.draw_phi_find_N_tachy(SFPI, t, phi)
    plot.draw_para(SFPI, t, phi, phi_dot, xlims=(0, 3))

    osci.find_period(t, phi)
