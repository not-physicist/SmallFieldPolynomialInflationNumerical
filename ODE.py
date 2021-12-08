#################################################################
# solving EOM of inflaton
#################################################################
import globals as gl
from scipy.integrate import solve_ivp
import time
import numpy as np

# all function should be fine with any unit system
# as long as get_V() etc are in the corresponding units


def get_EOM_phi_aux(t, y, inf_model):
    '''
    first order (coupled) differential equations to solve
    y = (phi, phi_dot)^T
    '''
    phi = y[0]
    phi_dot = y[1]

    V = inf_model.get_V(phi)
    V_p = inf_model.get_V_p(phi)
    H = gl.get_Hubble(phi, phi_dot, V)

    dydt = [phi_dot, -3.0*H*phi_dot - V_p]
    return dydt


def solve_EOM_phi(inf_model):
    '''
    returns sol's of EOM
    phi_i: initial field value; t_max: final cosmic time
    '''
    phi_i = inf_model.get_phi_i()
    phi_dot_i = gl.get_phi_dot_SL(phi_i, inf_model)
    #  print("phi_dot_i = %e" % (phi_dot_i))
    t_range = [0, inf_model.get_t_max()]
    #  print(phi_i, phi_dot_i, t_range)
    sol = solve_ivp(lambda t, y: get_EOM_phi_aux(t, y, inf_model), t_range,
                    [phi_i, phi_dot_i],
                    max_step=inf_model.get_t_step()
                    )
    return sol.t, sol.y


def solve_EOM_save2file(inf_model, fn):
    """
    Solving ODE and storing solution into files
    """
    print("Solving ODEs...", end='')
    t0 = time.time()

    phi0 = inf_model.get_phi0()
    t, y = solve_EOM_phi(inf_model)
    phi = y[0]
    phi_dot = y[1]
    #  print(t, phi)
    #  phi_ddot = np.diff(phi_dot)/np.diff(t)

    np.savetxt(fn, np.array([t, phi, phi_dot], dtype=np.float128).T,
               header="phi0=" + str(phi0) + "\n"
               + "t\tphi\tphi_dot")
    t1 = time.time()
    t = t1 - t0
    print("it took %f seconds" % (t))


def read_sol(fn):
    """
    read EOM solutions from file
    """
    data = np.genfromtxt(fn, skip_header=2).T
    t = data[0]
    phi = data[1]
    phi_dot = data[2]
    return t, phi, phi_dot
