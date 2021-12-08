import numpy as np
from scipy.integrate import solve_ivp
import PPS
from globals import inf_model, reh_model
import globals as gl
# TODO: make the function exact! <\psi \psi> without approx


def get_ODE_phi_dot_func(t, phi_dot, phiStar):
    # aux function for solving ODE of phi_dot
    # use t' = t - t_*
    HStar = gl.get_Hubble_SL(phiStar)
    nStar = gl.get_nStar(phiStar)
    f = -3 * HStar * phi_dot - inf_model.get_V_p(phiStar)
    corr = reh_model.N * reh_model.lamb * nStar * np.exp(-3*HStar*t)
    if t > 0:
        f += corr
    return f


def solve_ODE_phi_dot(phiStar, m_scale):
    # solving ODE for inflaton velocity
    # m_scale is a mass scale to make numbers nicer
    phi_dot_i = gl.get_phi_dot_SL(phiStar)
    #  to put [-0.1, 1] into planck unit (unit of ODE)
    t_range = np.array([-0.1, 1])/m_scale  # m already in planck unit
    N = 300
    max_step = (t_range[1] - t_range[0])/N
    sol = solve_ivp(get_ODE_phi_dot_func, t_range, [phi_dot_i],
                    args=[phiStar], max_step=max_step)
    # output number still in planck units
    return sol.t, sol.y[0]


def get_phi_dot_ana(t, phiStar):
    # returns phi_dot WITH particle production
    unperturb_phi_dot = gl.get_phi_dot_SL(phiStar)
    if t <= 0:
        return unperturb_phi_dot
    if t > 0:
        HStar = gl.get_Hubble_SL(phiStar)
        exp_fac = np.exp(-3*HStar*t)
        nStar = PPS.get_nStar(phiStar)
        V_p = inf_model.get_V_p(phiStar)
        return unperturb_phi_dot*exp_fac - V_p/(3*HStar)*(1-exp_fac) + \
            reh_model.N * reh_model.lamb * nStar * t * exp_fac
