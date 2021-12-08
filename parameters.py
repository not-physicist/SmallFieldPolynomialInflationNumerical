import numpy as np
import globals as gl


def get_SR_epsilon(phi, inf_model):
    '''
    slow-roll epsilon parameter
    '''
    return (inf_model.get_V_p(phi)/inf_model.get_V(phi))**2/2.0


def get_SR_eta(phi, inf_model):
    '''
    slow-roll eta parameter
    '''
    return inf_model.get_V_pp(phi)/inf_model.get_V(phi)


def get_Hubble_SR_para(hubble, hubble_dot):
    '''
    hubble slow-roll parameter
    '''
    return -hubble_dot/hubble**2


def print_pr_para(inf_model):
    '''
    read CosmoLattice parameters
    in reduced Planck unit
    '''
    phi0 = inf_model.get_phi0()
    para_list = inf_model.get_ODE()
    fStar = para_list[0] * phi0
    omegaStar = 2.0 * np.sqrt(inf_model.get_d()) * fStar

    # simulation parameters in program units
    t_max = para_list[1] * omegaStar
    dt = t_max / para_list[2]

    phi_dot_i = gl.get_phi_dot_SL(fStar, inf_model) * gl.m_pl**2

    print("phiStar = %1.10e GeV, t_max = %e, dt = %e, init. mom. = %e GeV^2"
          % (fStar*gl.m_pl, t_max, dt, phi_dot_i))