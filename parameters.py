import numpy as np
import globals as gl


######################################################################
# in reduced planck unit
def get_SR_epsilon_planck(phi, inf_model):
    '''
    slow-roll epsilon parameter
    '''
    return (inf_model.get_V_p(phi)/inf_model.get_V(phi))**2/2.0


def get_SR_eta_planck(phi, inf_model):
    '''
    slow-roll eta parameter
    '''
    return inf_model.get_V_pp(phi)/inf_model.get_V(phi)


def get_Hubble_SR_para_planck(hubble, hubble_dot):
    '''
    hubble slow-roll parameter
    '''
    return -hubble_dot/hubble**2
######################################################################


######################################################################
# in H0, omega0 unit
def get_SR_epsilon(phi, inf_model):
    return get_SR_eta_planck(phi, inf_model) / inf_model.get_phi0()**2


def get_SR_eta(phi, inf_model):
    return get_SR_eta_planck(phi, inf_model) / inf_model.get_phi0()**2


def get_Hubble_SR_para(hubble, hubble_dot):
    # same in Planck unit
    return get_Hubble_SR_para_planck(hubble, hubble_dot)
######################################################################


def print_pr_para(inf_model, phi, phi_dot):
    '''
    read CosmoLattice parameters
    in reduced Planck unit
    '''
    phi0 = inf_model.get_phi0()  # in m_pl
    phi_end = inf_model.get_phi_end()
    para_list = inf_model.get_ODE()
    fStar =  phi_end * phi0
    #  omegaStar = 2.0 * np.sqrt(inf_model.get_d()) * fStar
    omegaStar = inf_model.get_omegaStar()

    # simulation parameters in program units
    t_max = para_list[1]
    dt = t_max / para_list[2]

    # initial field velocity
    phi_dot_end = np.interp(phi_end, phi[::-1], phi_dot[::-1]) * omegaStar * phi0

    print("phiStar = %1.10e GeV, omegaStar = %1.10e GeV, t_max = %e, dt = %e, init. mom. = %e GeV^2, alpha = %f"
          % (fStar*gl.m_pl, omegaStar*gl.m_pl, t_max, dt, phi_dot_end*gl.m_pl**2, inf_model.get_alpha()))
