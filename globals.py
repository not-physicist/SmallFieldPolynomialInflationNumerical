import numpy as np
#  import matplotlib.pyplot as plt

m_pl = 2.4e18  # GeV


######################################################################
# in reduced planck unit
def get_Hubble_SL_planck(phi, inf_model):
    '''
    return Hubble constant with slow-roll
    '''
    return np.sqrt(inf_model.get_V(phi)/3.0)


def get_phi_dot_SL_planck(phi, inf_model):
    '''
    returns inflaton velocity with SL
    '''
    H = get_Hubble_SL(phi, inf_model)
    return -inf_model.get_V_p(phi)/(3.0*H)


def get_Hubble_planck(phi, phi_dot, inf_model):
    '''
    get Hubble parameter, exact in homogeneous background
    '''
    V = inf_model.get_V(phi)
    return np.sqrt((phi_dot**2/2.0 + V)/3.0)
######################################################################


######################################################################
# in phi0, H0 planck unit
def get_phi_dot_SL(phi, inf_model):
    '''
    rescaled phi dot in slow roll
    '''
    return get_phi_dot_SL_planck(phi, inf_model) / inf_model.get_phi0()


def get_Hubble_SL(phi, inf_model):
    '''
    rescale Hubble parameter in slow-roll
    '''
    return get_Hubble_SL_planck(phi, inf_model) * inf_model.get_phi0()


def get_Hubble(phi, phi_dot, inf_model):
    '''
    rescale Hubble parameter, exact
    '''
    return get_Hubble_planck(phi, phi_dot, inf_model) * inf_model.get_phi0()
######################################################################
