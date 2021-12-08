import numpy as np
#  import matplotlib.pyplot as plt

# all in reduced planck unit
m_pl = 2.4e18  # GeV


def get_Hubble_SL(phi, inf_model):
    '''
    return Hubble constant with slow-roll
    '''
    return np.sqrt(inf_model.get_V(phi)/3.0)


def get_phi_dot_SL(phi, inf_model):
    '''
    returns inflaton velocity with SL
    '''
    H = get_Hubble_SL(phi, inf_model)
    return -inf_model.get_V_p(phi)/(3.0*H)


def get_Hubble(phi, phi_dot, V):
    '''
    get Hubble parameter, exact in homogeneous background
    '''
    return np.sqrt((phi_dot**2/2.0 + V)/3.0)
