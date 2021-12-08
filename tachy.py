import numpy as np


def delta_phi(get_phi_dot, model):
    # return sqrt of variance of quantum fluctuations
    # using num. results
    V0 = model.get_V(model.phi0)
    H = np.sqrt(V0/3.0)  # / M_pl

    print("H=", H, "m_pl")
    print("phi_dot(phi0/3)=", get_phi_dot(model.phi0/3.0), "m_pl^2")
    print("phi_dot(phi0)=", get_phi_dot(model.phi0), "m_pl^2")
    delta_p = H/(3.0*np.pi) * get_phi_dot(model.phi0/3.0) / get_phi_dot(model.phi0)
    return delta_p
