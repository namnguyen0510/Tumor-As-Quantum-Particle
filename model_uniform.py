import torch
import torch.nn as nn
import numpy as np
from torchquad import MonteCarlo, set_up_backend

# UNIFORM DIST


def surv_prob(t, gamma, eta, return_log = True, debug = False):
    set_up_backend("torch", data_type="float32")
    # TODO: MORE INTEGRATOR
    mc = MonteCarlo()
    if debug:
        print('t:     {}'.format(t))
        print('gamma: {}'.format(gamma))
        print('eta :  {}'.format(eta))
    def A(t):
        x = 1/(gamma + (4-t**2)**0.5)**2
        return x.requires_grad_()
    def B(t):
        x = (1/(gamma**2 + t**2 - 4))*((1/2*(t-(t**2-4)**0.5))**(2*eta))
        return x.requires_grad_()
    def _integrate(f, a, b):
        x = mc.integrate(f, dim=1, N= int(1e5),integration_domain=[[a, b]],backend="torch").requires_grad_()
        return x
    A_int = _integrate(A,0,2)
    B_int = _integrate(B,2,t)
    S = 1 - 2/np.pi*gamma*(A_int + B_int)
    if return_log:
        logS = torch.log(S)
        return S, logS
    else:
        return S














































#
