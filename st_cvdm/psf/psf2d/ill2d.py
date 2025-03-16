import numpy as np
import warnings
from .psf2d import *

def isologlike2d(theta,adu,cam_params):
    nx,ny = adu.shape
    x0,y0,N0 = theta; sigma=1.0
    eta,texp,gain,offset,var = cam_params
    X,Y = np.meshgrid(np.arange(0,nx),np.arange(0,ny),indexing='ij')
    lam = lamx(X,x0,sigma)*lamy(Y,y0,sigma)
    i0 = gain*eta*texp*N0
    muprm = i0*lam + var
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    stirling = adu * np.nan_to_num(np.log(adu)) - adu
    p = adu*np.log(muprm)
    warnings.filterwarnings("default", category=RuntimeWarning)
    p = np.nan_to_num(p)
    nll = stirling + muprm - p
    nll = np.sum(nll)
    return nll
