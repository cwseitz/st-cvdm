import numpy as np
import matplotlib.pyplot as plt
from ..psf.psf2d.psf2d import *
from scipy.stats import poisson

class Density:
    def __init__(self):
        pass
        
class Disc(Density):
    """Uniform distribution on a disc"""
    def __init__(self,radius):
        super().__init__()
        self.radius=radius
    def sample(self,n):
        theta = np.random.uniform(0,2*np.pi,n)
        radius = self.radius*np.sqrt(np.random.uniform(0, 1, n))
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return x, y

class Uniform(Density):
    """Uniform distribution"""
    def __init__(self,size,edgew=5.0):
        super().__init__()
        self.size=size
        self.edgew = edgew
    def sample(self,n):
        x = np.random.uniform(self.edgew,self.size-self.edgew,n)
        y = np.random.uniform(self.edgew,self.size-self.edgew,n)
        return x, y
       
class Generator:
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny
    def show(self,theta,adu,noise,muS,S):
        fig,ax=plt.subplots(1,3,figsize=(10,4))
        ax[0].scatter(theta[1,:],theta[0,:],color='black',s=5,marker='o')
        ax[1].imshow(muS,cmap='gray')
        ax[2].imshow(adu,cmap='gray')
        ax[0].set_xticks([]); ax[0].set_yticks([])
        ax[1].set_xticks([]); ax[1].set_yticks([])
        ax[2].set_xticks([]); ax[2].set_yticks([])
        ax[0].set_xlim([0,adu.shape[0]])
        ax[0].set_ylim([0,adu.shape[1]])
        ax[0].set_aspect(1.0)
        ax[0].invert_yaxis()
        plt.show()

        
    def sample_frames(self,theta,nframes,texp,eta,B0,gain,offset,var,show=False):
        _adu = []; _spikes = []
        for n in range(nframes):
            muS = self._mu_s(theta,texp=texp,eta=eta)
            S = self.shot_noise(muS)
            if B0 is not None:
                muB = self._mu_b(B0)
                B = self.shot_noise(muB)
            else:
                B = 0
            read_noise = self.read_noise(offset=offset,var=var)
            adu = gain*(S+B) + read_noise
            adu = np.clip(adu,0,None)
            adu = np.squeeze(adu)
            spikes = self.spikes(theta)
            _adu.append(adu); _spikes.append(spikes)
            if show:
                self.show(theta,adu,read_noise,muS,S)
            
        adu = np.squeeze(np.array(_adu))
        spikes = np.squeeze(np.array(_spikes))
        return adu,spikes
    def _mu_s(self,theta,texp=1.0,eta=1.0,patch_hw=3):
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y,indexing='ij')
        mu = np.zeros((self.nx,self.ny),dtype=np.float32)
        ntheta,nspots = theta.shape
        for n in range(nspots):
            x0,y0,sigma,N0 = theta[:,n]
            i0 = eta*N0*texp
            patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
            x0p = x0-patchx; y0p = y0-patchy
            this_mu = i0*lamx(X,x0p,sigma)*lamy(Y,y0p,sigma)
            mu[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += this_mu
        return mu

    def _mu_b(self,B0):
        rate = B0*np.ones((self.nx,self.ny))
        return rate
       
    def shot_noise(self,rate):
        """Universal for all types of detectors"""
        electrons = np.random.poisson(lam=rate)
        return electrons
                
    def read_noise(self,offset=100.0,var=5.0):
        """Gaussian readout noise"""
        noise = np.random.normal(offset,np.sqrt(var),size=(self.nx,self.ny))
        return noise
        
    def spikes(self,theta,upsample=4):
        new_nx = self.nx * upsample
        new_ny = self.ny * upsample
        theta = theta[:2, :, np.newaxis, np.newaxis]
        x_vals = np.linspace(0, new_nx, new_nx, endpoint=False)
        y_vals = np.linspace(0, new_ny, new_ny, endpoint=False)

        x_indices = np.floor(theta[0] * upsample).astype(int)
        y_indices = np.floor(theta[1] * upsample).astype(int)

        x_indices = np.clip(x_indices, 0, new_nx - 1)
        y_indices = np.clip(y_indices, 0, new_ny - 1)

        spikes = np.zeros((new_nx, new_ny), dtype=int)
        np.add.at(spikes, (x_indices, y_indices), 1)

        return spikes


        

    
        

