import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from sklearn.cluster import DBSCAN
from cvdm.psf import LoGDetector
from cvdm.psf.psf2d import MLE2D_BFGS
from numpy.linalg import inv
import matplotlib.cm as cm

class PipelineMLE2D:
    """A collection of functions for maximum likelihood localization"""
    def __init__(self,stack):
        self.stack = stack
    def localize(self,plot_spots=False,plot_fit=False,tmax=None,threshold=0.1,min_sigma=0.75,max_sigma=1.5):
        nt,nx,ny = self.stack.shape
        if tmax is not None: nt = tmax
        spotst = []
        for n in range(nt):
            framed = self.stack[n]
            log = LoGDetector(framed,threshold=threshold,min_sigma=min_sigma,max_sigma=max_sigma)
            spots = log.detect() #image coordinates
            print(f'Det in frame {n}: {len(spots)}')
            if plot_spots:
                log.show(); plt.show()
            #spots = self.fit(framed,spots,plot_fit=plot_fit)
            spots = spots.assign(frame=n)
            spotst.append(spots)
        spotst = pd.concat(spotst)
        return spotst

    def fit(self,frame,spots,patchw=3,max_iters=100,plot_fit=False):
        for i in spots.index:
            start = time.time()
            x0 = int(spots.at[i,'x']) #image coordinates (row)
            y0 = int(spots.at[i,'y']) #image coordinates (column)
            adu = frame[x0-patchw:x0+patchw+1,y0-patchw:y0+patchw+1]
            adu = np.clip(adu,0,None)
            theta0 = np.array([patchw,patchw,1.0])
            opt = MLE2D_BFGS(theta0,adu) #cartesian coordinates with top-left origin
            theta_mle, loglike, conv, err = opt.optimize(max_iters=max_iters,
                                                         plot_fit=plot_fit)
            #theta_mle is in subpixel image coordinates (row,column) within the patch
            dx = theta_mle[0] - patchw; dy = theta_mle[1] - patchw
            spots.at[i, 'x_mle'] = x0 + dx
            spots.at[i, 'y_mle'] = y0 + dy
            spots.at[i, 'N0'] = theta_mle[2]
            spots.at[i, 'conv'] = conv
            end = time.time()
            elapsed = end-start
            #print(f'Fit spot {i} in {elapsed} sec')

        return spots

    def dbscan(self,data,eps=3,min_samples=10,plot=False):
        coords = data[['x_mle','y_mle']].values
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        data['cluster'] = db.labels_
        if plot:
            fig,ax=plt.subplots()
            ax.imshow(np.sum(self.stack,axis=0),cmap='gray')
            unique_clusters = data['cluster'].unique()
            colors = cm.get_cmap('tab20b', len(unique_clusters))
            for i, cluster in enumerate(unique_clusters):
                cluster_data = data[data['cluster'] == cluster]
                color = colors(i) if cluster != -1 else 'lightgray'
                ax.scatter(cluster_data['y_mle'],cluster_data['x_mle'],c=[color], s=20, alpha=0.7, 
                            label=f'Cluster {cluster}' if cluster != -1 else 'Noise')
                ax.invert_yaxis()
            plt.show()
        data = data.loc[data['cluster'] != -1]
        return data
