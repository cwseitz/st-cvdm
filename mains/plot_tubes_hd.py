from glob import glob 
from skimage.io import imread,imsave
from cvdm.utils.errors import *
from cvdm.make import BasicKDE
from cvdm.psf.mle2d import PipelineMLE2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/N/slate/cwseitz/cvdm/Tubes/Real/Real_High_Density/eval/'
spots = pd.read_csv(path+'spots-cvdm.csv')
filtered = spots.groupby('frame').filter(lambda x: len(x) < 200)

fig,ax=plt.subplots(1,2)

theta = filtered[['x','y']].values
kde = BasicKDE(theta)
render = kde.forward(256,upsample=1)

ax[0].scatter(filtered['y'],filtered['x'],color='red',s=3,marker='x')
ax[0].set_aspect(1.0)
ax[0].invert_yaxis()
ax[1].imshow(render,cmap='gray')
plt.show()

imsave(path+'render-cvdm.tif',render)

