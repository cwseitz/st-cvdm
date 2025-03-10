import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np

path='/N/slate/cwseitz/cvdm/Nup96/4x/Archive/sum1/'
lr1x_sum1 = imread(path+'lr-1x.tif')[:-1]

nframes,nx,ny = lr1x_sum1.shape
nframes = 5
lr1x_sum5 = lr1x_sum1.reshape((-1,nframes,nx,ny))
lr1x_sum5 = np.sum(lr1x_sum5,axis=1)

fig,ax=plt.subplots(1,2,sharex=True,sharey=True)
ax[0].imshow(lr1x_sum1[0,:190,:190],cmap='gray')
ax[1].imshow(lr1x_sum5[0,:190,:190],cmap='gray')
for axi in ax.ravel():
    axi.set_xticks([])
    axi.set_yticks([])
plt.tight_layout()
plt.savefig('/N/slate/cwseitz/cvdm/Sim/4x/Figure-9.png', dpi=300)
plt.show()
