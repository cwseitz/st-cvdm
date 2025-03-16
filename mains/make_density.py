import matplotlib.pyplot as plt
import spatialdata as sd
import numpy as np
from skimage.io import imsave

xenium_path = ""
sdata = sd.read_zarr(xenium_path)
pdata = sdata.points['transcripts'].compute()
print(pdata['feature_name'].nunique())
nbins=512

xmin,xmax = 0,10000
ymin,ymax = 1000,3000
bin_size = 20
xbins = np.arange(xmin,xmax,bin_size)
ybins = np.arange(ymin,ymax,bin_size)

stack = []
for n,gene in enumerate(pdata['feature_name'].unique()):
    print('Binning data for gene: ' + gene)
    select = pdata.loc[pdata['feature_name'] == gene]
    H, xedges, yedges = np.histogram2d(select['x'], select['y'], bins=(xbins,ybins))
    stack.append(H.T)
    if n == 0:
        fig,ax=plt.subplots()
        ax.scatter(select['x'],select['y'],marker='x',s=1,color='black')
        fig,ax=plt.subplots()
        ax.imshow(H.T,cmap='plasma',origin='lower')
        plt.show()
               
stack = np.array(stack)
#imsave('/N/slate/cwseitz/spatial/Stack.tif',stack)
