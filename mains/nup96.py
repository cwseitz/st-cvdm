from glob import glob 
from skimage.io import imread,imsave
from cvdm.utils.errors import *
from cvdm.psf.mle2d import PipelineMLE2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/N/slate/cwseitz/cvdm/Nup96/4x/Archive/_sum5/'
zfiles = glob(path+'z-*-0.tif')
zfiles = sort_and_group(zfiles)
Z = np.array([imread(path+zfiles[idx][0]) for idx in range(0,30)])

pipe = PipelineMLE2D(Z)
spots = pipe.localize(plot_spots=False,plot_fit=False)
spots.to_csv(path+'spots-cvdm.csv')
