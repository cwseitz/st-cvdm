from glob import glob 
from skimage.io import imread,imsave
from cvdm.utils.errors import *
from cvdm.psf.mle2d import PipelineMLE2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = '/N/slate/cwseitz/cvdm/Sim/4x/N-500_N0-500-1000/eval_data/'

prefixes = [
'N500-1',
'N500-2',
'N500-3',
'N500-4',
'N500-5',
'N500-6',
'N500-7',
'N500-8',
'N500-9',
'N500-10'
]

batch_size=2
spotdfs = []; errordfs = []; all_set_metrics = []
for prefix in prefixes:
    print(f'Prefix: {prefix}')
    xfiles = glob(path+prefix+'/x*.tif')
    yfiles = glob(path+prefix+'/y*.tif')
    zfiles = glob(path+prefix+'/z*.tif')
    xfiles = sort_and_group(xfiles)
    yfiles = sort_and_group(yfiles)
    zfiles = sort_and_group(zfiles)
    idxs = xfiles.keys()
    stack1x = imread(path+prefix+'/lr-1x.tif')
    all_set_metrics_ = []
    for idx in idxs:
        print(f'Index: {idx}')
        this_xfiles = xfiles[idx]
        this_yfiles = yfiles[idx]
        this_zfiles = zfiles[idx]
        cpath = path+prefix+f'/coords/coords-{batch_size*idx}.npz'
        coordsgt = np.load(cpath)['theta']
        X,Y,Z = prepare_data(path+prefix,this_xfiles,this_yfiles,this_zfiles)
        X1x = stack1x[batch_size*idx]
        pipe = PipelineMLE2D(Z)
        spots = pipe.localize(plot_spots=False,plot_fit=False)
        spots = pipe.dbscan(spots)
        frame_counts = spots['frame'].value_counts()
        max_frame = frame_counts.idxmax()
        spots = spots[spots['frame'] == max_frame]
        errordf,set_metrics = get_errors(spots,coordsgt)
        errordf = errordf.assign(prefix=prefix)
        spots = spots.assign(prefix=prefix)
        errordf = errordf.assign(idx=idx)
        spots = spots.assign(idx=idx)
        errordfs.append(errordf); spotdfs.append(spots)
        all_set_metrics_.append(set_metrics[0])
    all_set_metrics.append(all_set_metrics_)

print(all_set_metrics)
all_set_metrics = np.array(all_set_metrics)
errordfs = pd.concat(errordfs)
spotdfs = pd.concat(spotdfs)
errordfs.to_csv(path+'N500-error.csv')
spotdfs.to_csv(path+'N500-spots.csv')
np.savez(path+'N500-set.npz',metrics=all_set_metrics)


    

