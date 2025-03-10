import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.patches as patches

path = '/N/slate/cwseitz/cvdm/Sim/4x/N-100_N0-500-1000/eval_data/N100-1/'
lr100 = imread(path+'lr-1x.tif')[0]
hr100_true = imread(path+'y-0-0.tif')
hr100 = imread(path+'z-0-0.tif')
path = '/N/slate/cwseitz/cvdm/Sim/4x/N-200_N0-500-1000/eval_data/N200-1/'
lr200 = imread(path+'lr-1x.tif')[0]
hr200_true = imread(path+'y-0-0.tif')
hr200 = imread(path+'z-0-0.tif')
path = '/N/slate/cwseitz/cvdm/Sim/4x/N-500_N0-500-1000/eval_data/N500-1/'
lr500 = imread(path+'lr-1x.tif')[0]
hr500_true = imread(path+'y-0-0.tif')
hr500 = imread(path+'z-0-0.tif')

hr100[hr100 < 0] = 0
hr200[hr200 < 0] = 0
hr500[hr500 < 0] = 0

densities = [100, 200, 500]
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
plt.subplots_adjust(wspace=0.02, hspace=0.1)

hr_inset_coords_list = [(15, 40), (50, 50), (100, 100)]

lr_inset_size = 15
hr_inset_size = 60

for row, (lr, hr_true, hr, hr_inset_coords) in enumerate(zip(
        [lr100, lr200, lr500], 
        [hr100_true, hr200_true, hr500_true], 
        [hr100, hr200, hr500], 
        hr_inset_coords_list)):

    lr_inset_coords = (hr_inset_coords[0] // 4, hr_inset_coords[1] // 4)

    axes[row, 0].imshow(lr, cmap='gray')
    axes[row, 0].set_ylabel(f'$\\rho$={densities[row]}', fontsize=16)
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])


    inset = axes[row, 0].inset_axes([0.65, 0.65, 0.4, 0.4])
    inset.imshow(lr[lr_inset_coords[0]:lr_inset_coords[0]+lr_inset_size, lr_inset_coords[1]:lr_inset_coords[1]+lr_inset_size], cmap='gray', interpolation='nearest')
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color('red')
        spine.set_linewidth(1)

    axes[row, 1].imshow(hr_true, cmap='gray')
    axes[row, 1].set_xticks([])
    axes[row, 1].set_yticks([])

    inset = axes[row, 1].inset_axes([0.65, 0.65, 0.4, 0.4])
    inset.imshow(hr_true[hr_inset_coords[0]:hr_inset_coords[0]+hr_inset_size, hr_inset_coords[1]:hr_inset_coords[1]+hr_inset_size], cmap='gray', interpolation='nearest')
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color('red')
        spine.set_linewidth(1)

    axes[row, 2].imshow(hr, cmap='gray')
    axes[row, 2].set_xticks([])
    axes[row, 2].set_yticks([])

    inset = axes[row, 2].inset_axes([0.65, 0.65, 0.4, 0.4])
    inset.imshow(hr[hr_inset_coords[0]:hr_inset_coords[0]+hr_inset_size, hr_inset_coords[1]:hr_inset_coords[1]+hr_inset_size], cmap='gray', interpolation='nearest')
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color('red')
        spine.set_linewidth(1)

axes[0, 0].set_title(r'$x$', fontsize=16)
axes[0, 1].set_title(r'$y_0$', fontsize=16)
axes[0, 2].set_title(r'$\hat{y}_0$', fontsize=16)

plt.savefig('/N/slate/cwseitz/cvdm/Sim/4x/Figure-3.png',dpi=300)
plt.show()

