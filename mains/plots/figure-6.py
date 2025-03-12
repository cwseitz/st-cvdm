import numpy as np
from skimage.io import imread
from skimage.restoration import rolling_ball
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

path = '/N/slate/cwseitz/st_cvdm/Nup96/4x/Archive/_sum5/'
lr100 = imread(path + 'lr-1x-sum.tif')

x0, y0 = 57, 57
lr100_1 = lr100[0][x0:x0 + 64, y0:y0 + 64]
lr100_2 = lr100[2][x0:x0 + 64, y0:y0 + 64]
lr100_3 = lr100[4][x0:x0 + 64, y0:y0 + 64]

hr100_1 = imread(glob(path + 'z-0-0.tif')[0])
hr100_2 = imread(glob(path + 'z-1-0.tif')[0])
hr100_3 = imread(glob(path + 'z-2-0.tif')[0])

hr100_1[hr100_1 < 0.0] = 0
hr100_2[hr100_2 < 0.0] = 0
hr100_3[hr100_3 < 0.0] = 0

hr100_1 = hr100_1 - rolling_ball(hr100_1, radius=20)
hr100_2 = hr100_2 - rolling_ball(hr100_2, radius=20)
hr100_3 = hr100_3 - rolling_ball(hr100_3, radius=20)

lr_inset_coords = [(12,20),(40,40),(14,22)]
hr_inset_coords = [(4*c[0],4*c[1]) for c in lr_inset_coords]

lr_inset_size = 15
hr_inset_size = 60

fig, ax = plt.subplots(3, 2, figsize=(4, 6))

for i, (lr, hr, lr_coords, hr_coords) in enumerate(zip([lr100_1, lr100_2, lr100_3], 
                                                       [hr100_1, hr100_2, hr100_3], 
                                                       lr_inset_coords, 
                                                       hr_inset_coords)):
    ax[i, 0].imshow(lr, cmap='gray')
    ax[i, 0].set_xticks([]); ax[i, 0].set_yticks([])
    
    inset_lr = ax[i, 0].inset_axes([0.65, 0.65, 0.4, 0.4])
    inset_lr.imshow(lr[lr_coords[0]:lr_coords[0] + lr_inset_size, lr_coords[1]:lr_coords[1] + lr_inset_size], cmap='gray')
    inset_lr.set_xticks([]); inset_lr.set_yticks([])
    for spine in inset_lr.spines.values():
        spine.set_color('red')
        spine.set_linewidth(1)

    ax[i, 1].imshow(hr, cmap='gray')
    ax[i, 1].set_xticks([]); ax[i, 1].set_yticks([])
    
    inset_hr = ax[i, 1].inset_axes([0.65, 0.65, 0.4, 0.4])
    inset_hr.imshow(hr[hr_coords[0]:hr_coords[0] + hr_inset_size, hr_coords[1]:hr_coords[1] + hr_inset_size], cmap='gray')
    inset_hr.set_xticks([]); inset_hr.set_yticks([])
    for spine in inset_hr.spines.values():
        spine.set_color('red')
        spine.set_linewidth(1)

ax[0, 0].set_title(r'$\hat{x}$', fontsize=14)
ax[0, 1].set_title(r'$\hat{y}_0$', fontsize=14)

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('/N/slate/cwseitz/st_cvdm/Sim/4x/Figure-6.png', dpi=300)
plt.show()

