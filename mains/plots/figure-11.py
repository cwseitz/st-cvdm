import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

path_hd = '/N/slate/cwseitz/cvdm/Tubes/Real/Real_High_Density/'
path_ls = '/N/slate/cwseitz/cvdm/Tubes/Real/Real_Long_Sequence/'

summed_hd = imread(path_hd + 'SUM_lr-1x.tif')
summed_ls = imread(path_ls + 'SUM_lr-1x.tif')

ls_cvdm = imread(path_ls + 'eval/render-cvdm.tif')
ls_thunder = imread(path_ls + 'thunderstorm/render.tif')

hd_cvdm = imread(path_hd + 'eval/render-cvdm.tif')
hd_thunder = imread(path_hd + 'thunderstorm/render-crop.tif')

ls_cvdm = np.roll(ls_cvdm,1,axis=1)
ls_combined_image = np.zeros((*ls_cvdm.shape, 3), dtype=np.float32)
ls_combined_image[..., 0] = ls_cvdm / ls_cvdm.max()
ls_combined_image[..., 2] = ls_thunder / ls_thunder.max()

hd_cvdm = np.roll(hd_cvdm,20,axis=0)
hd_combined_image = np.zeros((*hd_cvdm.shape, 3), dtype=np.float32)
hd_combined_image[..., 0] = hd_cvdm / hd_cvdm.max()
hd_combined_image[..., 2] = hd_thunder / hd_thunder.max()

fig, ax = plt.subplots(2, 2, figsize=(5,5))

ax[0, 0].imshow(summed_ls, cmap='gray', vmin=0.0)
ax[1, 0].imshow(summed_hd, cmap='gray', vmin=0.0)
ax[0, 1].imshow(ls_combined_image)
ax[1, 1].imshow(hd_combined_image[128:,128:,:], cmap='gray', vmin=0.0)

#inset_ax = ax[0, 1].inset_axes([0.6, 0.6, 0.35, 0.35])
x_center, y_center = 70, 140
zoom_size = 40
x_min, x_max = x_center - zoom_size, x_center + zoom_size
y_min, y_max = y_center - zoom_size, y_center + zoom_size

#inset_ax.imshow(combined_image[y_min:y_max, x_min:x_max])
#inset_ax.set_xticks([])
#inset_ax.set_yticks([])
#inset_ax.set_aspect(1.0)

#for spine in inset_ax.spines.values():
#    spine.set_edgecolor("white")
#    spine.set_linewidth(2)

ax[0, 0].axis('off')
ax[1, 0].axis('off')
ax[0, 1].axis('off')
ax[1, 1].axis('off')

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
plt.savefig('/N/slate/cwseitz/cvdm/Tubes/Real/figure-11-1.png',dpi=200)
plt.show()

fig_line, ax_line = plt.subplots(2,1,figsize=(5,4))

x_hr = np.arange(50,80)
y_combined_r = ls_combined_image[120, x_hr, 0]
y_combined_r /= y_combined_r.max()
y_combined_b = ls_combined_image[120, x_hr, 2]
y_combined_b /= y_combined_b.max()
pixel_size = 25.0  # nm

x_hr -= 40
ax_line[0].plot((x_hr-x_hr[0]) * pixel_size, y_combined_r, 'r-', marker='o', label='CVDM')
ax_line[0].plot((x_hr-x_hr[0]) * pixel_size, y_combined_b, 'b-', marker='o', label='ThunderSTORM')
ax_line[0].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False)

ax_line[0].set_xlabel('Distance (nm)')
ax_line[0].set_ylabel('Intensity (a.u.)')
ax_line[0].spines['top'].set_visible(False)
ax_line[0].spines['right'].set_visible(False)

x_hr = np.arange(120,150)
y_combined_r = hd_combined_image[50, x_hr, 0]
y_combined_r /= y_combined_r.max()
y_combined_b = hd_combined_image[50, x_hr, 2]
y_combined_b /= y_combined_b.max()
pixel_size = 25.0  # nm

x_hr -= 40
ax_line[1].plot((x_hr-x_hr[0]) * pixel_size, y_combined_r, 'r-', marker='o', label='CVDM')
ax_line[1].plot((x_hr-x_hr[0]) * pixel_size, y_combined_b, 'b-', marker='o', label='ThunderSTORM')
ax_line[1].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False)

ax_line[1].set_xlabel('Distance (nm)')
ax_line[1].set_ylabel('Intensity (a.u.)')
ax_line[1].spines['top'].set_visible(False)
ax_line[1].spines['right'].set_visible(False)

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)

plt.tight_layout()
plt.savefig('/N/slate/cwseitz/cvdm/Tubes/Real/figure-11-2.png',dpi=200)
plt.show()
