import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave

num_samples = 1000
num_channels = 32
height, width = 64,64

A = np.random.randn(num_channels, num_channels)
cov_matrix = A @ A.T

image_stack = np.zeros((num_samples, height, width, num_channels), dtype=np.float32)

for i in range(num_samples):
    noise = np.random.randn(num_channels, height * width)
    L = np.linalg.cholesky(cov_matrix)
    correlated_noise = L @ noise
    image_stack[i] = correlated_noise.reshape(num_channels, height, width).transpose(1, 2, 0)

print(image_stack.shape)
reshaped_data = image_stack[0].reshape(-1, num_channels)
empirical_cov_matrix = np.cov(reshaped_data, rowvar=False)
image_stack = image_stack.astype(np.float32)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(cov_matrix, cmap="coolwarm")
axes[0].set_title("Theoretical Covariance Matrix")
axes[1].imshow(empirical_cov_matrix, cmap="coolwarm")
axes[1].set_title("Empirical Covariance Matrix")
plt.show()

print(image_stack.shape)
path = "/N/slate/cwseitz/st-cvdm/Ncorr/"
imsave(path + "lr.tif", image_stack)
np.save(path + "cov_matrix.npy", cov_matrix)
