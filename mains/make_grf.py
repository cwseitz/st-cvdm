import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky
from skimage.io import imsave

H, W = 64, 64
x = np.linspace(0, np.pi * 2, H)
y = np.linspace(0, np.pi * 2, W)
X, Y = np.meshgrid(x, y)
coords = np.column_stack([X.ravel(), Y.ravel()])

def exponential_covariance(d, k):
    return np.exp(-k * d)

D = cdist(coords, coords)
mu = np.zeros(H * W)

nk = 32
num_samples = 500
kspace = np.linspace(0.01, 1.0, nk)

plt.imshow(exponential_covariance(D, 0.1))
plt.colorbar()
plt.title("Exponential Covariance Matrix (k=1.0)")
plt.show()

C_chol = []
for i, k in enumerate(kspace):
    print(f"Computing Cholesky decomposition {i+1}/{nk} for k={k:.2f}")
    C_chol.append(cholesky(exponential_covariance(D, k) + 1e-6 * np.eye(H * W), lower=True))

out = []
for n in range(num_samples):
    stack = []
    for m, L in enumerate(C_chol):
        print(f'Sample {n+1}/{num_samples}, k={kspace[m]:.2f}')
        Z = mu + L @ np.random.randn(H * W)
        stack.append(Z.reshape((H, W)))
    
    if n == 0:
        plt.imshow(stack[n], cmap="viridis")
        plt.colorbar()
        plt.title(f"Sample GRF (k={kspace[n]:.2f})")
        plt.show()

    out.append(np.array(stack))

out = np.array(out)
out = np.moveaxis(out,1,-1)
print(out.shape)
path = "/N/slate/cwseitz/st-cvdm/GRF/"
imsave(path + "lr.tif", out)

