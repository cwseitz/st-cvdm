import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

H, W = 40, 40
x = np.linspace(0, np.pi * 2, H)
y = np.linspace(0, np.pi * 2, W)
X, Y = np.meshgrid(x, y)
coords = np.column_stack([X.ravel(), Y.ravel()])

def cosine_covariance(d, k):
    return np.cos(k * d)

D = cdist(coords, coords)
mu = np.zeros(H*W)


nk = 32
kspace = np.linspace(1.0,5.0,nk)

for k in kspace:
    C = cosine_covariance(D, k)
    Z = np.random.multivariate_normal(mean=mu, cov=C)
    Z = Z.reshape((H,W))

    fig, ax = plt.subplots()
    ax.imshow(Z,cmap='coolwarm')
    plt.show()

