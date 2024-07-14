import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

from imi_assignment_2.kernel import GaussKernel, LinearKernel
from imi_assignment_2.kernel_k_means import KernelKMeans

if __name__ == "__main__":
    X, y = sklearn.datasets.make_circles(n_samples=1000, noise=0.1, factor=0.2)

    linear_k_means = KernelKMeans(LinearKernel())
    start = time.perf_counter()
    linear_k_means_labels = linear_k_means.classify(X, K=2)
    end = time.perf_counter()
    print(f"Linear training time: {end - start}s")
    if np.count_nonzero(linear_k_means_labels != y) > y.shape[0] // 2:
        linear_k_means_labels = np.where(linear_k_means_labels == 0, 1, 0)

    gauss_k_means = KernelKMeans(GaussKernel(gamma=2.5))
    start = time.perf_counter()
    gauss_k_means_labels = gauss_k_means.classify(X, K=2, converge_threshold=50)
    end = time.perf_counter()
    print(f"Gauss training time: {end - start}s")
    if np.count_nonzero(gauss_k_means_labels != y) > y.shape[0] // 2:
        gauss_k_means_labels = np.where(gauss_k_means_labels == 0, 1, 0)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].scatter(X[:, 0], X[:, 1], c=y, cmap="bwr")
    axs[1].scatter(X[:, 0], X[:, 1], c=linear_k_means_labels, cmap="bwr")
    axs[2].scatter(X[:, 0], X[:, 1], c=gauss_k_means_labels, cmap="bwr")
    fig.tight_layout()
    plt.show()
