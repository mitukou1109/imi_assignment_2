import numpy as np

from .kernel import Kernel


class KernelKMeans:
    def __init__(self, kernel: Kernel) -> None:
        self.kernel = kernel
        self.rng = np.random.default_rng()

    def classify(self, X: np.ndarray, K: int) -> np.ndarray:
        labels = self.rng.integers(K, size=X.shape[0])
        while True:
            distances = np.zeros((X.shape[0], K))
            for n in range(X.shape[0]):
                for k in range(K):
                    distances[n, k] = self.calc_distance(X[n], X[labels == k])
            new_labels = np.argmin(distances, axis=1)
            if (new_labels == labels).all():
                break
            else:
                labels = new_labels
        return labels

    def calc_distance(self, x: np.ndarray, C_k: np.ndarray) -> np.ndarray:
        distance = self.kernel(x, x)
        for x_j in C_k:
            distance += -2 / C_k.shape[0] * self.kernel(x, x_j)
            for x_i in C_k:
                distance += 1 / (C_k.shape[0] ** 2) * self.kernel(x_i, x_j)
        return distance
