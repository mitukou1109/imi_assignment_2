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
            for k in range(K):
                distances[:, k] = self.calc_distance(X, X[labels == k])
            new_labels = np.argmin(distances, axis=1)
            if (new_labels == labels).all():
                break
            else:
                labels = new_labels
        return labels

    def calc_distance(self, X: np.ndarray, C_k: np.ndarray) -> np.ndarray:
        return (
            self.kernel(X, X)
            - 2
            / C_k.shape[0]
            * np.sum(
                self.kernel(
                    X[:, :, np.newaxis].repeat(C_k.shape[0], axis=2),
                    C_k.reshape(1, -1, C_k.shape[0]).repeat(X.shape[0], axis=0),
                ),
                axis=1,
            )
            + 1
            / (C_k.shape[0] ** 2)
            * np.sum(
                self.kernel(
                    C_k[:, :, np.newaxis].repeat(C_k.shape[0], axis=2),
                    C_k.reshape(1, -1, C_k.shape[0]).repeat(C_k.shape[0], axis=0),
                )
            )
        )
