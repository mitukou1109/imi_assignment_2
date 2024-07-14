import numpy as np


class Kernel:
    def __init__(self) -> None:
        pass

    def __call__(self) -> float:
        raise NotImplementedError


class LinearKernel(Kernel):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> float:
        return np.sum(X * Y, axis=1)


class GaussKernel(Kernel):
    def __init__(self, gamma: float) -> None:
        super().__init__()
        self.gamma = gamma

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> float:
        return np.exp(-self.gamma * np.sum((X - Y) ** 2, axis=1))
