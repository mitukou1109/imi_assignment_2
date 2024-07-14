import numpy as np


class Kernel:
    def __init__(self) -> None:
        pass

    def __call__(self) -> np.ndarray:
        raise NotImplementedError


class LinearKernel(Kernel):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.dot(x, y)


class GaussKernel(Kernel):
    def __init__(self, gamma: float) -> None:
        super().__init__()
        self.gamma = gamma

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.exp(-self.gamma * np.sqrt(np.sum((x - y) ** 2)))
