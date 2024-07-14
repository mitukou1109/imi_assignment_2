import numpy as np


class DecisionStump:
    def __init__(self) -> None:
        self.sign: int = None
        self.index: int = None
        self.threshold: float = None
        self.reliability: float = None
        self.rng = np.random.default_rng()

    def _predict(
        self, X: np.ndarray, sign: int, index: int, threshold: float
    ) -> np.ndarray:
        return sign * (
            2 * ((X[:, index] - threshold) > 0) - 1
        )  # Not using np.sign to prevent zero predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict(X, self.sign, self.index, self.threshold)

    def fit(self, X: np.ndarray, y: np.ndarray, weight: np.ndarray) -> None:
        indices = np.arange(X.shape[1])
        self.rng.shuffle(indices)
        done = False
        for index in indices:
            values = np.unique(X[:, index])
            self.rng.shuffle(values)
            for threshold in values:
                error_rate = np.dot(
                    weight, (self._predict(X, 1, index, threshold) != y).astype(int)
                ) / np.sum(weight)
                if error_rate == 0.5:
                    continue
                else:
                    self.sign = 1 if error_rate < 0.5 else -1
                    self.index = index
                    self.threshold = threshold
                    if error_rate <= 0:
                        self.reliability = np.inf
                    elif error_rate >= 1:
                        self.reliability = 0
                    else:
                        self.reliability = np.log(1 / error_rate - 1)
                    done = True
                    break
            if done:
                break
