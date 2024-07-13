import numpy as np


class DecisionStump:
    def __init__(self, error_rate_epsilon: float = 1e-3) -> None:
        self.error_rate_epsilon = error_rate_epsilon
        self.sign: int = None
        self.index: int = None
        self.threshold: float = None
        self.alpha: float = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.sign * np.sign(X[:, self.index] - self.threshold).astype(int)

    def fit(self, X: np.ndarray, y: np.ndarray, weight: np.ndarray) -> None:
        num_samples, num_features = X.shape
        min_error = np.inf
        for i in range(num_features):
            row_ind_sorted = np.argsort(X[:, i])
            labels_sorted = y[row_ind_sorted]
            weights_sorted = weight[row_ind_sorted]
            for j in range(1, num_samples):
                threshold = (X[row_ind_sorted[j - 1], i] + X[row_ind_sorted[j], i]) / 2
                error = np.dot(
                    weights_sorted,
                    (np.sign(X[:, i] - threshold).astype(int) != labels_sorted).astype(
                        int
                    ),
                )
                error_inv = np.dot(
                    weights_sorted,
                    (-np.sign(X[:, i] - threshold).astype(int) != labels_sorted).astype(
                        int
                    ),
                )
                if error < min_error or error_inv < min_error:
                    min_error = min(error, error_inv)
                    error_rate = min_error / np.sum(weights_sorted)
                    self.sign = 1 if error < error_inv else -1
                    self.index = i
                    self.threshold = threshold
                    self.alpha = np.log((1 - error_rate) / error_rate)
