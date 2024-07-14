import numpy as np

from .decision_stump import DecisionStump


class AdaBoost:
    def __init__(self, *, num_classifiers: int):
        self.num_classifiers = num_classifiers
        self.classifiers: list[DecisionStump] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.zeros((X.shape[0], self.num_classifiers), dtype=int)
        for i, classifier in enumerate(self.classifiers):
            predictions[:, i] = classifier.predict(X)
        return 2 * (np.sum(predictions, axis=1) > 0) - 1  # Prevent zero predictions

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classifiers = []
        weight = np.ones(y.shape) / y.shape[0]
        for _ in range(self.num_classifiers):
            classifier = DecisionStump()
            classifier.fit(X, y, weight)
            self.classifiers.append(classifier)
            weight *= np.exp(
                classifier.reliability * (classifier.predict(X) != y).astype(int)
            )
            weight /= np.sum(weight**2) ** 0.5
