import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

from imi_assignment_2.adaboost import AdaBoost

if __name__ == "__main__":
    classes_to_use = [1, 2]
    features_to_use = [2, 3]
    x_1_range = (2.5, 7.5)
    x_2_range = (0, 3.5)

    X, y = sklearn.datasets.load_iris(return_X_y=True)
    X = X[np.ix_(np.isin(y, classes_to_use), features_to_use)]
    y = np.where(y[np.isin(y, classes_to_use)] == classes_to_use[0], -1, 1)

    adaboost = AdaBoost(num_classifiers=10)
    start = time.perf_counter()
    adaboost.fit(X, y)
    end = time.perf_counter()
    print(f"Training time: {end - start:.3f}s")

    X1, X2 = np.meshgrid(
        np.arange(x_1_range[0], x_1_range[1] + 0.1, 0.1),
        np.arange(x_2_range[0], x_2_range[1] + 0.1, 0.1),
    )
    Z = adaboost.predict(np.column_stack((X1.ravel(), X2.ravel())))

    plt.rcParams["font.size"] = 16
    plt.contourf(X1, X2, Z.reshape(X1.shape), alpha=0.5, levels=1, cmap="bwr")
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c="b", label="versicolor")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="r", label="virginica")
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    plt.legend()
    plt.tight_layout()
    plt.show()
