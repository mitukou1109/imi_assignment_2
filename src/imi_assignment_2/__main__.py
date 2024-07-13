import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

from .adaboost import AdaBoost

iris = sklearn.datasets.load_iris()
X = iris.data[iris.target != 0, 2:]
y = iris.target[iris.target != 0] * 2 - 3

adaboost = AdaBoost(num_classifiers=100)
adaboost.fit(X, y)

x_1_min, x_1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x_2_min, x_2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
X1, X2 = np.meshgrid(np.arange(x_1_min, x_1_max, 0.1), np.arange(x_2_min, x_2_max, 0.1))
y_mesh = adaboost.predict(np.column_stack((X1.ravel(), X2.ravel())))

plt.contourf(X1, X2, y_mesh.reshape(X1.shape), alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
