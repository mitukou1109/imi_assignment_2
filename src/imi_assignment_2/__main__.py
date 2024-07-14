import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

from .adaboost import AdaBoost

classes_to_use = [1, 2]
features_to_use = [2, 3]

X, y = sklearn.datasets.load_iris(return_X_y=True)
X = X[np.ix_(np.isin(y, classes_to_use), features_to_use)]
y = np.where(y[np.isin(y, classes_to_use)] == classes_to_use[0], -1, 1)

adaboost = AdaBoost(num_classifiers=10)
adaboost.fit(X, y)

x_1_min, x_1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x_2_min, x_2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
X1, X2 = np.meshgrid(np.arange(x_1_min, x_1_max, 0.1), np.arange(x_2_min, x_2_max, 0.1))
Z = adaboost.predict(np.column_stack((X1.ravel(), X2.ravel()))).reshape(X1.shape)

plt.contourf(X1, X2, Z, alpha=0.5, levels=1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
