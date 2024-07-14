import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

from .adaboost import AdaBoost
from .kernel import GaussKernel, LinearKernel
from .kernel_k_means import KernelKMeans

classes_to_use = [1, 2]
features_to_use = [2, 3]
x_1_range = (2.5, 7.5)
x_2_range = (0, 3.5)

X, y = sklearn.datasets.load_iris(return_X_y=True)
X = X[np.ix_(np.isin(y, classes_to_use), features_to_use)]
y = np.where(y[np.isin(y, classes_to_use)] == classes_to_use[0], -1, 1)

adaboost = AdaBoost(num_classifiers=10)
adaboost.fit(X, y)

X1, X2 = np.meshgrid(
    np.arange(x_1_range[0], x_1_range[1] + 0.1, 0.1),
    np.arange(x_2_range[0], x_2_range[1] + 0.1, 0.1),
)
adaboost_labels = adaboost.predict(np.column_stack((X1.ravel(), X2.ravel())))

linear_k_means = KernelKMeans(LinearKernel())
linear_k_means_labels = linear_k_means.classify(X, K=2)

gauss_k_means = KernelKMeans(GaussKernel(gamma=1.0))
gauss_k_means_labels = gauss_k_means.classify(X, K=2)

fig, axs = plt.subplots(1, 3, figsize=(14, 4))
plt.setp(axs, xlim=x_1_range, ylim=x_2_range)

axs[0].contourf(X1, X2, adaboost_labels.reshape(X1.shape), alpha=0.5, levels=1)
axs[0].scatter(X[:, 0], X[:, 1], c=y)

axs[1].scatter(
    X[linear_k_means_labels == 0, 0],
    X[linear_k_means_labels == 0, 1],
    c=y[linear_k_means_labels == 0],
    marker="o",
    cmap="bwr",
    vmin=-1,
    vmax=1,
)
axs[1].scatter(
    X[linear_k_means_labels == 1, 0],
    X[linear_k_means_labels == 1, 1],
    c=y[linear_k_means_labels == 1],
    marker="x",
    cmap="bwr",
    vmin=-1,
    vmax=1,
)

axs[2].scatter(
    X[gauss_k_means_labels == 0, 0],
    X[gauss_k_means_labels == 0, 1],
    c=y[gauss_k_means_labels == 0],
    marker="o",
    cmap="bwr",
    vmin=-1,
    vmax=1,
)
axs[2].scatter(
    X[gauss_k_means_labels == 1, 0],
    X[gauss_k_means_labels == 1, 1],
    c=y[gauss_k_means_labels == 1],
    marker="x",
    cmap="bwr",
    vmin=-1,
    vmax=1,
)

plt.show()
