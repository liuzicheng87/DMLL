import scipy.sparse
import scipy.stats
import DMLL
import matplotlib.pyplot as plt
import pylab
import os
import pickle
from sklearn import linear_model

import numpy as np

import sklearn.datasets

X, y = sklearn.datasets.make_classification(n_samples=100000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2)
X = scipy.sparse.csr_matrix(X)

Xtrain, ytrain = X[:99000], y[:99000]
Xtest, ytest = X[99000:], y[99000:]

Jext = 20
MahaFeatExt = DMLL.RBFMahaFeatExtSparse(Xtrain, Jext)
MahaFeatExt.fit(Xtrain, ytrain, optimiser=DMLL.GradientDescent(0.001, 0.5), GlobalBatchSize=500, MaxNumIterations=30)
Xext = MahaFeatExt.transform(Xtrain)

if DMLL.rank == 0:
    for i in range(Jext):
      print scipy.stats.pearsonr(Xext[:,i], ytrain)

#Erstelle Netz (fuer heatmaps)
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = DMLL.np.meshgrid(DMLL.np.arange(x_min, x_max, h),
                     DMLL.np.arange(y_min, y_max, h))

#----------------------------------Zeichne Heat Map-----------------------------------------

logreg = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6).fit(Xext, ytrain)

grid_data = MahaFeatExt.transform(scipy.sparse.csr_matrix(np.c_[xx.ravel(), yy.ravel()]))

Z = logreg.predict_proba(grid_data)[:,1]
Z = Z.reshape(xx.shape)

plt.imshow(Z, extent=[xx.min(), xx.max(), yy.max(), yy.min()])

#Plotte Trainingspunkte
plt.plot(Xtest[ytest==0, 0].todense(), Xtest[ytest==0, 1].todense(), 'co')
plt.plot(Xtest[ytest==1, 0].todense(), Xtest[ytest==1, 1].todense(), 'ro')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()
