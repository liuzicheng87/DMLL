import scipy.sparse
import scipy.stats
import DMLL
import matplotlib.pyplot as plt
import pylab
import os
import pickle

import sklearn.datasets

X, y = sklearn.datasets.make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2)
X = scipy.sparse.csr_matrix(X)

Xtrain, ytrain = X[:9000], y[:9000]
Xtest, ytest = X[9000:], y[9000:]

Jext = 1
c = Xtrain[DMLL.np.random.randint(Xtrain.shape[0])]

MahaFeatExt = DMLL.RBFMahaFeatExtSparse(Xtrain.shape[1], c)
MahaFeatExt.fit(X, y, DMLL.GradientDescent(0.5, 0.1), 0, 1e-08, 500)

Xext = MahaFeatExt.transform(Xtrain)

if DMLL.rank == 0:
    for i in range(Jext):
      print scipy.stats.pearsonr(Xext[:,i], ytrain)
        
    for i in range(Jext):
       for j in range(i):
           print str(i) + ", " + str(j) + ":"
           print scipy.stats.pearsonr(Xext[:,i], Xext[:,j])

if DMLL.rank == 0:
   SumGradients = MahaFeatExt.GetSumGradients()
   
   plt.grid(True)
   plt.plot(SumGradients)
   plt.show()
   

#Erstelle Netz (fuer heatmaps)
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = DMLL.np.meshgrid(DMLL.np.arange(x_min, x_max, h),
                     DMLL.np.arange(y_min, y_max, h))

#----------------------------------Zeichne Heat Map-----------------------------------------

Z = MahaFeatExt.transform(scipy.sparse.csr_matrix(DMLL.np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

print Z

plt.imshow(Z, extent=[xx.min(), xx.max(), yy.max(), yy.min()])

#Plotte Trainingspunkte
plt.plot(Xtest[ytest==0, 0].todense(), Xtest[ytest==0, 1].todense(), 'co')
plt.plot(Xtest[ytest==1, 0].todense(), Xtest[ytest==1, 1].todense(), 'ro')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()
