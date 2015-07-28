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

root = 0

if DMLL.rank == root:
	X, y = sklearn.datasets.make_classification(n_samples=100000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2)
	#X = scipy.sparse.csr_matrix(X)
	XtrainGlobal, ytrainGlobal = X[:99000], y[:99000]
	Xtest, ytest = X[99000:], y[99000:]
	Xtest = scipy.sparse.csr_matrix(Xtest)
	Xtrain = DMLL.Scatter(XtrainGlobal, root)
	ytrain = DMLL.Scatter1d(ytrainGlobal, root)
	del XtrainGlobal
	del ytrainGlobal
	del X
	del y
else:
    Xtrain = DMLL.Scatter()
    ytrain = DMLL.Scatter1d()

Xtrain = scipy.sparse.csr_matrix(Xtrain)

Jext = 50
MahaFeatExt = DMLL.RBFMahaFeatExtSparse(Xtrain, Jext, clipW=True)
MahaFeatExt.fit(Xtrain, ytrain, optimiser=DMLL.AdaGrad(0.08, 0.0), GlobalBatchSize=0, MaxNumIterations=120)

Xext = MahaFeatExt.transform(Xtrain)
Xtransform = scipy.sparse.csr_matrix(Xext)

Jext2 = 2

MahaFeatExt2 = DMLL.LinearMahaFeatExtSparse(Jext, Jext2)
MahaFeatExt2.fit(Xtransform, ytrain, optimiser=DMLL.GradientDescentWithMomentum(25.0, 0.5, 0.5), GlobalBatchSize=0, tol=1e-08, MaxNumIterations=1200, root=0)
Xtransform2 = MahaFeatExt2.transform(Xtransform)

if DMLL.rank == 0:
   SumGradients = MahaFeatExt2.GetSumGradients()
   
   plt.grid(True)
   plt.plot(SumGradients)
   plt.show()
   
   W = MahaFeatExt2.GetParams()
   W = W.reshape(Jext2, Jext)
   print W
   print np.dot(W, W.transpose())
   
if DMLL.rank == root:
	Xexttest = MahaFeatExt.transform(Xtest)
	Xtransformtest = scipy.sparse.csr_matrix(Xexttest)
	Xtransformtest2 = MahaFeatExt2.transform(Xtransformtest)
	for i in range(Jext2):
		print scipy.stats.pearsonr(Xtransformtest2[:,i], ytest)
	for i in range(Jext2):
		for j in range(i):
			print str(i) + ", " + str(j) + ":"
			print scipy.stats.pearsonr(Xtransformtest2[:,i], Xtransformtest2[:,j])
	print
	W =  MahaFeatExt.GetParams()
	for i in range(Jext):
		print W[i]
           
if DMLL.rank == root:
	for i in range(Jext):
		print scipy.stats.pearsonr(Xext[:,i], ytrain)
	#Erstelle Netz (fuer heatmaps)
	h = .02
	x_min, x_max = Xtest[:, 0].min() - 1, Xtest[:, 0].max() + 1
	y_min, y_max = Xtest[:, 1].min() - 1, Xtest[:, 1].max() + 1
	xx, yy = DMLL.np.meshgrid(DMLL.np.arange(x_min, x_max, h),
                     DMLL.np.arange(y_min, y_max, h))
	
	#----------------------------------Zeichne Heat Map-----------------------------------------
	
	logreg = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6).fit(Xtransform2, ytrain)

	grid_data = MahaFeatExt2.transform(scipy.sparse.csr_matrix(MahaFeatExt.transform(scipy.sparse.csr_matrix(np.c_[xx.ravel(), yy.ravel()]))))

	Z = logreg.predict_proba(grid_data)[:,1]
	#Z = logreg.predict(grid_data)	
	Z = Z.reshape(xx.shape)
	
	plt.imshow(Z, extent=[xx.min(), xx.max(), yy.max(), yy.min()])
	
	#Plotte Trainingspunkte
	plt.plot(Xtest[ytest==0, 0].todense(), Xtest[ytest==0, 1].todense(), 'co')
	plt.plot(Xtest[ytest==1, 0].todense(), Xtest[ytest==1, 1].todense(), 'ro')
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.show()
	
	#----------------------------------SumGradients-----------------------------------------
	
	SumGradients = MahaFeatExt.GetSumGradients()
	
	bg = scipy.sparse.csr_matrix(np.c_[xx.ravel(), yy.ravel()])
	
	for i in range(5):
		plt.grid(True)
		plt.plot(SumGradients[i])
		plt.title(scipy.stats.pearsonr(Xext[:,i], ytrain)[0])
		plt.show()
		Z = np.zeros((1, bg.shape[0]))
		MahaFeatExt.thisptr[i].transform(Z, bg.data, bg.indices, bg.indptr, bg.shape[1])
		Z = Z.reshape(xx.shape)
		plt.imshow(Z, extent=[xx.min(), xx.max(), yy.max(), yy.min()])
		plt.plot(Xtest[ytest==0, 0].todense(), Xtest[ytest==0, 1].todense(), 'co')
		plt.plot(Xtest[ytest==1, 0].todense(), Xtest[ytest==1, 1].todense(), 'ro')
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.title(scipy.stats.pearsonr(Xext[:,i], ytrain)[0])		
		plt.show()
