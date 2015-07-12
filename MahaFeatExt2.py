import scipy.sparse
import scipy.stats
import DMLL
import matplotlib.pyplot as plt
import pylab
import os
import pickle

#Set working directory
os.chdir("/home/patrick/SportScheckDaten/")

#Load data
data = pickle.load(open("Predicting Product Returns 2015-06-29.p", "rb"))

X_train_dense = data[0]
X_train_sparse = data[1]
X_test_dense  = data[2]
X_test_sparse = data[3]
Y_train = data[4]
Y_test = data[5]
del data

Jext = 10
MahaFeatExt = DMLL.LinearMahaFeatExtSparse(X_train_sparse.shape[1], Jext)
MahaFeatExt.fit(X_train_sparse, Y_train, optimiser=DMLL.GradientDescent(25.0, 0.1), GlobalBatchSize=0, tol=1e-08, MaxNumIterations=500, root=0)

Xext = MahaFeatExt.transform(X_train_sparse)

if DMLL.rank == 0:
    for i in range(Jext):
      print scipy.stats.pearsonr(Xext[:,i], Y)
        
    for i in range(Jext):
       for j in range(i):
           print str(i) + ", " + str(j) + ":"
           print scipy.stats.pearsonr(Xext[:,i], Xext[:,j])

if DMLL.rank == 0:
   SumGradients = MahaFeatExt.GetSumGradients()
   
   plt.grid(True)
   plt.plot(SumGradients)
   plt.show()
   
   print SumGradients
