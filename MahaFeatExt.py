import scipy.sparse
import scipy.stats
import DMLL
import matplotlib.pyplot as plt
import pylab

root = 0
N = 10000
    
#Create random sparse dataset and scatter
X = scipy.sparse.csr_matrix(scipy.sparse.rand(N, 100, 0.1))
Y = DMLL.np.random.rand(N) + DMLL.np.asarray(X.sum(axis=1).ravel())
Y = Y.ravel()

Jext = 4
MahaFeatExt = DMLL.LinearMahaFeatExtSparse(X.shape[1], Jext)
MahaFeatExt.fit(X, Y, optimiser=DMLL.DMLLCpp.GradientDescent(100.0, 0.5, DMLL.size, DMLL.rank), GlobalBatchSize=0, tol=1e-08, MaxNumIterations=1000, root=0)

#Xext = MahaFeatExt.transform(X)

#for i in range(Jext):
#  print scipy.stats.pearsonr(Xext[:,i], Y)
    
#for i in range(Jext):
#   for j in range(i):
#       print str(i) + ", " + str(j) + ":"
 #       print scipy.stats.pearsonr(Xext[:,i], Xext[:,j])

SumGradients = MahaFeatExt.GetSumGradients()

if DMLL.rank == root:
	plt.grid(True)
	plt.plot(SumGradients)
	plt.show()
	
	print SumGradients
