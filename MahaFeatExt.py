import scipy.sparse
import scipy.stats
import DMLL
import matplotlib.pyplot as plt
import pylab

#Set SampleSize
#You can vary this number to see what happens
SampleSize = 25000

#Generate random dataset
if DMLL.rank == 0:
	GlobalX = DMLL.np.random.normal(0.0, 1.0, SampleSize*100).reshape(SampleSize, 100)
	GlobalY = 20.0 + DMLL.np.asarray(GlobalX.sum(axis=1).ravel()) + DMLL.np.random.normal(0.0, 3.0, SampleSize)

#Scatter GlobalX and GlobalY
if DMLL.rank == 0:
	X = DMLL.Scatter(GlobalX)
	Y = DMLL.Scatter1d(GlobalY)	
	#Since GlobalX is  now divided among the processes, we don't need it anymore and we should delete it to save space
	del GlobalX
	del GlobalY
else:
	X = DMLL.Scatter()
	Y = DMLL.Scatter1d()	
 
X = scipy.sparse.csr_matrix(X)

Jext = 4
MahaFeatExt = DMLL.LinearMahaFeatExtSparse(X.shape[1], Jext)
MahaFeatExt.fit(X, Y, optimiser=DMLL.GradientDescent(25.0, 0.1), GlobalBatchSize=0, tol=1e-08, MaxNumIterations=500, root=0)

Xext = MahaFeatExt.transform(X)

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
