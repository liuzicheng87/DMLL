import matplotlib.pyplot as plt

import DMLL

#Set SampleSize
#You can vary this number to see what happens
SampleSize = 250000

#Generate random dataset
if DMLL.rank == 0:
	GlobalX = DMLL.np.random.normal(0.0, 1.0, SampleSize)
	GlobalY = 20.0 + 10.0*GlobalX + DMLL.np.random.normal(0.0, 3.0, SampleSize)
	#Reshape GlobalX and GlobalY
	GlobalX = GlobalX.reshape(len(GlobalX),1)

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

#Show random dataset
if DMLL.rank == 0:
	plt.grid(True)
	plt.plot(X,Y, 'ro')
	plt.show()

thisptr = DMLL.LinearRegression(1)
thisptr.fit(X, Y, optimiser=DMLL.GradientDescentWithMomentum(1.0, 0.1, 0.5), MaxNumIterations=100)

SumGradients = thisptr.GetSumGradients()

if DMLL.rank == 0:
	plt.grid(True)
	plt.plot(SumGradients)
	plt.show()

Yhat = thisptr.predict(X)

if DMLL.rank == 0:
	plt.grid(True)
	plt.plot(X,Y, 'ro')
	plt.plot(X,Yhat, '-')
	plt.show()
