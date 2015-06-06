import matplotlib.pyplot as plt
import pickle

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

#Initialise LinReg. Note that we do not fit!
LinReg = DMLL.LinearRegression(1)

#Get parameters and save
#Each process saves its own copy of the parameters
params = LinReg.GetParams()

#Load parameters
#Only root process needs to load parameters
root = 0
if (DMLL.rank == root):
	params = pickle.load(open("LinRegParams" + str(DMLL.rank) + ".p", "rb"))
	LinReg.SetParams(params, root)
else:
	LinReg.SetParams(root)
	
#Predict
Yhat = LinReg.predict(X)

#Show results
#You can also vary the number of the rank (as long as the process exists) to verify that the parameters are properly broadcast
#Show results
if DMLL.rank == 0:
	#Print W
	params = LinReg.GetParams()
	print params
	#Plot W
	plt.grid(True)
	plt.plot(X,Y, 'ro')
	plt.plot(X,Yhat, '-')
	plt.show()
