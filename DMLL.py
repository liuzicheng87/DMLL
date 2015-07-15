from datetime import datetime
from mpi4py import MPI
import numpy as np

import DMLLCpp

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
name = MPI.Get_processor_name()

#Useful functions

def CalcLocalI (GlobalI):
	LocalI = np.zeros(size).astype(np.int32)
	CumulativeLocalI = np.zeros(size).astype(np.int32)
	DMLLCpp.CalcLocalICpp(LocalI, CumulativeLocalI, GlobalI)
	return LocalI, CumulativeLocalI

class SayHello:
    def __init__(self):
        self.thisptr = DMLLCpp.SayHelloCpp()
    def hello(self):
        self.thisptr.hello(MPI.COMM_WORLD, rank)

#Scatter is used to scatter a numpy array
def Scatter1d(GlobalX=np.zeros(0), root=0, rank=rank):
	#Place a barrier before getting the time
	MPI.COMM_WORLD.barrier()
	StartTiming = datetime.now()	#Calculate the length of the respective arrays and broadcast them to all processes
	#Calculate the length of the respective arrays and broadcast them to all processes
	#Only the the root process knows the length of GlobalX, so only the root process can calculate the local lengths properly
	if rank == root:
		LocalI, CumulativeLocalI = CalcLocalI(GlobalX.shape[0])
		DMLLCpp.BcastCpp(MPI.COMM_WORLD, rank, LocalI, root)			
		DMLLCpp.BcastCpp(MPI.COMM_WORLD, rank, CumulativeLocalI, root)					
	else:
		LocalI = np.zeros(size).astype(np.int32)
		CumulativeLocalI = np.zeros(size).astype(np.int32)
		DMLLCpp.BcastCpp(MPI.COMM_WORLD, rank, LocalI, root)			
		DMLLCpp.BcastCpp(MPI.COMM_WORLD, rank, CumulativeLocalI, root)					
	#Now set up X according to the local length and scatter GlobalX
	X = np.zeros(LocalI[rank])
	DMLLCpp.Scatter1dCpp(MPI.COMM_WORLD, rank, GlobalX, X, LocalI, CumulativeLocalI, root) 
	#Get the time
	StopTiming = datetime.now()
	TimeElapsed = StopTiming - StartTiming
	if rank==root:
		print "Scattered two-dimensional numpy array."
		print "Time taken: %.2dh:%.2dm:%.2d.%.6ds" % (TimeElapsed.seconds//3600, TimeElapsed.seconds//60, TimeElapsed.seconds%60, TimeElapsed.microseconds)	
		print		
	return X

#Scatter is used to scatter a two-dimensional numpy array
def Scatter(GlobalX=np.zeros((0,0)), root=0, rank=rank):
	#Place a barrier before getting the time
	MPI.COMM_WORLD.barrier()
	StartTiming = datetime.now()	#Calculate the length of the respective arrays and broadcast them to all processes
	#Only the the root process knows I and J for GlobalX, so only the root process can calculate the local lengths properly
	if rank == root:
		J = np.zeros(1).astype(np.int32)		
		J[0] = GlobalX.shape[1]
		LocalI, CumulativeLocalI = CalcLocalI(GlobalX.shape[0])
		DMLLCpp.BcastCpp(MPI.COMM_WORLD, rank, J, root)					
		DMLLCpp.BcastCpp(MPI.COMM_WORLD, rank, LocalI, root)			
		DMLLCpp.BcastCpp(MPI.COMM_WORLD, rank, CumulativeLocalI, root)					
	else:
		LocalI = np.zeros(size).astype(np.int32)
		J = np.zeros(1).astype(np.int32)				
		CumulativeLocalI = np.zeros(size).astype(np.int32)
		DMLLCpp.BcastCpp(MPI.COMM_WORLD, rank, J, root)							
		DMLLCpp.BcastCpp(MPI.COMM_WORLD, rank, LocalI, root)			
		DMLLCpp.BcastCpp(MPI.COMM_WORLD, rank, CumulativeLocalI, root)					
	#Now set up X according to the local length and scatter GlobalX
	X = np.zeros((LocalI[rank], J[0]))
	#Because X is two-dimensional, we need to recalculate LocalI and CumulativeLocalI before broadcasting
	LocalI *= J[0]
	CumulativeLocalI *= J[0]
	#Scatter
	DMLLCpp.ScatterCpp(MPI.COMM_WORLD, rank, GlobalX, X, LocalI, CumulativeLocalI, root) 
	#Get the time
	StopTiming = datetime.now()
	TimeElapsed = StopTiming - StartTiming
	if rank==root:
		print "Scattered one-dimensional numpy array."
		print "Time taken: %.2dh:%.2dm:%.2d.%.6ds" % (TimeElapsed.seconds//3600, TimeElapsed.seconds//60, TimeElapsed.seconds%60, TimeElapsed.microseconds)	
		print		
	return X

#optimisers
#This is just a template that cannot be called
class NumericallyOptimisedMLAlgorithm:
	def GetSumGradients(self):
		IterationsNeeded = self.thisptr.GetIterationsNeeded()
		SumGradients = np.zeros(IterationsNeeded)
		self.thisptr.GetSumGradients(SumGradients)
		return SumGradients
	def GetParams(self):
		#Initialise W
		lengthW = self.thisptr.GetLengthW()
		W = np.zeros(lengthW)
		#Get J and W
		self.thisptr.GetParams(W)
		#Return J and W as a tuple
		return W
	def SetParams(self, params=0, root=0):
		#Broadcast parameters from root to all other processes
		params = MPI.COMM_WORLD.bcast(params, root)
		#Set W
		self.thisptr.SetParams(params)	
		
#Wrapper class for GradientDescent optimiser
class GradientDescent:
	def __init__(self, LearningRate, LearningRatePower):
		self.thisptr = DMLLCpp.GradientDescentCpp(LearningRate, LearningRatePower, size, rank)

#Wrapper class for BacktrackingLineSearch optimiser
class BacktrackingLineSearch:
	def __init__(self, LearningRateStart, LearningRateReduction, c, tol):
		self.thisptr = DMLLCpp.BacktrackingLineSearchCpp(LearningRateStart, LearningRateReduction, c, tol, size, rank)
	
#DimensionalityReduction
class LinearMahaFeatExtSparse(NumericallyOptimisedMLAlgorithm):
	def __init__(self, J, Jext):
		self.Jext = Jext
		self.thisptr = DMLLCpp.LinearMahaFeatExtSparseCpp(J, Jext)
	def fit(self, X, Y, optimiser=GradientDescent(25.0, 0.1), GlobalBatchSize=0, tol=1e-08, MaxNumIterations=500, root=0):
		#Place a barrier before getting the time
		MPI.COMM_WORLD.barrier()
		StartTiming = datetime.now()	#Calculate the length of the respective arrays and broadcast them to all processes
		#Do the actual fitting
		self.thisptr.fit(MPI.COMM_WORLD, rank, size, X.data, X.indices, X.indptr, Y, X.shape[1], optimiser.thisptr, GlobalBatchSize, tol, MaxNumIterations)
		#Get the time
		StopTiming = datetime.now()
		TimeElapsed = StopTiming - StartTiming		
		if rank==root:
			print "Trained Linear Mahalanobis Feature Extraction."
			print "Time taken: %.2dh:%.2dm:%.2d.%.6ds" % (TimeElapsed.seconds//3600, TimeElapsed.seconds//60, TimeElapsed.seconds%60, TimeElapsed.microseconds)	
			print				
	def transform(self, X):
		Xext = np.zeros((self.Jext, X.shape[0]))
		self.thisptr.transform(Xext, X.data, X.indices, X.indptr, X.shape[1])
		#transform returns the transpose of Xext, so we transpose it
		Xext = Xext.transpose()
		return Xext

#DimensionalityReduction
class RBFMahaFeatExtSparse:
	def __init__(self, X, Jext):
		self.J = X.shape[1]		
		self.Jext = Jext
		self.thisptr = list()
		for i in range(self.Jext):
			c = X[np.random.randint(X.shape[0])]
			self.thisptr.append(DMLLCpp.RBFMahaFeatExtSparseCpp(self.J, c.data, c.indices, c.indptr, 1))
	def fit(self, X, Y, optimiser=GradientDescent(25.0, 0.1), GlobalBatchSize=0, tol=1e-08, MaxNumIterations=500, root=0):
		#Place a barrier before getting the time
		MPI.COMM_WORLD.barrier()
		StartTiming = datetime.now()	#Calculate the length of the respective arrays and broadcast them to all processes
		#Do the actual fitting
		for i in range(self.Jext):
			self.thisptr[i].fit(MPI.COMM_WORLD, rank, size, X.data, X.indices, X.indptr, Y, X.shape[1], optimiser.thisptr, GlobalBatchSize, tol, MaxNumIterations)
		#Get the time
		StopTiming = datetime.now()
		TimeElapsed = StopTiming - StartTiming		
		if rank==root:
			print "Trained RBF Mahalanobis Feature Extraction."
			print "Time taken: %.2dh:%.2dm:%.2d.%.6ds" % (TimeElapsed.seconds//3600, TimeElapsed.seconds//60, TimeElapsed.seconds%60, TimeElapsed.microseconds)	
			print				
	def transform(self, X):
		Xext = np.zeros((X.shape[0], 0))
		for i in range(self.Jext):
			XextCol = np.zeros((1, X.shape[0]))
			self.thisptr[i].transform(XextCol, X.data, X.indices, X.indptr, self.J)
			Xext = np.hstack((Xext, XextCol.transpose()))
		return Xext

#linear
class LinearRegression(NumericallyOptimisedMLAlgorithm):
	def __init__(self, J):
		self.thisptr = DMLLCpp.LinearRegressionCpp(J)
	def fit(self, X, Y, optimiser=GradientDescent(1.0, 0.1), GlobalBatchSize=0, tol=1e-08, MaxNumIterations=500, root=0):
		#Place a barrier before getting the time
		MPI.COMM_WORLD.barrier()
		StartTiming = datetime.now()	#Calculate the length of the respective arrays and broadcast them to all processes
		#Do the actual fitting
		self.thisptr.fit(MPI.COMM_WORLD, X, Y, optimiser.thisptr, GlobalBatchSize, tol, MaxNumIterations)
		#Get the time
		StopTiming = datetime.now()
		TimeElapsed = StopTiming - StartTiming		
		if rank==root:
			print "Trained linear regression."
			print "Time taken: %.2dh:%.2dm:%.2d.%.6ds" % (TimeElapsed.seconds//3600, TimeElapsed.seconds//60, TimeElapsed.seconds%60, TimeElapsed.microseconds)	
			print				
	def predict(self, X):
		Yhat = np.zeros(len(X))
		self.thisptr.predict(Yhat, X)
		return Yhat

