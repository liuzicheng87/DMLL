import DMLL

#We create a two-dimensional array called GlobalX, which we want to divide evenly among the processes along its first dimension
# Note that DMLL.Scatter only works for 2-dimensional double arrays.
if DMLL.rank == 0:
	GlobalX = DMLL.np.arange(20000.0).reshape(2000, 10)
	X = DMLL.Scatter(GlobalX)
	#Since GlobalX is  now divided among the processes, we don't need it anymore and we should delete it to save space
	del GlobalX
else:
	X = DMLL.Scatter()

#We print only the first line of the newly created local variable X
print X[0]

#If the sending process is not the main thread, we can set root accordingly
root = 1

#We create a two-dimensional array called GlobalX, which we want to divide evenly among the processes along its first dimension
# Note that DMLL.Scatter only works for 2-dimensional double arrays.
if DMLL.rank == root:
	GlobalX = DMLL.np.arange(20000.0).reshape(2000, 10)
	X = DMLL.Scatter(GlobalX=GlobalX, root=root)
	#Since GlobalX is  now divided among the processes, we don't need it anymore and we should delete it to save space
	del GlobalX
else:
	X = DMLL.Scatter(root=root)

#We print only the first line of the newly created local variable X
print X[0]
