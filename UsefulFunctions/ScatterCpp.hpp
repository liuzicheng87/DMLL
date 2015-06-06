void Scatter1dCpp(MPI_Comm comm, int rank, double *GlobalX, int GlobalI, double *X, int I, int *LocalI, int SizeLocalI, int *CumulativeLocalI, int SizeCumulativeLocalI, int root) {
	
	//Scatter the input values to each of the processes and wait for all
	MPI_Scatterv(GlobalX, LocalI, CumulativeLocalI, MPI_DOUBLE, X, LocalI[rank], MPI_DOUBLE, root, comm);
	MPI_Barrier(comm);

}

void ScatterCpp(MPI_Comm comm, int rank, double *GlobalX, int GlobalI, int GlobalJ, double *X, int I, int J, int *LocalI, int SizeLocalI, int *CumulativeLocalI, int SizeCumulativeLocalI, int root) {
	
	//Scatter the input values to each of the processes and wait for all
	MPI_Scatterv(GlobalX, LocalI, CumulativeLocalI, MPI_DOUBLE, X, LocalI[rank], MPI_DOUBLE, root, comm);
	MPI_Barrier(comm);

}
