void BcastCpp(MPI_Comm comm, int rank, int *X, int I, int root) {
				
	//Brodcast the input values to each of the processes and wait for all
	MPI_Bcast(X, I, MPI_INT, root, comm);
	MPI_Barrier(comm);

}
