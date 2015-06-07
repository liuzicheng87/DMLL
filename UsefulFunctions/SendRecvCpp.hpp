//Send

void SendDoubleCpp(MPI_Comm comm, double *X, int I, int destination) {
				
	//Send the input values to destination
	MPI_Send(X, I, MPI_DOUBLE, destination, 0, comm);
	
}

void SendIntCpp(MPI_Comm comm, int *X, int I, int destination) {
				
	//Send the input values to destination
	MPI_Send(X, I, MPI_INT, destination, 0, comm);
	
}

//Recv

void RecvDoubleCpp(MPI_Comm comm, double *X, int I, int source) {
				
	//Send the input values to destination
	MPI_Recv(X, I, MPI_DOUBLE, source, 0, comm, MPI_STATUS_IGNORE);
	
}

void RecvIntCpp(MPI_Comm comm, int *X, int I, int source) {
				
	//Send the input values to destination
	MPI_Recv(X, I, MPI_INT, source, 0, comm, MPI_STATUS_IGNORE);
	
}
