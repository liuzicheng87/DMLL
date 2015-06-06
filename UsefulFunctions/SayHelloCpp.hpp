class SayHelloCpp {
	
	public:
	SayHelloCpp() {};
	~SayHelloCpp() {};
	
	void hello (MPI_Comm comm, int rank) {
		
		// Get the number of processes
		int world_size;
		MPI_Comm_size(comm, &world_size);

		// Print off a hello world message
		printf("Hello world, rank %d out of %d processors\n", rank, world_size);	
	}
	
};
