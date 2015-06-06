class SayHelloCpp {
	
	public:
	SayHelloCpp();
	~SayHelloCpp();
	
	void hello (MPI_Comm comm, int rank);
	
};

void BcastCpp(MPI_Comm comm, int rank, int *X, int I, int root);
void CalcLocalICpp (int *LocalI, int SizeLocalI, int *CumulativeLocalI, int SizeCumulativeLocalI, const int GlobalI);
void Scatter1dCpp(MPI_Comm comm, int rank, double *GlobalX, int GlobalI, double *X, int I, int *LocalI, int SizeLocalI, int *CumulativeLocalI, int SizeCumulativeLocalI, int root);
void ScatterCpp(MPI_Comm comm, int rank, double *GlobalX, int GlobalI, int GlobalJ, double *X, int I, int J, int *LocalI, int SizeLocalI, int *CumulativeLocalI, int SizeCumulativeLocalI, int root);
