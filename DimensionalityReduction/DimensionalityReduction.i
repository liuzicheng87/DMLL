class LinearMahaFeatExtSparseCpp: public NumericallyOptimisedMLAlgorithmCpp {

	public:

	LinearMahaFeatExtSparseCpp(const int J, const int Jext): NumericallyOptimisedMLAlgorithmCpp();
	
	~LinearMahaFeatExtSparseCpp();

	void fit (MPI_Comm comm, int rank, int size, double *XData, int XDataLength,  int *XIndices, int XIndicesLength,  int *XIndptr, int XIndptrLength,  double *Y,  int IY, const int J, OptimiserCpp *optimiser, int GlobalBatchSize, const double tol, const int MaxNumIterations);

	void transform(double *Xext, int I, int Jext, double *XData, int XDataLength, int *XIndices, int XIndicesLength, int *XIndptr, int XIndptrLength, const int J);
			
};
