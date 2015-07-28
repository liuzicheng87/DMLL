class LinearRegressionCpp: public NumericallyOptimisedMLAlgorithmCpp {

	public:

	LinearRegressionCpp(int J, RegulariserCpp *regulariser): NumericallyOptimisedMLAlgorithmCpp();
	
	~LinearRegressionCpp();

	void fit (MPI_Comm comm, double *X, int I, int J, double *Y, int IY, OptimiserCpp *optimiser, int BatchSize, const double tol, const int MaxNumIterations);
		
	void predict (double *Yhat, int IY, double *X, int I, int J);
			
};

class LogisticRegressionCpp: public NumericallyOptimisedMLAlgorithmCpp {

	public:

	LogisticRegressionCpp(int J, RegulariserCpp *regulariser): NumericallyOptimisedMLAlgorithmCpp();
	
	~LogisticRegressionCpp();

	void fit (MPI_Comm comm, double *X, int I, int J, double *Y, int IY, OptimiserCpp *optimiser, int BatchSize, const double tol, const int MaxNumIterations);
		
	void predict (double *Yhat, int IY, double *X, int I, int J);
			
};
