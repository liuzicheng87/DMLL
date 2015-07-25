class NumericallyOptimisedMLAlgorithmCpp {

	public:
			
	NumericallyOptimisedMLAlgorithmCpp();
	
	virtual ~NumericallyOptimisedMLAlgorithmCpp();
	
	int GetIterationsNeeded();
		
	void GetSumGradients (double *SumGradients, int IterationsNeeded);
	
	void GetParams(double *W, int lengthW);
		
	void SetParams(double *W, int lengthW);
	
	int GetLengthW();	
};

class OptimiserCpp {
	
	public:

	OptimiserCpp(const int size, const int rank);

	virtual ~OptimiserCpp();
	
};

class GradientDescentCpp: public OptimiserCpp {
	
	public:
		
	GradientDescentCpp (double LearningRate, double LearningRatePower, const int size, const int rank):OptimiserCpp(size, rank);

	~GradientDescentCpp();
				
};

class BacktrackingLineSearchCpp: public OptimiserCpp {
	
	public:
		
	BacktrackingLineSearchCpp (double LearningRateStart, double LearningRateReduction, double c, double tol, const int size, const int rank):OptimiserCpp(size, rank);

	~BacktrackingLineSearchCpp();
				
};

class GradientDescentWithMomentumCpp: public OptimiserCpp {
	
	public:
		
	GradientDescentWithMomentumCpp (double LearningRate, double LearningRatePower, double momentum, const int size, const int rank):OptimiserCpp(size, rank);

	~GradientDescentWithMomentumCpp();
				
};

class AdaGradCpp: public OptimiserCpp {
	
	public:
		
	AdaGradCpp (double LearningRate, double LearningRatePower, const int size, const int rank):OptimiserCpp(size, rank);

	~AdaGradCpp();
				
};


