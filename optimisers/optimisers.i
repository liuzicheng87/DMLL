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

class GradientDescent: public OptimiserCpp {
	
	public:
		
	GradientDescent (double LearningRate, double LearningRatePower, const int size, const int rank):OptimiserCpp(size, rank);

	~GradientDescent();
				
};

