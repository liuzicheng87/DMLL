class NumericallyOptimisedMLAlgorithmCpp {

	public:
	double *W, *SumGradients;
	int I, GlobalI, lengthW, IterationsNeeded;
						
	NumericallyOptimisedMLAlgorithmCpp() {
		this->W = NULL;
		this->SumGradients = NULL;
		this->IterationsNeeded = 0;
		}
	
	//Virtual destructor		
	virtual ~NumericallyOptimisedMLAlgorithmCpp() {
				
		//Free W and SumGradients
		if (this->W != NULL) free(this->W);		
		if (this->SumGradients != NULL) free(this->SumGradients);		
		}

	//Z: the value to be optimised	
	//W: weights to be used for this iteration			
	//dZdW: the number attributes or features		
	//SumdZdW: sum over all batches in one iteration (needed for the stopping criterion)	
	//BatchBegin: the number of the instance or sample at which this process is supposed to begin iterating
	//BatchEnd: the end of the batch assigned to this process. The process will iterate from sample number BatchBegin to sample number BatchEnd.
	//BatchSize: the end of the batch assigned to this process. The process will iterate from sample number BatchBegin to sample number BatchEnd.	
	//rank: Each process is assigned an individual number ranging from 0 to the number of processs minus one. This is used for identification. You may or may not need this depending on your algorithm.
	//size: number of processes
	//BatchNum: batch number
	//IterationNum: iteration number
	virtual void f(MPI_Comm comm, double &Z, const double *W, const int BatchBegin, const int BatchEnd, const int BatchSize, const int BatchNum, const int IterationNum) {}
	virtual void g(MPI_Comm comm, const double *W, const int BatchBegin, const int BatchEnd, const int BatchSize, const int BatchNum, const int IterationNum) {}
			
	//This is used to determine the size of the array needed
	int GetIterationsNeeded() {return this->IterationsNeeded;}
		
	//This function is used to pass the gradients
	void GetSumGradients (double *SumGradients, int IterationsNeeded) {
	
		int i;
		if (this->SumGradients == NULL) throw std::invalid_argument("Algorithm not fitted!");
		if (this->IterationsNeeded != IterationsNeeded) throw std::invalid_argument("Length does not match number of iterations needed!");		
		for (i=0; i<this->IterationsNeeded; ++i) SumGradients[i] = this->SumGradients[i];
	
	}
	
	//SetParams sets the weights
	void SetParams(double *W, int lengthW) {
		
		int i;
		
		if (lengthW != this->lengthW) throw std::invalid_argument("Length of provided W does not match lenghtW!");
		
		for (i=0; i<lengthW; ++i) this->W[i] = W[i];
		
	} 	
	
	//GetParams gets the weights
	void GetParams(double *W, int lengthW) {
		
		int i;
		
		if (lengthW != this->lengthW) throw std::invalid_argument("Length of provided W does not match lengthW!");
		
		for (i=0; i<lengthW; ++i) W[i] = this->W[i];
		
	} 		
	
	int GetLengthW() {return this->lengthW;} 			
	
};
