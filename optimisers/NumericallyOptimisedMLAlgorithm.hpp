class NumericallyOptimisedMLAlgorithmCpp {

	public:
	double *W, *SumGradients, *wMax, *wMin;
	int I, GlobalI, lengthW, IterationsNeeded, *wMaxIndices, *wMinIndices, wMaxLength, wMinLength;
						
	NumericallyOptimisedMLAlgorithmCpp() {
		this->W = NULL;
		this->SumGradients = NULL;
		this->IterationsNeeded = 0;
		
		//These variables are only relevant when we want to impose
		this->wMax = NULL;
		this->wMin = NULL;
		this->wMaxLength = 0; //if there is no maximum imposed on W, we need to know that wMaxLength is zero
		this->wMinLength = 0;//if there is no minimum imposed on W, we need to know that wMinLength is zero
		}
	
	//Virtual destructor		
	virtual ~NumericallyOptimisedMLAlgorithmCpp() {
				
		//Free W and SumGradients
		if (this->W != NULL) free(this->W);		
		if (this->SumGradients != NULL) free(this->SumGradients);		
		if (this->wMax != NULL) {free(wMax); free(wMaxIndices);}
		if (this->wMin != NULL) {free(wMin); free(wMinIndices);}

		}

	//Z: the value to be optimised	
	//W: weights to be used for this iteration			
	//dZdW: derivative of the value to be optimised (contained in optimiser class)
	//localdZdW: local version of the derivative of the value to be optimised (contained in optimiser class)	
	//SumdZdW: sum over all batches in one iteration (needed for the stopping criterion)	
	//BatchBegin: the number of the instance or sample at which this process is supposed to begin iterating
	//BatchEnd: the end of the batch assigned to this process. The process will iterate from sample number BatchBegin to sample number BatchEnd.
	//BatchSize: the end of the batch assigned to this process. The process will iterate from sample number BatchBegin to sample number BatchEnd.	
	//rank: Each process is assigned an individual number ranging from 0 to the number of processs minus one. This is used for identification. You may or may not need this depending on your algorithm.
	//size: number of processes
	//BatchNum: batch number
	//IterationNum: iteration number
	virtual void f(MPI_Comm comm, double &Z, const double *W, const int BatchBegin, const int BatchEnd, const int BatchSize, const int BatchNum, const int IterationNum) {}
	virtual void g(MPI_Comm comm, double *dZdW, double *localdZdW, const double *W, const int BatchBegin, const int BatchEnd, const int BatchSize, const int BatchNum, const int IterationNum) {}
			
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
