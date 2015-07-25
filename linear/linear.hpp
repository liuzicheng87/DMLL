class LinearRegressionCpp: public NumericallyOptimisedMLAlgorithmCpp {

	public:
	double *X, *Y;
	int J;
	
	OptimiserCpp *optimiser;
	
	RegulariserCpp *regulariser;

	LinearRegressionCpp(int J, RegulariserCpp *regulariser): NumericallyOptimisedMLAlgorithmCpp() {
		
		int i;
		
		//Necessary for pseudo-random number generator
		std::mt19937 gen(1);//Note that we deliberately choose a constant seed to get the same output every time we call the function
		std::normal_distribution<double> dist(0.0, 1.0);//Normal distribution with mean 0 and standard deviation of 1
		
		this->J = J;
		this->lengthW = J+1;
		this->regulariser = regulariser;
				
		//IMPORTANT: You must malloc W and initialise values randomly. How you randomly initialise them is your decision, but make sure you the the same values every time you call the function.
		this->W = (double*)malloc(this->lengthW*sizeof(double));
		for (i=0; i<this->lengthW; ++i) this->W[i] = dist(gen);

	}
	
	~LinearRegressionCpp() {}

	//Z: the value to be optimised	
	//W: weights to be used for this iteration			
	//ThreadBatchBegin: the number of the instance or sample at which this thread is supposed to begin iterating
	//ThreadBatchEnd: the end of the batch assigned to this thread. The thread will iterate from sample number ThreadBatchBegin to sample number ThreadBatchEnd.
	//ThreadNum: Each thread is assigned an individual number ranging from 0 to the number of threads minus one. This is used for identification. You may or may not need this depending on your algorithm.
	void f(MPI_Comm comm, double &Z, const double *W, const int BatchBegin, const int BatchEnd, const int BatchSize, const int BatchNum, const int IterationNum) {
		
		int i,j;
		double localZ=0.0, Yhat;
				
		//Let Z equal 0
		Z = 0.0;
				
		//Calculate results for instances determined by the optimiser
		for (i=BatchBegin; i<BatchEnd; ++i) {
			
			//IMPORTANT: Use W that has been passed to function, not this->W			
			Yhat = 0.0;
			for (j=0; j<this->J; ++j) Yhat += W[j]*this->X[this->J*i + j];
			//Bias is W[this->J]			
			Yhat += W[this->J];
			
			//Calculate localZ
			localZ += (Y[i] - Yhat)*(Y[i] - Yhat);
						
		}
		
		//Apply regulariser
		this->regulariser->f(localZ, W, 0, this->J, this->J, (double)BatchSize); 
						
		//Add all localZ and store the result in Z
		MPI_Allreduce(&localZ, &Z, 1, MPI_DOUBLE, MPI_SUM, comm);
						
	}
	
	//dZdW: derivative of the value to be optimised (contained in optimiser class)
	//localdZdW: local version of the derivative of the value to be optimised (contained in optimiser class)	
	//SumdZdW: sum over all batches in one iteration (needed for the stopping criterion)	
	//W: weights to be used for this iteration				
	//BatchBegin: the number of the instance or sample at which this process is supposed to begin iterating
	//BatchEnd: the end of the batch assigned to this process. The process will iterate from sample number BatchBegin to sample number BatchEnd.
	//BatchSize: the end of the batch assigned to this process. The process will iterate from sample number BatchBegin to sample number BatchEnd.	
	//BatchNum: batch number
	//IterationNum: iteration number
	void g(MPI_Comm comm, double *dZdW, double *localdZdW, const double *W, const int BatchBegin, const int BatchEnd, const int BatchSize, const int BatchNum, const int IterationNum) {
		
		int i,j;
		double Yhat, TwoYhatMinusY;
				
		//Calculate results for instances determined by the optimiser
		for (i=BatchBegin; i<BatchEnd; ++i) {
			
			//IMPORTANT: Use W that has been passed to function, not this->W
			Yhat = 0.0;
			for (j=0; j<this->J; ++j) Yhat += W[j]*this->X[this->J*i + j];
			Yhat += W[this->J];
						
			//Height parameter is W[this->J]
			//dZdW = sum 2*(Yhat - Y[i])*X[j]
			TwoYhatMinusY = 2.0*(Yhat - Y[i]);
			
			for (j=0; j<this->J; ++j) localdZdW[j] += this->X[this->J*i + j]*TwoYhatMinusY;
			localdZdW[this->J] += TwoYhatMinusY;
			
		}
		
		//Apply regulariser
		this->regulariser->g(localdZdW, W, 0, this->J, this->J, (double)BatchSize); 		
							
		//Add all localdZdW and store the result in dZdW
		MPI_Allreduce(localdZdW, this->optimiser->dZdW, this->lengthW, MPI_DOUBLE, MPI_SUM, comm);			
						
	}
	
	//optimiser: pointer to the optimiser used
	//X: training data
	//Y: data to be predicted
	//I: number of instances in dataset
	//J: number of attributes in dataset	
	//lengthW: number of weights
	//BatchSize: size of minibatches for updating
	//tol: the error tolerance
	//MaxNumIterations: the maximum number of iterations tolerated until optimiser stops
	//NumThreads: the number of threads used
	//maximise() and minimise() both run until either the sum of gradients is smaller than tol or the number of iterations reaches MaxNumIterations
	//ThreadNum: Each thread is assigned an individual number ranging from 0 to the number of threads minus one. This is used for identification. 
	void fit (MPI_Comm comm, double *X, int I, int J, double *Y, int IY, OptimiserCpp *optimiser, int GlobalBatchSize=200, const double tol=1e-08, const int MaxNumIterations=1000) {
		
		//Check input values
		if (J != this->J) throw std::invalid_argument("Number of attributes J does not match the J that has been defined when declaring the class!");
		if (I != IY) throw std::invalid_argument("Lengh of Y does not match length of X!");
		
		//Store input values (which we need for f() and g())
		this->optimiser = optimiser;
		this->X = X;
		this->Y = Y;
		this->I = I;
				
		//Optimise
		this->optimiser->minimise(comm, this, I, lengthW, GlobalBatchSize, tol, MaxNumIterations);
				
	}
	
	//Yhat: prediction
	//IY: length of Yhat (should equal I, if not, throw error)
	//X: training data
	//Yhat: prediction
	//I: number of instances in dataset
	//J: number of attributes in dataset	
	void predict (double *Yhat, int IY, double *X, int I, int J) {
		
		int i,j;
		
		//Check input values
		if (J != this->J) throw std::invalid_argument("Number of attributes J does not match the J that has been defined when declaring the class!");
		if (I != IY) throw std::invalid_argument("Lengh of Yhat does not match length of X!");
				
		//Set Yhat to zero		
		for (i=0; i<I; ++i) Yhat[i] = 0.0;		
				
		//Calculate Yhat
		for (i=0; i<I; ++i) {
			for (j=0; j<J; ++j) Yhat[i] += this->W[j]*X[J*i + j];
			Yhat[i] += W[J];
		}
		
		
	}	
	
//	void GetParams(double *W, int lengthW) {
		
//		int i;
		
//		if (lengthW != this->lengthW) throw std::invalid_argument("Length of provided W does not match lengthW!");
		
//		for (i=0; i<lengthW; ++i) W[i] = this->W[i];
		
//	} 
	
	
		
};

