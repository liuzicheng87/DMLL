class GradientDescentWithMomentumCpp: public OptimiserCpp {
	
	public:
	double LearningRate, LearningRatePower, momentum;
		
	//Initialise the GradientDescent function
	GradientDescentWithMomentumCpp (double LearningRate, double LearningRatePower, double momentum, const int size, const int rank):OptimiserCpp(size, rank) {
	
		//Store all of the input values
		this->LearningRate = LearningRate; 
		this->LearningRatePower = LearningRatePower;
		this->momentum = momentum;
		
		}

	//Destructor		
	~GradientDescentWithMomentumCpp() {}
		
	void max(MPI_Comm comm, const double tol, const int MaxNumIterations);
	void min(MPI_Comm comm, const double tol, const int MaxNumIterations);
		
};

void GradientDescentWithMomentumCpp::max(MPI_Comm comm, const double tol, const int MaxNumIterations) {
		
	int i, IterationNum, BatchNum, BatchBegin, BatchEnd, BatchSize, GlobalBatchSize, CurrentBatchSize, WBatchBegin, WBatchEnd;
	double GlobalBatchSizeDouble;
	
	double CurrentLearningRate, LocalSumGradients;
	
	//WNew is used to test the updates using the Armijo condition
	double *update = (double*)calloc(this->lengthW, sizeof(double));	
	
	for (IterationNum = 0; IterationNum < MaxNumIterations; ++IterationNum) {//IterationNum layer
			
			//Calculate CurrentLearningRate
			CurrentLearningRate = this->LearningRate/pow((double)(IterationNum + 1), this->LearningRatePower);
			
			//this->NumBatches is inherited from the Optimiser class
			for (BatchNum = 0; BatchNum < this->NumBatches; ++BatchNum) {//BatchNum layer
				
				//We must find out our current values for BatchBegin and BatchEnd. We do so by calling this->CalcBatchBeginEnd, which is inherited from the optimiser class.
				this->CalcBatchBeginEnd(BatchBegin, BatchEnd, BatchSize, BatchNum);
	
				//VERY IMPORTANT CONVENTION: When passing this->localdZdw to g(), always set to zero first.
				for (i=0; i<this->lengthW; ++i) this->localdZdW[i] = 0.0;
				MPI_Barrier(comm);				
																				
				//Call g()
				//Note that it is the responsibility of whoever writes the MLalgorithm to make sure that this->dZdW and this->SumdZdW are passed to ALL processes
				//It is, however, your responsibility to place a barrier after that, if required
				this->MLalgorithm->g(comm, this->dZdW, this->localdZdW, this->W, BatchBegin, BatchEnd, BatchSize, BatchNum, IterationNum);
				
				//Add all BatchSize and store the result in GlobalBatchSize
				MPI_Allreduce(&BatchSize, &GlobalBatchSize, 1, MPI_INT, MPI_SUM, comm);		
				GlobalBatchSizeDouble = (double)GlobalBatchSize;	
				
				//Barrier: Wait until all processes have reached this point
				MPI_Barrier(comm);							
				for (i=0; i<this->lengthW; ++i) this->SumdZdW[i] += this->dZdW[i];  		
				
				//Update W (this is only line where max differs from min)
				//Learning rates are always divided by the sample size
				for (i=0; i<this->lengthW; ++i) this->W[i] += update[i] = this->momentum*update[i] + this->dZdW[i]*CurrentLearningRate/GlobalBatchSizeDouble;			
													
			}//BatchNum layer
									
			//The following lines should be left unchanged for all gradient-based-optimisers
			//Calculate LocalSumGradients
			this->MLalgorithm->SumGradients[IterationNum] = 0.0;
			for (i=0; i<this->lengthW; ++i) this->MLalgorithm->SumGradients[IterationNum] += this->SumdZdW[i]*this->SumdZdW[i];
			
			//Set SumdZdW to 0 for next iteration
			for (i=0; i<this->lengthW; ++i) this->SumdZdW[i] = 0.0;
									
			//Check whether convergence condition is met. If yes, break
			if (this->MLalgorithm->SumGradients[IterationNum]/((double)(this->lengthW)) < tol) break;
			
	}//IterationNum layer
		
		//Store number of IterationNums needed
		this->MLalgorithm->IterationsNeeded = IterationNum;

		//Free update
		free(update);		
		
}

void GradientDescentWithMomentumCpp::min(MPI_Comm comm, const double tol, const int MaxNumIterations) {
		
	int i, IterationNum, BatchNum, BatchBegin, BatchEnd, BatchSize, GlobalBatchSize, CurrentBatchSize, WBatchBegin, WBatchEnd;
	double GlobalBatchSizeDouble;
	
	double CurrentLearningRate, LocalSumGradients;
	
	//WNew is used to test the updates using the Armijo condition
	double *update = (double*)calloc(this->lengthW, sizeof(double));	
		
	for (IterationNum = 0; IterationNum < MaxNumIterations; ++IterationNum) {//IterationNum layer
			
			//Calculate CurrentLearningRate
			CurrentLearningRate = this->LearningRate/pow((double)(IterationNum + 1), this->LearningRatePower);
			
			//this->NumBatches is inherited from the Optimiser class
			for (BatchNum = 0; BatchNum < this->NumBatches; ++BatchNum) {//BatchNum layer
				
				//We must find out our current values for BatchBegin and BatchEnd. We do so by calling this->CalcBatchBeginEnd, which is inherited from the optimiser class.
				this->CalcBatchBeginEnd(BatchBegin, BatchEnd, BatchSize, BatchNum);
	
				//VERY IMPORTANT CONVENTION: When passing this->localdZdw to g(), always set to zero first.
				for (i=0; i<this->lengthW; ++i) this->localdZdW[i] = 0.0;
				MPI_Barrier(comm);							
																
				//Call g()
				//Note that it is the responsibility of whoever writes the MLalgorithm to make sure that this->dZdW and this->SumdZdW are passed to ALL processes
				//It is, however, your responsibility to place a barrier after that, if required
				this->MLalgorithm->g(comm, this->dZdW, this->localdZdW, this->W, BatchBegin, BatchEnd, BatchSize, BatchNum, IterationNum);
				
				//Add all BatchSize and store the result in GlobalBatchSize
				MPI_Allreduce(&BatchSize, &GlobalBatchSize, 1, MPI_INT, MPI_SUM, comm);		
				GlobalBatchSizeDouble = (double)GlobalBatchSize;	
				
				//Barrier: Wait until all processes have reached this point
				MPI_Barrier(comm);							
				for (i=0; i<this->lengthW; ++i) this->SumdZdW[i] += this->dZdW[i];  		
								
				//Update W (this is only line where max differs from min)
				//Learning rates are always divided by the sample size
				for (i=0; i<this->lengthW; ++i) this->W[i] += update[i] = this->momentum*update[i] - this->dZdW[i]*CurrentLearningRate/GlobalBatchSizeDouble;			
													
			}//BatchNum layer
									
			//The following lines should be left unchanged for all gradient-based-optimisers
			//Calculate LocalSumGradients
			this->MLalgorithm->SumGradients[IterationNum] = 0.0;
			for (i=0; i<this->lengthW; ++i) this->MLalgorithm->SumGradients[IterationNum] += this->SumdZdW[i]*this->SumdZdW[i];
			
			//Set SumdZdW to 0 for next iteration
			for (i=0; i<this->lengthW; ++i) this->SumdZdW[i] = 0.0;
									
			//Check whether convergence condition is met. If yes, break
			if (this->MLalgorithm->SumGradients[IterationNum]/((double)(this->lengthW)) < tol) break;
			
	}//IterationNum layer
	
		//Store number of IterationNums needed
		this->MLalgorithm->IterationsNeeded = IterationNum;
		
		//Free update
		free(update);		
		
}
