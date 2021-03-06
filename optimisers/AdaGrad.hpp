class AdaGradCpp: public OptimiserCpp {
	
	public:
	double LearningRate, LearningRatePower;
	
	double *SumdZdWSquared;
		
	//Initialise the GradientDescent function
	AdaGradCpp (double LearningRate, double LearningRatePower, const int size, const int rank):OptimiserCpp(size, rank) {
	
		//Store all of the input values
		this->LearningRate = LearningRate; 
		this->LearningRatePower = LearningRatePower;
				
		}

	//Destructor		
	~AdaGradCpp() {}
		
	void max(MPI_Comm comm, const double tol, const int MaxNumIterations);
	void min(MPI_Comm comm, const double tol, const int MaxNumIterations);
		
};

void AdaGradCpp::max(MPI_Comm comm, const double tol, const int MaxNumIterations) {
		
	int i, IterationNum, BatchNum, BatchBegin, BatchEnd, BatchSize, GlobalBatchSize, CurrentBatchSize, WBatchBegin, WBatchEnd;
	double GlobalBatchSizeDouble;
	
	double CurrentLearningRate, LocalSumGradients;
	
	this->SumdZdWSquared = (double*)calloc(this->lengthW, sizeof(double));
	
	for (IterationNum = 0; IterationNum < MaxNumIterations; ++IterationNum) {//IterationNum layer
			
			//Calculate CurrentLearningRate
			CurrentLearningRate = this->LearningRate/pow((double)(IterationNum + 1), this->LearningRatePower);
			
			//this->NumBatches is inherited from the Optimiser class
			for (BatchNum = 0; BatchNum < this->NumBatches; ++BatchNum) {//BatchNum layer
				
				//We must find out our current values for BatchBegin and BatchEnd. We do so by calling this->CalcBatchBeginEnd, which is inherited from the optimiser class.
				this->CalcBatchBeginEnd(BatchBegin, BatchEnd, BatchSize, BatchNum);
	
				//VERY IMPORTANT CONVENTION: When passing this->localdZdw to g(), always set to zero first.
				for (i=0; i<this->lengthW; ++i) this->localdZdW[i] = 0.0;

				//If weight is greater than wMax or smaller than wMin, then clip
				//If there is no wMin or wMax, this will have no effect				
				wClip(this->W);
				
				//Barrier: Wait until all processes have reached this point
				MPI_Barrier(comm);	
																				
				//Call g()
				//Note that it is the responsibility of whoever writes the MLalgorithm to make sure that this->dZdW and this->SumdZdW are passed to ALL processes
				//It is, however, your responsibility to place a barrier after that, if required
				this->MLalgorithm->g(comm, this->dZdW, this->localdZdW, this->W, BatchBegin, BatchEnd, BatchSize, BatchNum, IterationNum);
				
				//Add all BatchSize and store the result in GlobalBatchSize
				MPI_Allreduce(&BatchSize, &GlobalBatchSize, 1, MPI_INT, MPI_SUM, comm);		
				GlobalBatchSizeDouble = (double)GlobalBatchSize;	
				
				//If weight equals wMin (wMax) and dZdW is smaller than zero (greater than zero), set dZdW to zero
				//If there is no wMin or wMax, this will have no effect
				dZdWClipMax();				
				
				//Barrier: Wait until all processes have reached this point
				MPI_Barrier(comm);							
				for (i=0; i<this->lengthW; ++i) this->SumdZdW[i] += this->dZdW[i];  		

				//Calculate "dampening parameter"
				for (i=0; i<this->lengthW; ++i) this->SumdZdWSquared[i] += this->dZdW[i]*this->dZdW[i];  		

				//Update W
				for (i=0; i<this->lengthW; ++i) if (this->SumdZdWSquared[i] > 0.0) this->W[i] += (this->dZdW[i]/sqrt(this->SumdZdWSquared[i]))*CurrentLearningRate;						
													
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
	
		//If weight is greater than wMax or smaller than wMin, then clip
		//If there is no wMin or wMax, this will have no effect				
		wClip(this->W);	
		
		//Store number of IterationNums needed
		this->MLalgorithm->IterationsNeeded = IterationNum;
		
		free(this->SumdZdWSquared);
		
}

void AdaGradCpp::min(MPI_Comm comm, const double tol, const int MaxNumIterations) {
		
	int i, IterationNum, BatchNum, BatchBegin, BatchEnd, BatchSize, GlobalBatchSize, CurrentBatchSize, WBatchBegin, WBatchEnd;
	double GlobalBatchSizeDouble;
	
	double CurrentLearningRate, LocalSumGradients;
	
	this->SumdZdWSquared = (double*)calloc(this->lengthW, sizeof(double));
			
	for (IterationNum = 0; IterationNum < MaxNumIterations; ++IterationNum) {//IterationNum layer
			
			//Calculate CurrentLearningRate
			CurrentLearningRate = this->LearningRate/pow((double)(IterationNum + 1), this->LearningRatePower);
			
			//this->NumBatches is inherited from the Optimiser class
			for (BatchNum = 0; BatchNum < this->NumBatches; ++BatchNum) {//BatchNum layer
				
				//We must find out our current values for BatchBegin and BatchEnd. We do so by calling this->CalcBatchBeginEnd, which is inherited from the optimiser class.
				this->CalcBatchBeginEnd(BatchBegin, BatchEnd, BatchSize, BatchNum);
	
				//VERY IMPORTANT CONVENTION: When passing this->localdZdw to g(), always set to zero first.
				for (i=0; i<this->lengthW; ++i) this->localdZdW[i] = 0.0;

				//If weight is greater than wMax or smaller than wMin, then clip
				//If there is no wMin or wMax, this will have no effect				
				wClip(this->W);
				
				//Barrier: Wait until all processes have reached this point
				MPI_Barrier(comm);	
																
				//Call g()
				//Note that it is the responsibility of whoever writes the MLalgorithm to make sure that this->dZdW and this->SumdZdW are passed to ALL processes
				//It is, however, your responsibility to place a barrier after that, if required
				this->MLalgorithm->g(comm, this->dZdW, this->localdZdW, this->W, BatchBegin, BatchEnd, BatchSize, BatchNum, IterationNum);
				
				//Add all BatchSize and store the result in GlobalBatchSize
				MPI_Allreduce(&BatchSize, &GlobalBatchSize, 1, MPI_INT, MPI_SUM, comm);		
				GlobalBatchSizeDouble = (double)GlobalBatchSize;	
				
				//If weight equals wMin (wMax) and dZdW is smaller than zero (greater than zero), set dZdW to zero
				//If there is no wMin or wMax, this will have no effect
				dZdWClipMin();
								
				//Barrier: Wait until all processes have reached this point
				MPI_Barrier(comm);							
				for (i=0; i<this->lengthW; ++i) this->SumdZdW[i] += this->dZdW[i];  	
				
				//Calculate "dampening parameter"
				for (i=0; i<this->lengthW; ++i) this->SumdZdWSquared[i] += this->dZdW[i]*this->dZdW[i];  		
					
				//Update W
				for (i=0; i<this->lengthW; ++i) if (this->SumdZdWSquared[i] > 0.0) this->W[i] -= (this->dZdW[i]/sqrt(this->SumdZdWSquared[i]))*CurrentLearningRate;						
								
													
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
		
		//If weight is greater than wMax or smaller than wMin, then clip
		//If there is no wMin or wMax, this will have no effect				
		wClip(this->W);			
		
		//Store number of IterationNums needed
		this->MLalgorithm->IterationsNeeded = IterationNum;
		
		free(this->SumdZdWSquared);
				
}
