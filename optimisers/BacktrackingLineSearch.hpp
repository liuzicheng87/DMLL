class BacktrackingLineSearchCpp: public OptimiserCpp {
	
	public:
	double LearningRateStart, LearningRateReduction, c, tol;
		
	//Initialise the GradientDescent function
	BacktrackingLineSearchCpp (double LearningRateStart, double LearningRateReduction, double c,  double tol, const int size, const int rank):OptimiserCpp(size, rank) {
	
		//Store all of the input values
		this->LearningRateStart = LearningRateStart; 
		this->LearningRateReduction = LearningRateReduction;
		this->c = c;
		this->tol = tol;
		
		}

	//Destructor		
	~BacktrackingLineSearchCpp() {}
		
	void max(MPI_Comm comm, const double tol, const int MaxNumIterations);
	void min(MPI_Comm comm, const double tol, const int MaxNumIterations);
		
};

void BacktrackingLineSearchCpp::max(MPI_Comm comm, const double tol, const int MaxNumIterations) {
		
	int i, IterationNum, BatchNum, BatchBegin, BatchEnd, BatchSize, GlobalBatchSize, CurrentBatchSize, WBatchBegin, WBatchEnd;
	double GlobalBatchSizeDouble;
	
	double CurrentLearningRate, LocalSumGradients, Z, ZNew, LocalSlope;
		
	//WNew is used to test the updates using the Armijo condition
	double *WNew = (double*)calloc(this->lengthW, sizeof(double));
	
	//clip W, if necessary
	// if there is no wMin of wMax, this will have no effect
	wClip(this->W);	
		
	//Calculate Z once, so it can be used in the first iteration
	this->CalcBatchBeginEnd(BatchBegin, BatchEnd, BatchSize, 0);
	this->MLalgorithm->f(comm, Z, this->W, BatchBegin, BatchEnd, BatchSize, BatchNum, IterationNum);

	for (IterationNum = 0; IterationNum < MaxNumIterations; ++IterationNum) {//IterationNum layer
						
			//this->NumBatches is inherited from the Optimiser class
			for (BatchNum = 0; BatchNum < this->NumBatches; ++BatchNum) {//BatchNum layer
				
				//We must find out our current values for BatchBegin and BatchEnd. We do so by calling this->CalcBatchBeginEnd, which is inherited from the optimiser class.
				this->CalcBatchBeginEnd(BatchBegin, BatchEnd, BatchSize, BatchNum);
	
				//VERY IMPORTANT CONVENTION: When passing this->localdZdw to g(), always set to zero first.
				for (i=0; i<this->lengthW; ++i) this->localdZdW[i] = 0.0;
				
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
				
				//Calculate LocalSlope
				LocalSlope = 0.0;				
				for (i=0; i<this->lengthW; ++i) LocalSlope += this->dZdW[i]*this->dZdW[i];  		
								
				//Find the optimal learning rate using backtracking and the Armijo condition
				CurrentLearningRate = this->LearningRateStart;
				while(true) {
					
					//Calculate WNew
					for (i=0; i<this->lengthW; ++i) WNew[i] = this->W[i] + this->dZdW[i]*CurrentLearningRate/GlobalBatchSizeDouble;		
					
					//clip WNew, if necessary
					// if there is no wMin of wMax, this will have no effect
					wClip(WNew);								
					
					//Calculate ZNew based on WNew
					this->MLalgorithm->f(comm, ZNew, WNew, BatchBegin, BatchEnd, BatchSize, BatchNum, IterationNum);
																	
					//Test whether Armijo condition is fulfilled. If yes, break					
					if (ZNew > Z - this->tol + this->c*(CurrentLearningRate/GlobalBatchSizeDouble)*LocalSlope) break;
					
					//Reduce CurrentLearningRate
					CurrentLearningRate *= this->LearningRateReduction;
					
					//Barrier: Wait until all processes have reached this point
					MPI_Barrier(comm);						
				}
				
				//Update W
				//Learning rates are always divided by the sample size				
				for (i=0; i<this->lengthW; ++i) this->W[i] = WNew[i];
				Z = ZNew;
									
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
		
		//Free WNew
		free(WNew);
		
}


void BacktrackingLineSearchCpp::min(MPI_Comm comm, const double tol, const int MaxNumIterations) {
		
	int i, IterationNum, BatchNum, BatchBegin, BatchEnd, BatchSize, GlobalBatchSize, CurrentBatchSize, WBatchBegin, WBatchEnd;
	double GlobalBatchSizeDouble;
	
	double CurrentLearningRate, LocalSumGradients, Z, ZNew, LocalSlope;
		
	//WNew is used to test the updates using the Armijo condition
	double *WNew = (double*)calloc(this->lengthW, sizeof(double));
	
	//clip W, if necessary
	// if there is no wMin of wMax, this will have no effect
	wClip(this->W);		
		
	//Calculate Z once, so it can be used in the first iteration
	this->CalcBatchBeginEnd(BatchBegin, BatchEnd, BatchSize, 0);
	this->MLalgorithm->f(comm, Z, this->W, BatchBegin, BatchEnd, BatchSize, BatchNum, IterationNum);

	for (IterationNum = 0; IterationNum < MaxNumIterations; ++IterationNum) {//IterationNum layer
						
			//this->NumBatches is inherited from the Optimiser class
			for (BatchNum = 0; BatchNum < this->NumBatches; ++BatchNum) {//BatchNum layer
				
				//We must find out our current values for BatchBegin and BatchEnd. We do so by calling this->CalcBatchBeginEnd, which is inherited from the optimiser class.
				this->CalcBatchBeginEnd(BatchBegin, BatchEnd, BatchSize, BatchNum);
	
				//VERY IMPORTANT CONVENTION: When passing this->localdZdw to g(), always set to zero first.
				for (i=0; i<this->lengthW; ++i) this->localdZdW[i] = 0.0;
				
				//Barrier: Wait until all processes have reached this point
				MPI_Barrier(comm);	
																
				//Call g()
				//Note that it is the responsibility of whoever writes the MLalgorithm to make sure that this->dZdW and this->SumdZdW are passed to ALL processes
				//It is, however, your responsibility to place a barrier after that, if required
				this->MLalgorithm->g(comm, this->dZdW, this->localdZdW, this->W, BatchBegin, BatchEnd, BatchSize, BatchNum, IterationNum);
				
				//Add all BatchSize and store the result in GlobalBatchSize
				MPI_Allreduce(&BatchSize, &GlobalBatchSize, 1, MPI_INT, MPI_SUM, comm);		
				GlobalBatchSizeDouble = (double)GlobalBatchSize;	
				
				//If weight equals wMin (wMax) and dZdW is greater than zero (smaller than zero), set dZdW to zero
				//If there is no wMin or wMax, this will have no effect
				dZdWClipMin();						
				
				//Barrier: Wait until all processes have reached this point
				MPI_Barrier(comm);	
				for (i=0; i<this->lengthW; ++i) this->SumdZdW[i] += this->dZdW[i];  		
				
				//Calculate LocalSlope
				LocalSlope = 0.0;				
				for (i=0; i<this->lengthW; ++i) LocalSlope -= this->dZdW[i]*this->dZdW[i];  		
								
				//Find the optimal learning rate using backtracking and the Armijo condition
				CurrentLearningRate = this->LearningRateStart;
				while(true) {
					
					//Calculate WNew
					for (i=0; i<this->lengthW; ++i) WNew[i] = this->W[i] - this->dZdW[i]*CurrentLearningRate/GlobalBatchSizeDouble;			
					
					//clip WNew, if necessary
					// if there is no wMin of wMax, this will have no effect
					wClip(WNew);
		
					//Calculate ZNew based on WNew
					this->MLalgorithm->f(comm, ZNew, WNew, BatchBegin, BatchEnd, BatchSize, BatchNum, IterationNum);
																	
					//Test whether Armijo condition is fulfilled. If yes, break					
					if (ZNew < Z + this->tol + this->c*(CurrentLearningRate/GlobalBatchSizeDouble)*LocalSlope) break;
					
					//Reduce CurrentLearningRate
					CurrentLearningRate *= this->LearningRateReduction;
					
					//Barrier: Wait until all processes have reached this point
					MPI_Barrier(comm);						
				}
				
				//Update W
				//Learning rates are always divided by the sample size				
				for (i=0; i<this->lengthW; ++i) this->W[i] = WNew[i];
				Z = ZNew;
									
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
		
		//Free WNew
		free(WNew);
		
}
