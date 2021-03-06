class LinearMahaFeatExtSparseCpp: public NumericallyOptimisedMLAlgorithmCpp {
	
	public:	
	//Variables and pointers
	const double *XData, *Y;
	const int *XIndices, *XIndptr;	
	int J, Jext, WBatchBegin, WBatchEnd, NumBatches;
	int *WBatchSize, *CumulativeWBatchSize;

	OptimiserCpp *optimiser;
		
	//Variables that do not depend on W and therefore need to be calculated only once
	double *sumY, *sumYY;
	double *LocalsumY, *LocalsumYY;	
	double  **sumdXextdW, **sumdXextdWY;
	double  **LocalsumdXextdW, **LocalsumdXextdWY;
	
	//Variables that need to be recalculated in every iteration
	double  *sumXext, *sumXextY, *sumXextXext,  **sumdXextdWXext;
	double  *LocalsumXext, *LocalsumXextY, *LocalsumXextXext,  **LocalsumdXextdWXext;
	double *WTW; //gram matrix for, which is necessary for our particular optimisation strategy
	
	//Matrices
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> V, VInv, ZEZ, dVdW, dZEZdW;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> eigensolver;
	Eigen::Matrix<double, 1, 1> chi, gradient;	
		
	//dXextdWData, dXextdWIndices and dXextdWIndptr store the local versions of dXextdW in sparse format
	double **dXextdWData;
	int **dXextdWIndices;
	int **dXextdWIndptr;	
		
	LinearMahaFeatExtSparseCpp (const int J, const int Jext): NumericallyOptimisedMLAlgorithmCpp()  {
		
		int i;
		
		//Necessary for pseudo-random number generator
		std::mt19937 gen(1);//Note that we deliberately choose a constant seed to get the same output every time we call the function
		std::normal_distribution<double> dist(0.0, 1.0);//Normal distribution with mean 0 and standard deviation of 1
		
		//Store input values
		this->J = J; 
		this->Jext = Jext;
		
		//IMPORTANT: this->lengthW needs to be defined when initialising an algorithm!
		this->lengthW = J*Jext + 1; 		
				
		//IMPORTANT: You must malloc W and initialise values randomly. How you randomly initialise them is your decision, but make sure you the the same values every time you call the function.
		this->W = (double*)malloc(this->lengthW*sizeof(double));
		for (i=0; i<J*Jext; ++i) this->W[i] = dist(gen);
		this->W[this->lengthW - 1] = -0.5;
		
		//Resize matrices		
		this->V.resize(this->Jext, this->Jext);
		this->VInv.resize(this->Jext, this->Jext);		
		this->ZEZ.resize(this->Jext, 1);
		this->dVdW.resize(this->Jext, this->Jext);
		this->dZEZdW.resize(1, this->Jext);
		
		//Impose restriction on lambda (for most optimisers this actually not necessary, because the derivative for lambda is always going to be greater or equal to zero, but do so anyway, just in case)
		this->wMaxLength = 1;
		this->wMaxIndices = (int*)malloc(1*sizeof(int));
		this->wMaxIndices[0] = this->lengthW - 1;
			
		this->wMax = (double*)calloc(1, sizeof(double));//wMax is set to zero
		
	};
	
	~LinearMahaFeatExtSparseCpp() {};
						
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
	void f(MPI_Comm comm, double &Z, const double *W, const int BatchBegin, const int BatchEnd, const int BatchSize, const int BatchNum, const int IterationNum) {	

			const double *y;
			int i,j,k,a,c,d;
			double BatchSizeDouble = (double)BatchSize;
			double GlobalBatchSizeDouble;
																																											
			//The size of Xext depends on BatchSize. Therefore, it needs to be initalised within an iteration																																				
			double *Xext = (double*)calloc(this->Jext*BatchSize, sizeof(double));
				
			//Declare a pointer that points to the part of this->Y we are interested in
			y = this->Y + BatchBegin;		
																																		
			calcXext(Xext, W, BatchBegin, BatchSize); //Calculate Xext										
			reduce5(Xext, BatchSize); //Calculate LocalsumXext
			reduce6(Xext, y, BatchSize); //Calculate LocalsumXextY							
			reduce7(Xext, BatchSize); //Calculate LocalsumXextXext
									
			free(Xext);		
			
			//Execute AllReduce operations
			MPI_Allreduce(&BatchSizeDouble, &GlobalBatchSizeDouble, 1, MPI_DOUBLE, MPI_SUM, comm); //Add all BatchSizeDouble and store the result in GlobalBatchSizeDouble				
			MPI_Allreduce(this->LocalsumXext, this->sumXext, this->Jext, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumXext and store the result in sumXext	
			MPI_Allreduce(this->LocalsumXextY, this->sumXextY, this->Jext, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumXextY and store the result in sumXextY	
			MPI_Allreduce(this->LocalsumXextXext, this->sumXextXext, (this->Jext*(this->Jext + 1))/2, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumXextXext and store the result in sumXextXext		

			//Barrier: Wait until all threads have reached this point
			MPI_Barrier(comm);							
																 
			//Calculate Z - E(Z)
			for(a=0; a<Jext; ++a) this->ZEZ(a, 0) = this->sumXextY[a] - this->sumXext[a]*this->sumY[BatchNum]/GlobalBatchSizeDouble;
																							
			//Calculate V							
			for (i=0; i<Jext; ++i) for (k=0; k<=i; ++k)
				this->V(i, k) = this->V(k, i) =
					calcvar1(sumXext[i], sumXext[k], sumXextXext[i*(i+1)/2 + k], sumY[BatchNum], sumYY[BatchNum], GlobalBatchSizeDouble); 
																																																			
				//Calculate V_inv			
			this->VInv = this->V.inverse(); 
			
			//Calculate Z
			chi = this->ZEZ.transpose()*this->VInv*this->ZEZ;
			Z = chi(0,0);			
								
			//Barrier: Wait until all threads have reached this point
			//It the the responsibility of every ML algorithm to pass the complete dZdW to the optimiser
			MPI_Barrier(comm);				
		
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
	void g(MPI_Comm comm, double *dZdW, double *localdZdW, const double *W, const int BatchBegin, const int BatchEnd, const int BatchSize, const int BatchNum, const int IterationNum) {	
								
			const double *y;
			int i,j,k,a,c,d;
			double BatchSizeDouble = (double)BatchSize;
			double GlobalBatchSizeDouble;
			
			//Calculate WTW. As the name implies WTW is the gram matrix for W.
			for (i=0; i<this->Jext*(this->Jext + 1)/2; ++i) this->WTW[i] = 0.0;
			for (i=0; i<this->Jext; ++i) for (j=0; j<=i; ++j) for (k=0; k<this->J; ++k) this->WTW[i*(i+1)/2 + j] += W[i*this->J + k]*W[j*this->J + k];
			for (i=0; i<this->Jext; ++i) this->WTW[i*(i+1)/2 + i] -= 1.0; //We substract 1 from the diagonal, because this is the condition we impose on WTW.
																																											
			//The size of Xext depends on BatchSize. Therefore, it needs to be initalised within an iteration																																				
			double *Xext = (double*)calloc(this->Jext*BatchSize, sizeof(double));
				
			//Declare a pointer that points to the part of this->Y we are interested in
			y = this->Y + BatchBegin;		
																																		
			calcXext(Xext, W, BatchBegin, BatchSize); //Calculate Xext										
			reduce5(Xext, BatchSize); //Calculate LocalsumXext
			reduce6(Xext, y, BatchSize); //Calculate LocalsumXextY							
			reduce7(Xext, BatchSize); //Calculate LocalsumXextXext
			reduce8(Xext,  BatchSize, BatchNum); //Calculate LocalsumdXextdWXext
									
			free(Xext);		
			
			//Execute AllReduce operations
			MPI_Allreduce(&BatchSizeDouble, &GlobalBatchSizeDouble, 1, MPI_DOUBLE, MPI_SUM, comm); //Add all BatchSizeDouble and store the result in GlobalBatchSizeDouble				
			MPI_Allreduce(this->LocalsumXext, this->sumXext, this->Jext, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumXext and store the result in sumXext	
			MPI_Allreduce(this->LocalsumXextY, this->sumXextY, this->Jext, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumXextY and store the result in sumXextY	
			MPI_Allreduce(this->LocalsumXextXext, this->sumXextXext, (this->Jext*(this->Jext + 1))/2, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumXextXext and store the result in sumXextXext		
			for (i=0; i<this->Jext; ++i) MPI_Allreduce(this->LocalsumdXextdWXext[i], this->sumdXextdWXext[i], this->J, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumXextXext and store the result in sumXextXext		

			//Barrier: Wait until all threads have reached this point
			MPI_Barrier(comm);							
																 
			//Calculate Z - E(Z)
			for(a=0; a<Jext; ++a) this->ZEZ(a, 0) = this->sumXextY[a] - this->sumXext[a]*this->sumY[BatchNum]/GlobalBatchSizeDouble;
																							
			//Calculate V							
			for (i=0; i<Jext; ++i) for (k=0; k<=i; ++k)
				this->V(i, k) = this->V(k, i) =
					calcvar1(sumXext[i], sumXext[k], sumXextXext[i*(i+1)/2 + k], sumY[BatchNum], sumYY[BatchNum], GlobalBatchSizeDouble); 
																																																			
				//Calculate V_inv			
			this->VInv = this->V.inverse(); 
							
			//Update W
			//this->WBatchBegin and this->WBatchEnd are declared in fit. They define the batches of W assigned to this process. 
			for (i=this->WBatchBegin; i < this->WBatchEnd; ++i) if (i < this->Jext*this->J) {//Calculate derivative for normal weights
										
				//Calculate c and d
				//c=0,1,...,J-1 is the index for the original feature
				//d=0,1,...,Jext-1 is the index for the extracted feature	
				c = i%(this->J);
				d = i/(this->J);
				
				//If there is no data anyway, continue
				if (dXextdWIndptr[BatchNum][c+1] == dXextdWIndptr[BatchNum][c]) continue;
								
				//Calculate dZEZdw	
				dZEZdW.setZero();
				dZEZdW(0, d) = sumdXextdWY[BatchNum][c] - sumdXextdW[BatchNum][c]*sumY[BatchNum]/GlobalBatchSizeDouble;
																													
				//Calculate dVdw						
				dVdW.setZero();
												
				for (a=0; a<d; ++a) this->dVdW(d, a) = this->dVdW(a, d) =
					calcdvardw1(sumdXextdW[BatchNum][c], sumXext[a], sumdXextdWXext[a][c], sumY[BatchNum], sumYY[BatchNum], GlobalBatchSizeDouble);
														
				this->dVdW(d, d)	= 2.0*calcdvardw1(sumdXextdW[BatchNum][c], sumXext[d], sumdXextdWXext[d][c], sumY[BatchNum], sumYY[BatchNum], GlobalBatchSizeDouble);
																						
				for (a=d+1; a<this->Jext; ++a) this->dVdW(d, a) = this->dVdW(a, d) = 
					calcdvardw1(sumdXextdW[BatchNum][c], sumXext[a], sumdXextdWXext[a][c], sumY[BatchNum], sumYY[BatchNum], GlobalBatchSizeDouble);
																																															
				//Calculate the gradient
				gradient = 2.0*dZEZdW*VInv*ZEZ - ZEZ.transpose()*VInv*dVdW*VInv*ZEZ;
													
				localdZdW[i] = gradient(0,0);
				
				//Add LaGrange multiplier. Note that lambda = W[this->lengthW - 1]
				for (a=0; a<d; ++a) localdZdW[i] += W[this->lengthW - 1]*this->WTW[d*(d + 1)/2 + a]*W[a*this->J + c];
														
				localdZdW[i] += 2.0*W[this->lengthW - 1]*this->WTW[d*(d + 1)/2 + d]*W[d*this->J + c]; //Note that we have already subtracted 1.0 for all diagonal elements in the gram matrix
																						
				for (a=d+1; a<this->Jext; ++a) localdZdW[i] += W[this->lengthW - 1]*this->WTW[a*(a + 1)/2 + d]*W[a*this->J + c];
																
			} else {//Calculate derivative for lambda
				
				for (j=0; j<this->Jext*(this->Jext + 1)/2; ++j)  localdZdW[i] -=  0.5*this->WTW[j]*this->WTW[j]; //Since we are minimising for lambda, rather than maximising, we intentionally multiply the derivative with -1.
				
			}			
			
			//Apply regularisation
			//Regulariser does not apply to LaGrange parameters!
			//this->regulariser->g(localdZdW, W, this->WBatchBegin, min(this->WBatchEnd, Jext*J), min(this->WBatchEnd, Jext*J) - this->WBatchBegin, GlobalBatchSizeDouble);
						
			//Gather all localdZdW and store the result in dZdW
			MPI_Allgatherv(localdZdW + this->WBatchBegin, this->WBatchSize[this->optimiser->rank], MPI_DOUBLE, dZdW, this->WBatchSize, this->CumulativeWBatchSize, MPI_DOUBLE, comm);				
			
			//Barrier: Wait until all threads have reached this point
			//It the the responsibility of every ML algorithm to pass the complete dZdW to the optimiser
			MPI_Barrier(comm);						
					
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
	void fit (MPI_Comm comm, int rank, int size, double *XData, int XDataLength,  int *XIndices, int XIndicesLength,  int *XIndptr, int XIndptrLength,  double *Y,  int IY, const int J, OptimiserCpp *optimiser, int GlobalBatchSize, const double tol, const int MaxNumIterations) {

		//Make sure there are no invalid arguments
		if (J != this->J) throw std::invalid_argument("Number of attributes J does not match the J that has been defined when declaring the class!");				
		if (Jext != this->Jext) throw std::invalid_argument("Number of attributes Jext does not match the Jext that has been defined when declaring the class!");		
		if (XDataLength != XIndicesLength) throw std::invalid_argument("Length of XData does not match length of XIndices!");		
				
		int i, j, BatchNum, BatchBegin, BatchEnd, BatchSize, SparseDataSize;
						
		//Store input values (which we need for f() and g())
		this->optimiser = optimiser;
		this->XData = XData;
		this->XIndices = XIndices;
		this->XIndptr = XIndptr;		
		this->Y = Y;
		this->I = XIndptrLength - 1;//This relationship follows from the structure of a CSR_matrix
		
		//We must find the batches of W assigned to this process. We do so by calling CalcLocalICpp, which is contained in UsefulFunctions.
		this->WBatchSize = (int*)calloc(size, sizeof(double));
		this->CumulativeWBatchSize = (int*)calloc(size, sizeof(double));
		
		CalcLocalICpp (this->WBatchSize, size, this->CumulativeWBatchSize, size, this->lengthW);
		this->WBatchBegin = this->CumulativeWBatchSize[rank];
		this->WBatchEnd = this->CumulativeWBatchSize[rank] + this->WBatchSize[rank];
		
		//Calculate the number of batches
		//This function is inherited from the optimiser class
		optimiser->CalcNumBatches (comm, this->I, GlobalBatchSize, this->NumBatches);				
				
		//Initialise variables that DO NOT depend on W
		this->sumY = (double*)calloc(this->NumBatches, sizeof(double));
		this->sumYY = (double*)calloc(this->NumBatches, sizeof(double));
		
		this->LocalsumY = (double*)calloc(this->NumBatches, sizeof(double));
		this->LocalsumYY = (double*)calloc(this->NumBatches, sizeof(double));		

		this->sumdXextdW= (double**)malloc(this->NumBatches*sizeof(double*));
		this->sumdXextdWY = (double**)malloc(this->NumBatches*sizeof(double*));		

		for (BatchNum=0; BatchNum<this->NumBatches; ++BatchNum) this->sumdXextdW[BatchNum] = (double*)calloc(J, sizeof(double));	
		for (BatchNum=0; BatchNum<this->NumBatches; ++BatchNum) this->sumdXextdWY[BatchNum] = (double*)calloc(J, sizeof(double));		

		this->LocalsumdXextdW= (double**)malloc(this->NumBatches*sizeof(double*));
		this->LocalsumdXextdWY = (double**)malloc(this->NumBatches*sizeof(double*));		
						
		for (BatchNum=0; BatchNum<this->NumBatches; ++BatchNum) this->LocalsumdXextdW[BatchNum] = (double*)calloc(J, sizeof(double));	
		for (BatchNum=0; BatchNum<this->NumBatches; ++BatchNum) this->LocalsumdXextdWY[BatchNum] = (double*)calloc(J, sizeof(double));		
												
		//Initialise variables that DO depend on W
		this->sumXext = (double*)calloc(Jext, sizeof(double));
		this->sumXextY = (double*)calloc(Jext, sizeof(double));		
		this->sumXextXext = (double*)calloc((Jext*(Jext + 1))/2, sizeof(double));

		this->LocalsumXext = (double*)calloc(Jext, sizeof(double));
		this->LocalsumXextY = (double*)calloc(Jext, sizeof(double));		
		this->LocalsumXextXext = (double*)calloc((Jext*(Jext + 1))/2, sizeof(double));
		
		sumdXextdWXext = (double**)malloc(Jext*sizeof(double*));
		for (i=0; i<this->Jext; ++i) this->sumdXextdWXext[i] = (double*)calloc(this->J, sizeof(double));	
		
		LocalsumdXextdWXext = (double**)malloc(Jext*sizeof(double*));
		for (i=0; i<this->Jext; ++i) this->LocalsumdXextdWXext[i] = (double*)calloc(this->J, sizeof(double));	
				
		this->dXextdWData = (double**)malloc(this->NumBatches*sizeof(double*));
		this->dXextdWIndices  = (int**)malloc(this->NumBatches*sizeof(int*));
		this->dXextdWIndptr  = (int**)malloc(this->NumBatches*sizeof(int*));	
																		
		this->WTW = (double*)calloc((Jext*(Jext + 1))/2, sizeof(double));
																		
		//Calculate variables that DO NOT depend on W. Since they, by defintion, will not change as we optimise W, we calculate them before conducting the actual optimisation.
		for (BatchNum=0; BatchNum<this->NumBatches; ++BatchNum) {
								
			//We must find out our current values for BatchBegin and BatchEnd. We do so by calling this->CalcBatchBeginEnd, which is inherited from the optimiser class.
			this->optimiser->CalcBatchBeginEnd(BatchBegin, BatchEnd, BatchSize, BatchNum, this->I, this->NumBatches);
								
			//sumY and sumYY are part of the sufficient statistics, but obviously do not depend on W
			reduce1(BatchBegin, BatchEnd, BatchNum);
			reduce2(BatchBegin, BatchEnd, BatchNum);
						
			//We do not know the size of the vectors  this->dXextdWData and this->dXextdWIndices beforehand, so we cannot allocate them earlier
			SparseDataSize = this->XIndptr[BatchEnd] - this->XIndptr[BatchBegin];
			
			this->dXextdWData[BatchNum] = (double*)calloc(SparseDataSize, sizeof(double));
			this->dXextdWIndices[BatchNum] = (int*)calloc(SparseDataSize, sizeof(int));
			this->dXextdWIndptr[BatchNum] = (int*)calloc(J+1, sizeof(int));		
			
			//calcdXextdW() calculates dXextdWData, dXextdWIndptr and dXextdWIndptr for the respective process and batch				
			calcdXextdW(BatchBegin, BatchSize, BatchEnd, BatchNum);
							
			//reduce3() calculates sumdXextdW for the respective process and batch				
			reduce3(BatchNum);		
				
			//reduce5() calculates sumdXextdWY for the respective process and batch															
			reduce4(BatchBegin, BatchNum);	
									
		}			
		
		//Reduce variables
		MPI_Allreduce(this->LocalsumY, this->sumY, this->NumBatches, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumY and store the result in sumY	
		MPI_Allreduce(this->LocalsumYY, this->sumYY, this->NumBatches, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumYY and store the result in sumYY	
		for (BatchNum=0; BatchNum<this->NumBatches; ++BatchNum) MPI_Allreduce(this->LocalsumdXextdW[BatchNum], this->sumdXextdW[BatchNum], J, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumdXextdW and store the result in sumdXextdW	
		for (BatchNum=0; BatchNum<this->NumBatches; ++BatchNum) MPI_Allreduce(this->LocalsumdXextdWY[BatchNum], this->sumdXextdWY[BatchNum], J, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumdXextdWY and store the result in sumdXextdWY	
		
		//Barrier: Wait until all threads have reached this point
		MPI_Barrier(comm);	
																	
		//Optimise
		//This algorithm needs to be maximised, so we call the maximise function
		this->optimiser->maximise(comm, this, I, lengthW, GlobalBatchSize, tol, MaxNumIterations);	
		
		//Reformat W such that the transformations variance is one and their covariance is zero
		//This will have no impact on the chi value			
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Wmatrix(this->J, this->Jext), WTW(this->Jext, this->Jext), WTWInv(this->Jext, this->Jext), WTWInvSqrt(this->Jext, this->Jext);
		
		//Copy W to Wmatrix
		for (j=0; j < this->Jext; ++j)  for (i=0; i < this->J; ++i) Wmatrix(i, j) = this->W[j*(this->J) + i];
		
		//Calculate WTW
		WTW = Wmatrix.transpose()*Wmatrix;
		
		//Calculate WTWInv			
		WTWInv = WTW.inverse(); 		
		
		//Calculate WTWInvSqrt
		this->eigensolver.compute(WTWInv);
		WTWInvSqrt = this->eigensolver.operatorSqrt();		
		
		//Recalculate Wmatrix
		Wmatrix = Wmatrix*WTWInvSqrt;	
		
		//Copy Wmatrix to W
		for (j=0; j < this->Jext; ++j)  for (i=0; i < this->J; ++i)  this->W[j*(this->J) + i] = Wmatrix(i, j);		
					
		//Free all the arrays that have been allocated previously
		free(this->WTW);
		
		free(this->WBatchSize);
		free(this->CumulativeWBatchSize);
		
		free(this->sumY);
		free(this->sumYY);
		
		free(this->LocalsumY);
		free(this->LocalsumYY);		
																										
		for (i=0; i<this->NumBatches; ++i) free(this->sumdXextdW[i]);		
		for (i=0; i<this->NumBatches; ++i) free(this->sumdXextdWY[i]);				
						
		free(this->sumdXextdW);
		free(this->sumdXextdWY);		
		
		for (i=0; i<this->NumBatches; ++i) free(this->LocalsumdXextdW[i]);		
		for (i=0; i<this->NumBatches; ++i) free(this->LocalsumdXextdWY[i]);				
						
		free(this->LocalsumdXextdW);
		free(this->LocalsumdXextdWY);				
		
		free(this->sumXext);
		free(this->sumXextY);		
		free(this->sumXextXext);		
		
		free(this->LocalsumXext);
		free(this->LocalsumXextY);		
		free(this->LocalsumXextXext);				

		for (i=0; i<this->Jext; ++i) free(this->sumdXextdWXext[i]);			
		free(this->sumdXextdWXext);		
		
		for (i=0; i<this->Jext; ++i) free(this->LocalsumdXextdWXext[i]);			
		free(this->LocalsumdXextdWXext);			
		
		for (BatchNum=0; BatchNum<this->NumBatches; ++BatchNum) free(dXextdWData[BatchNum]);
		for (BatchNum=0; BatchNum<this->NumBatches; ++BatchNum) free(dXextdWIndices[BatchNum]);
		for (BatchNum=0; BatchNum<this->NumBatches; ++BatchNum) free(dXextdWIndptr[BatchNum]);
		
		free(dXextdWData);
		free(dXextdWIndices);
		free(dXextdWIndptr);		
										
	}
	
	void transform(double *Xext, int Jext, int I, double *XData, int XDataLength, int *XIndices, int XIndicesLength, int *XIndptr, int XIndptrLength, const int J) {
		
		int i,jext,k;
		
		//Make sure there are no invalid arguments
		if (J != this->J) throw std::invalid_argument("Number of attributes J does not match the J that has been defined when declaring the class!");				
		if (Jext != this->Jext) throw std::invalid_argument("Number of attributes Jext does not match the Jext that has been defined when declaring the class!");		
		if (XDataLength != XIndicesLength) throw std::invalid_argument("Length of XData does not match length of XIndices!");		
		if (XIndptrLength != I+1) throw std::invalid_argument("Number of instances in Xext does not match number of instances in X!");		
		
		//Make sure Xext is intialised to zero
		for (i=0; i<Jext*I; ++i) Xext[i] = 0.0;
		
		//Make sure Xext is intialised to zero		
		//Note that Xext is in column-major order!
		for (jext=0; jext<Jext; ++jext) for (i=0; i<I; ++i) 
			for (k=XIndptr[i]; k<XIndptr[i + 1]; ++k)
				Xext[jext*I + i] += XData[k]*this->W[jext*J + XIndices[k]];
	}		
	
	private:
	void calcXext(double *Xext, const double *W, const int BatchBegin, const int BatchSize);	
	void calcdXextdW(const int BatchBegin, const int BatchSize, const int BatchEnd, const int BatchNum);
		
	//Does not depend on W	
	void reduce1(const int BatchBegin, const int BatchEnd, const int BatchNum);//Calculate sumY
	void reduce2(const int BatchBegin, const int BatchEnd, const int BatchNum);//Calculate sumYY	
	void reduce3(const int BatchNum);//Calculate sumdXextdW
	void reduce4(const int BatchBegin, const int BatchNum);//Calculate sumdXextdWY
	
	//Does depend on W		
	void reduce5(const double *Xext, const int BatchSize);//Calculate sumXext
	void reduce6(const double *Xext, const double *y, const int BatchSize);	//Calculate sumXextY
	void reduce7(const double *Xext, const int BatchSize);//Calculate sumXextXext
	void reduce8(const double *Xext, const int BatchSize,  const int BatchNum);//Calculate sumXextdWXext

	double calcvar1(double &sumXext1, double &sumXext2, double &sumXextXext, 
			double &sumY, double &sumYY, double &BatchSize) {
		return (sumXextXext - sumXext1*sumXext2/BatchSize)*(sumYY - sumY*sumY/BatchSize)/(BatchSize-1.0);
	}	
		
	double calcdvardw1(double &sumdXextdW, double &sumXext, double &sumdXextdWXext, double &sumY, double &sumYY, double &BatchSize) {
		return (sumdXextdWXext - sumdXextdW*sumXext/BatchSize)*(sumYY - sumY*sumY/BatchSize)/(BatchSize-1.0);
	}
					
};

#include "LinearMahaFeatExtSparse_reduce1.hpp"



