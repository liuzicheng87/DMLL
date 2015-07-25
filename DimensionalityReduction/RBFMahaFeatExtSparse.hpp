class RBFMahaFeatExtSparseCpp: public NumericallyOptimisedMLAlgorithmCpp {
	
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
	double  *sumdXextdW, *sumdXextdWY;
	double  *LocalsumdXextdW, *LocalsumdXextdWY;
	
	//Variables that need to be recalculated in every iteration
	double  *sumXext, *sumXextY, *sumXextXext,  **sumdXextdWXext;
	double  *LocalsumXext, *LocalsumXextY, *LocalsumXextXext,  **LocalsumdXextdWXext;
	
	//Matrices
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> V, VInv, ZEZ, dVdW, dZEZdW;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> eigensolver;
	Eigen::Matrix<double, 1, 1> chi, gradient;	
		
	//XMinusCSquaredData, XMinusCSquaredData and XMinusCSquaredIndptr store the local versions of XMinusCSquared in sparse format
	double **XMinusCSquaredData;
	int **XMinusCSquaredIndices;
	int **XMinusCSquaredIndptr;	
	
	//cData, cIndices and cIndptr store the centres of this particular RBF
	double *cData;
	int *cIndices;
	int *cIndptr;		
	
	RegulariserCpp *regulariser;
	
	//Should all W's be smaller than zero?
	bool clipW;
		
	//We keep the parameter Jext in case we want to extend the algorithm to include a small number of different centres and RBFs
	RBFMahaFeatExtSparseCpp (const int J, double *cData, int cDataLength, int *cIndices, int cIndicesLength,  int *cIndptr, int cIndptrLength, const int Jext, RegulariserCpp *regulariser, bool clipW): NumericallyOptimisedMLAlgorithmCpp()  {
		
		int i;
		
		//Necessary for pseudo-random number generator
		std::mt19937 gen(1);//Note that we deliberately choose a constant seed to get the same output every time we call the function
		std::normal_distribution<double> dist(0.0, 1.0);//Normal distribution with mean 0 and standard deviation of 1
		
		//Store input values
		this->J = J; 
		this->Jext = Jext;
		this->cIndptr = (int*)malloc((this->Jext + 1)*sizeof(int));
		for (i=0; i<(this->Jext + 1); ++i) this->cIndptr[i] = cIndptr[i];
		this->cIndices = (int*)malloc((cIndptr[Jext])*sizeof(int));
		for (i=0; i<cIndptr[Jext]; ++i) this->cIndices[i] = cIndices[i];
		this->cData = (double*)malloc((cIndptr[Jext])*sizeof(double));
		for (i=0; i<cIndptr[Jext]; ++i) this->cData[i] = cData[i];		
		this->regulariser = regulariser;
		this->clipW = clipW;
		
		//IMPORTANT: this->lengthW needs to be defined when initialising an algorithm!
		this->lengthW = J*Jext; 		
				
		//IMPORTANT: You must malloc W and initialise values randomly. How you randomly initialise them is your decision, but make sure you the the same values every time you call the function.
		this->W = (double*)malloc(this->lengthW*sizeof(double));
		for (i=0; i<this->lengthW; ++i) this->W[i] = (-1.0)*abs(dist(gen));
		
		//Resize matrices		
		this->V.resize(this->Jext, this->Jext);
		this->VInv.resize(this->Jext, this->Jext);		
		this->ZEZ.resize(this->Jext, 1);
		this->dVdW.resize(this->Jext, this->Jext);
		this->dZEZdW.resize(1, this->Jext);
		
		//If clipW is True, then initialise maxW
		if (clipW) {
					
			this->wMaxLength = this->lengthW;
			this->wMaxIndices = (int*)malloc(this->wMaxLength*sizeof(int));
			for (i=0; i<this->lengthW; ++i) this->wMaxIndices[i] = i;
			
			this->wMax = (double*)calloc(this->wMaxLength, sizeof(double));//wMax is set to zero for all W
		}

	};
	
	~RBFMahaFeatExtSparseCpp() {
		free(cData);
		free(cIndices);
		free(cIndptr);
		};
						
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
	void f(MPI_Comm comm, double &Z, const double *W, const int BatchBegin, const int BatchEnd, const int BatchSize, const int BatchNum, const int IterationNum) {	

			const double *y;
			int i,j,k,a,c,d;
			double BatchSizeDouble = (double)BatchSize;
			double GlobalBatchSizeDouble;
																																												
			//The size of Xext depends on BatchSize. Therefore, it needs to be initalised within an iteration																																				
			double *Xext = (double*)calloc(this->Jext*BatchSize, sizeof(double));
				
			//Declare a pointer that points to the part of this->Y we are interested in
			y = this->Y + BatchBegin;		

			calcXext(Xext, W, BatchBegin, BatchEnd); //Calculate Xext				
			reduce3(Xext, BatchSize);//Calculate sumXext
			reduce4(Xext, y, BatchSize);//Calculate sumXextY				
			reduce5(Xext, BatchSize); //Calculate LocalsumXext

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
			
			//If all W are zero, then chi is undefined														
			if (chi(0,0) == chi(0,0)) Z = chi(0,0); else Z = 0.0;
			
			//Apply regulariser
			this->regulariser->f(Z, W, 0, this->lengthW, this->lengthW, GlobalBatchSizeDouble); 			
												
			//Barrier: Wait until all threads have reached this point
			//It the the responsibility of every ML algorithm to pass the complete dZdW to the optimiser
			MPI_Barrier(comm);			
		
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
	void g(MPI_Comm comm, double *dZdW, double *localdZdW, const double *W, const int BatchBegin, const int BatchEnd, const int BatchSize, const int BatchNum, const int IterationNum) {	
											
			const double *y;
			int i,j,k,a,c,d;
			double BatchSizeDouble = (double)BatchSize;
			double GlobalBatchSizeDouble;
			
			//The size of Xext depends on BatchSize. Therefore, it needs to be initalised within an iteration																																				
			double *Xext = (double*)calloc(this->Jext*BatchSize, sizeof(double));
				
			//Declare a pointer that points to the part of this->Y we are interested in
			y = this->Y + BatchBegin;		

			calcXext(Xext, W, BatchBegin, BatchEnd); //Calculate Xext				
			reduce3(Xext, BatchSize);//Calculate sumXext
			reduce4(Xext, y, BatchSize);//Calculate sumXextY				
			reduce5(Xext, BatchSize); //Calculate LocalsumXext
			reduce6to8(Xext, y, W, BatchBegin, BatchNum); //Calculate LocalsumdXextdW, LocalsumdXextdWY and LocalsumdXextdWXext

			free(Xext);		
			
			//Execute AllReduce operations
			MPI_Allreduce(&BatchSizeDouble, &GlobalBatchSizeDouble, 1, MPI_DOUBLE, MPI_SUM, comm); //Add all BatchSizeDouble and store the result in GlobalBatchSizeDouble	

			MPI_Allreduce(this->LocalsumXext, this->sumXext, this->Jext, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumXext and store the result in sumXext	
			MPI_Allreduce(this->LocalsumXextY, this->sumXextY, this->Jext, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumXextY and store the result in sumXextY	
			MPI_Allreduce(this->LocalsumXextXext, this->sumXextXext, (this->Jext*(this->Jext + 1))/2, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumXextXext and store the result in sumXextXext	
			MPI_Allreduce(this->LocalsumdXextdW, this->sumdXextdW, J, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumdXextdW and store the result in sumdXextdW	
			MPI_Allreduce(this->LocalsumdXextdWY, this->sumdXextdWY, J, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumdXextdWY and store the result in sumdXextdWY	
			MPI_Allreduce(this->LocalsumdXextdWXext[0], this->sumdXextdWXext[0], this->J, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumdXextdWXext and store the result in sumXextXext		

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
			for (i=this->WBatchBegin; i < this->WBatchEnd; ++i) {
										
				//Calculate c and d
				//c=0,1,...,J-1 is the index for the original feature
				//c=0,1,...,Jext-1 is the index for the extracted feature	
				c = i%(this->J);
				d = i/(this->J);
				
				//If there is no data anyway, continue
				if (XMinusCSquaredIndptr[BatchNum][c+1] == XMinusCSquaredIndptr[BatchNum][c]) continue;
								
				//Calculate dZEZdw	
				dZEZdW.setZero();
				dZEZdW(0, d) = sumdXextdWY[c] - sumdXextdW[c]*sumY[BatchNum]/GlobalBatchSizeDouble;
																													
				//Calculate dVdw						
				dVdW.setZero();
												
				for (a=0; a<d; ++a) this->dVdW(d, a) = this->dVdW(a, d) =
					calcdvardw1(sumdXextdW[c], sumXext[a], sumdXextdWXext[a][c], sumY[BatchNum], sumYY[BatchNum], GlobalBatchSizeDouble);
														
				this->dVdW(d, d)	= 2.0*calcdvardw1(sumdXextdW[c], sumXext[d], sumdXextdWXext[d][c], sumY[BatchNum], sumYY[BatchNum], GlobalBatchSizeDouble);
																						
				for (a=d+1; a<this->Jext; ++a) this->dVdW(d, a) = this->dVdW(a, d) = 
					calcdvardw1(sumdXextdW[c], sumXext[a], sumdXextdWXext[a][c], sumY[BatchNum], sumYY[BatchNum], GlobalBatchSizeDouble);
																																															
				//Calculate the gradient
				gradient = 2.0*dZEZdW*VInv*ZEZ - ZEZ.transpose()*VInv*dVdW*VInv*ZEZ;
				
				//If all W are zero, then gradient is undefined											
				if (gradient(0,0) == gradient(0,0)) localdZdW[i] = gradient(0,0); else localdZdW[i] = (-1.0)*GlobalBatchSizeDouble;
																			
			}			
			
			//Apply regulariser
			this->regulariser->g(localdZdW, W, this->WBatchBegin, this->WBatchEnd, this->WBatchSize[this->optimiser->rank], GlobalBatchSizeDouble);
						
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

		this->sumdXextdW= (double*)calloc(J, sizeof(double));	
		this->sumdXextdWY = (double*)calloc(J, sizeof(double));	

		this->LocalsumdXextdW= (double*)calloc(J, sizeof(double));	
		this->LocalsumdXextdWY = (double*)calloc(J, sizeof(double));	
																		
		//Initialise variables that DO depend on W
		this->sumXext = (double*)calloc(Jext, sizeof(double));
		this->sumXextY = (double*)calloc(Jext, sizeof(double));		
		this->sumXextXext = (double*)calloc((Jext*(Jext + 1))/2, sizeof(double));

		this->LocalsumXext = (double*)calloc(Jext, sizeof(double));
		this->LocalsumXextY = (double*)calloc(Jext, sizeof(double));		
		this->LocalsumXextXext = (double*)calloc((Jext*(Jext + 1))/2, sizeof(double));
		
		this->sumdXextdWXext = (double**)malloc(Jext*sizeof(double*));
		for (i=0; i<this->Jext; ++i) this->sumdXextdWXext[i] = (double*)calloc(this->J, sizeof(double));	
		
		this->LocalsumdXextdWXext = (double**)malloc(Jext*sizeof(double*));
		for (i=0; i<this->Jext; ++i) this->LocalsumdXextdWXext[i] = (double*)calloc(this->J, sizeof(double));	
				
		this->XMinusCSquaredData = (double**)malloc(this->NumBatches*sizeof(double*));
		this->XMinusCSquaredIndices  = (int**)malloc(this->NumBatches*sizeof(int*));
		this->XMinusCSquaredIndptr  = (int**)malloc(this->NumBatches*sizeof(int*));	
																			
		//Calculate variables that DO NOT depend on W. Since they, by defintion, will not change as we optimise W, we calculate them before conducting the actual optimisation.
		for (BatchNum=0; BatchNum<this->NumBatches; ++BatchNum) {
								
			//We must find out our current values for BatchBegin and BatchEnd. We do so by calling this->CalcBatchBeginEnd, which is inherited from the optimiser class.
			this->optimiser->CalcBatchBeginEnd(BatchBegin, BatchEnd, BatchSize, BatchNum, this->I, this->NumBatches);
								
			//sumY and sumYY are part of the sufficient statistics, but obviously do not depend on W
			reduce1(BatchBegin, BatchEnd, BatchNum);
			reduce2(BatchBegin, BatchEnd, BatchNum);
						
			//We do not know the size of the vectors  this->XMinusCSquaredData and this->XMinusCSquaredIndices beforehand, so we cannot allocate them earlier
			SparseDataSize = this->XIndptr[BatchEnd] - this->XIndptr[BatchBegin];
			
			this->XMinusCSquaredData[BatchNum] = (double*)calloc(SparseDataSize, sizeof(double));
			this->XMinusCSquaredIndices[BatchNum] = (int*)calloc(SparseDataSize, sizeof(int));
			this->XMinusCSquaredIndptr[BatchNum] = (int*)calloc(J+1, sizeof(int));		
			
			//calcXMinusCSquared() calculates XMinusCSquaredData, XMinusCSquaredIndptr and XMinusCSquaredIndptr for the respective process and batch				
			calcXMinusCSquared(BatchBegin, BatchSize, BatchEnd, BatchNum);
														
		}			
				
		//Reduce variables
		MPI_Allreduce(this->LocalsumY, this->sumY, this->NumBatches, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumY and store the result in sumY	
		MPI_Allreduce(this->LocalsumYY, this->sumYY, this->NumBatches, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumYY and store the result in sumYY	
				
		//Barrier: Wait until all processes have reached this point
		MPI_Barrier(comm);	
																			
		//Optimise
		//This algorithm needs to be maximised, so we call the maximise function
		this->optimiser->maximise(comm, this, I, lengthW, GlobalBatchSize, tol, MaxNumIterations);	
							
		//Free all the arrays that have been allocated previously
		free(this->WBatchSize);
		free(this->CumulativeWBatchSize);
		
		free(this->sumY);
		free(this->sumYY);
		
		free(this->LocalsumY);
		free(this->LocalsumYY);		
																																
		free(this->sumdXextdW);
		free(this->sumdXextdWY);		
								
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
		
		for (BatchNum=0; BatchNum<this->NumBatches; ++BatchNum) free(XMinusCSquaredData[BatchNum]);
		for (BatchNum=0; BatchNum<this->NumBatches; ++BatchNum) free(XMinusCSquaredIndices[BatchNum]);
		for (BatchNum=0; BatchNum<this->NumBatches; ++BatchNum) free(XMinusCSquaredIndptr[BatchNum]);
		
		free(XMinusCSquaredData);
		free(XMinusCSquaredIndices);
		free(XMinusCSquaredIndptr);		
										
	}
	
	void transform(double *Xext, int Jext, int I, double *XData, int XDataLength, int *XIndices, int XIndicesLength, int *XIndptr, int XIndptrLength, const int J) {
		
		int i,jext,k;
		
		//Make sure there are no invalid arguments
		if (J != this->J) throw std::invalid_argument("Number of attributes J does not match the J that has been defined when declaring the class!");				
		if (Jext != this->Jext) throw std::invalid_argument("Number of attributes Jext does not match the Jext that has been defined when declaring the class!");		
		if (XDataLength != XIndicesLength) throw std::invalid_argument("Length of XData does not match length of XIndices!");		
		if (XIndptrLength != I+1) throw std::invalid_argument("Number of instances in Xext does not match number of instances in X!");		
		
		this->XData = XData;
		this->XIndices = XIndices;
		this->XIndptr = XIndptr;		
		this->Y = Y;
		this->I = XIndptrLength - 1;//This relationship follows from the structure of a CSR_matrix		
		
		//Make sure Xext is intialised to zero
		for (i=0; i<this->Jext*I; ++i) Xext[i] = 0.0;
		
		//Note that Xext is in column-major order!
		calcXext(Xext, this->W, 0, I);
		
	}		
	
	private:
	void calcXext(double *Xext, const double *W, const int BatchBegin, const int BatchEnd);	
	void calcXMinusCSquared(const int BatchBegin, const int BatchSize, const int BatchEnd, const int BatchNum);
		
	//Does not depend on W	
	void reduce1(const int BatchBegin, const int BatchEnd, const int BatchNum);//Calculate sumY
	void reduce2(const int BatchBegin, const int BatchEnd, const int BatchNum);//Calculate sumYY	
	
	//Does depend on W		
	void reduce3(const double *Xext, const int BatchSize);//Calculate sumXext
	void reduce4(const double *Xext, const double *y, const int BatchSize);//Calculate sumXextY
	void reduce5(const double *Xext, const int BatchSize);//Calculate sumXextXext
	void reduce6to8(const double *Xext, const double *y, const double *W, const int BatchBegin, const int BatchNum);	//Calculate sumdXextdW, sumdXextdWY, sumdXextdWXext


	double calcvar1(double &sumXext1, double &sumXext2, double &sumXextXext, 
			double &sumY, double &sumYY, double &BatchSize) {
		return (sumXextXext - sumXext1*sumXext2/BatchSize)*(sumYY - sumY*sumY/BatchSize)/(BatchSize-1.0);
	}	
		
	double calcdvardw1(double &sumdXextdW, double &sumXext, double &sumdXextdWXext, double &sumY, double &sumYY, double &BatchSize) {
		return (sumdXextdWXext - sumdXextdW*sumXext/BatchSize)*(sumYY - sumY*sumY/BatchSize)/(BatchSize-1.0);
	}
					
};

#include "RBFMahaFeatExtSparse_reduce1.hpp"



