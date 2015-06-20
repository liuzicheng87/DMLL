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
	
	//Matrices
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> V, VInv, ZEZ, dVdW, dZEZdW;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> eigensolver;
	Eigen::Matrix<double, 1, 1> gradient;	
	
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
		this->lengthW = J*Jext; 		
				
		//IMPORTANT: You must malloc W and initialise values randomly. How you randomly initialise them is your decision, but make sure you the the same values every time you call the function.
		this->W = (double*)malloc(this->lengthW*sizeof(double));
		for (i=0; i<this->lengthW; ++i) this->W[i] = dist(gen);
		
		//Resize matrices		
		this->V.resize(this->Jext, this->Jext);
		this->VInv.resize(this->Jext, this->Jext);		
		this->ZEZ.resize(this->Jext, 1);
		this->dVdW.resize(this->Jext, this->Jext);
		this->dZEZdW.resize(1, this->Jext);

	};
	
	~LinearMahaFeatExtSparseCpp() {
		//IMPORTANT: These three free's should appear in all classes that inherit from NumericallyOptimisedMLAlgorithmCpp!
		if (this->W != NULL) free(this->W);		
		if (this->SumGradients != NULL) free(this->SumGradients);
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
	void f(MPI_Comm comm, double &Z, const double *W, const int BatchBegin, const int BatchEnd, const int BatchSize, const int BatchNum, const int IterationNum) {	}
	
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
	void g(MPI_Comm comm, const double *W, const int BatchBegin, const int BatchEnd, const int BatchSize, const int BatchNum, const int IterationNum) {	
								
			const double *y;
			int i,j,k,a,c,d;
			double BatchSizeDouble = (double)BatchSize;
								
			for (i=0; i<this->Jext; ++i) this->LocalsumXext[i] = 0.0;
			for (i=0; i<this->Jext; ++i) this->LocalsumXextY[i] = 0.0;				
			for (i=0; i<(this->Jext*(this->Jext + 1))/2; ++i) this->LocalsumXextXext[i] = 0.0;
			for (i=0; i<this->Jext; ++i) for (j=0; j<this->J; ++j) this->LocalsumdXextdWXext[i][j] = 0.0;			
																														
			double *Xext = (double*)calloc(this->Jext*BatchSize, sizeof(double));
				
			//Declare a pointer that points to the part of this->Y we are interested in
			y = this->Y + BatchBegin;		
																																		
			calcXext(Xext, W, BatchBegin, BatchSize); //Calculate Xext										
			reduce4(Xext, BatchSize); //Calculate LocalsumXext
			reduce5(Xext, y, BatchSize); //Calculate LocalsumXextY							
			reduce6(Xext, BatchSize); //Calculate LocalsumXextXext
			reduce7(Xext,  BatchSize, BatchNum); //Calculate LocalsumdXextdWXext
									
			free(Xext);		
			
			//Add all localdZdW and store the result in dZdW
			MPI_Allreduce(this->LocalsumXext, this->sumXext, this->Jext, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumXext and store the result in sumXext	
			MPI_Allreduce(this->LocalsumXextY, this->sumXextY, this->Jext, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumXextY and store the result in sumXextY	
			MPI_Allreduce(this->LocalsumXextXext, this->sumXextXext, (this->Jext*(this->Jext + 1))/2, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumXextXext and store the result in sumXextXext		
			for (i=0; i<this->Jext; ++i) MPI_Allreduce(this->LocalsumdXextdWXext[i], this->sumdXextdWXext[i], this->J, MPI_DOUBLE, MPI_SUM, comm); //Add all LocalsumXextXext and store the result in sumXextXext		

			//Barrier: Wait until all threads have reached this point
			MPI_Barrier(comm);							
																 
			//Calculate Z - E(Z)
			for(a=0; a<Jext; ++a) this->ZEZ(a, 0) = this->sumXextY[a] - this->sumXext[a]*this->sumY[BatchNum]/BatchSizeDouble;
																							
			//Calculate V							
			for (i=0; i<Jext; ++i) for (k=0; k<=i; ++k)
				this->V(i, k) = this->V(k, i) =
					calcvar1(sumXext[i], sumXext[k], sumXextXext[i*(i+1)/2 + k], sumY[BatchNum], sumYY[BatchNum], BatchSizeDouble); 
																																																			
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
				if (dXextdWIndptr[BatchNum][c+1] == dXextdWIndptr[BatchNum][c]) continue;
								
				//Calculate dZEZdw	
				dZEZdW.setZero();
				dZEZdW(0, d) = sumdXextdWY[BatchNum][c] - sumdXextdW[BatchNum][c]*sumY[BatchNum]/BatchSizeDouble;
																													
				//Calculate dVdw						
				dVdW.setZero();
												
				for (a=0; a<d; ++a) this->dVdW(d, a) = this->dVdW(a, d) =
					calcdvardw1(sumdXextdW[BatchNum][c], sumXext[a], sumdXextdWXext[a][c], sumY[BatchNum], sumYY[BatchNum], BatchSizeDouble);
														
				this->dVdW(d, d)	= 2.0*calcdvardw1(sumdXextdW[BatchNum][c], sumXext[d], sumdXextdWXext[d][c], sumY[BatchNum], sumYY[BatchNum], BatchSizeDouble);
																						
				for (a=d+1; a<this->Jext; ++a) this->dVdW(d, a) = this->dVdW(a, d) = 
					calcdvardw1(sumdXextdW[BatchNum][c], sumXext[a], sumdXextdWXext[a][c], sumY[BatchNum], sumYY[BatchNum], BatchSizeDouble);
																																															
				//Calculate the gradient
				gradient = 2.0*dZEZdW*VInv*ZEZ - ZEZ.transpose()*VInv*dVdW*VInv*ZEZ;
													
				this->optimiser->localdZdW[i] = gradient(0,0);
																			
			}			

			//Gather all localdZdW and store the result in dZdW
			MPI_Allgatherv(this->optimiser->localdZdW + this->WBatchBegin, this->WBatchSize[this->optimiser->rank], MPI_DOUBLE, this->optimiser->dZdW, this->WBatchSize, this->CumulativeWBatchSize, MPI_DOUBLE, comm);				
			
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
				
		int i, BatchNum, BatchBegin, BatchEnd, BatchSize, SparseDataSize;
						
		const double *y;

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
																		
		//Calculate variables that DO NOT depend on W. Since they, by defintion, will not change as we optimise W, we calculate them before conducting the actual optimisation.
		for (BatchNum=0; BatchNum<this->NumBatches; ++BatchNum) {
								
			//We must find out our current values for BatchBegin and BatchEnd. We do so by calling this->CalcBatchBeginEnd, which is inherited from the optimiser class.
			this->optimiser->CalcBatchBeginEnd(BatchBegin, BatchEnd, BatchSize, BatchNum, this->I, this->NumBatches);
				
			//Declare a pointer that points to the part of this->Y we are interested in
			y = this->Y + BatchBegin;
				
			//sumY and sumYY are part of the sufficient statistics, but obviously do not depend on W
			reduce1(y, BatchSize, this->sumY[BatchNum], this->sumYY[BatchNum]);
			
			//We do not know the size of the vectors  this->dXextdWData and this->dXextdWIndices beforehand, so we cannot allocate them earlier
			SparseDataSize = this->XIndptr[BatchEnd] - this->XIndptr[BatchBegin];
			
			this->dXextdWData[BatchNum] = (double*)calloc(SparseDataSize, sizeof(double));
			this->dXextdWIndices[BatchNum] = (int*)calloc(SparseDataSize, sizeof(int));
			this->dXextdWIndptr[BatchNum] = (int*)calloc(J+1, sizeof(int));		
			
			//calcdXextdW() calculates dXextdWData, dXextdWIndptr and dXextdWIndptr for the respective  batch
			calcdXextdW(BatchBegin, BatchSize, BatchEnd, BatchNum);
							
			//reduce2() calculates sumdXextdW for the respective thread and batch				
			reduce2(BatchNum);		
				
			//reduce3() calculates sumdXextdWY for the respective thread and batch															
			reduce3(BatchBegin, BatchNum);	
									
		}			
															
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
	
	void transform(double *Xext, int I, int Jext, double *XData, int XDataLength, int *XIndices, int XIndicesLength, int *XIndptr, int XIndptrLength, const int J) {
		int i,jext,k;
		
		//Make sure there are no invalid arguments
		if (J != this->J) throw std::invalid_argument("Number of attributes J does not match the J that has been defined when declaring the class!");				
		if (Jext != this->Jext) throw std::invalid_argument("Number of attributes Jext does not match the Jext that has been defined when declaring the class!");		
		if (XDataLength != XIndicesLength) throw std::invalid_argument("Length of XData does not match length of XIndices!");		
		if (XIndptrLength != I+1) throw std::invalid_argument("Number of instances in Xext does not match number of instances in X!");		
		
		//Make sure Xext is intialised to zero
		for (i=0; i<this->Jext*I; ++i) Xext[i] = 0.0;
		
		//Make sure Xext is intialised to zero		
		//Note that Xext is in column-major order!
		for (jext=0; jext<this->Jext; ++jext) for (i=0; i<I; ++i) 
			for (k=XIndptr[i]; k<XIndptr[i + 1]; ++k)
				Xext[jext*I + i] += this->XData[k]*this->W[jext*J + XIndices[k]];
	}		
	
	private:
	void calcXext(double *Xext, const double *W, const int BatchBegin, const int BatchSize);	
	void calcdXextdW(const int BatchBegin, const int BatchSize, const int BatchEnd, const int BatchNum);
		
	//Does not depend on W	
	void reduce1(const double *y, int BatchSize, double &LocalsumY, double &LocalsumYY);//Calculate sumY and sumYY
	void reduce2(const int BatchNum);//Calculate sumdXextdW
	void reduce3(const int BatchBegin, const int BatchNum);//Calculate sumdXextdWY
	
	//Does depend on W		
	void reduce4(const double *Xext, const int BatchSize);//Calculate sumXext
	void reduce5(const double *Xext, const double *y, const int BatchSize);	//Calculate sumXextY
	void reduce6(const double *Xext, const int BatchSize);//Calculate sumXextXext
	void reduce7(const double *Xext, const int BatchSize,  const int BatchNum);//Calculate sumXextdWXext

	double calcvar1(double &sumXext1, double &sumXext2, double &sumXextXext, 
			double &sumY, double &sumYY, double &BatchSize) {
		return (sumXextXext - sumXext1*sumXext2/BatchSize)*(sumYY - sumY*sumY/BatchSize)/(BatchSize-1.0);
	}	
		
	double calcdvardw1(double &sumdXextdW, double &sumXext, double &sumdXextdWXext, double &sumY, double &sumYY, double &BatchSize) {
		return (sumdXextdWXext - sumdXextdW*sumXext/BatchSize)*(sumYY - sumY*sumY/BatchSize)/(BatchSize-1.0);
	}
					
};

#include "LinearMahaFeatExtSparse_reduce1.hpp"



