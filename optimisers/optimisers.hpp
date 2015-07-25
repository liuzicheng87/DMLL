#include "NumericallyOptimisedMLAlgorithm.hpp"	
		
class OptimiserCpp {
	
	public:
	//All optimiser use minibatch updating and parallelisation. We split up the dataset into batches, the size of which is determined by BatchSize. 
	//These batches are then split up into smaller batches for each thread. The size of these "thread batches" is determined by BatchSize. The sum of all BatchSizes adds up to BatchSize.
	//BatchSize can vary, if I (the number of instances or samples) is not divisable by BatchSize	
	//Z: the value to be optimised	
	//W: the weights to be optimised
	//dZdW: the number attributes or features	
	//I: the number of instances or samples 
	//lengthW: the number of weights to be optimised	
	//BatchSize: the size of the minibatches in minibatch updating
	//size: the number of processes
	//rank: the rank of this process
	//NumBatches: (inversely proportional to size)
	double Z, *W, *dZdW, *localdZdW, *SumdZdW;
	int I, lengthW, GlobalBatchSize, size, rank, NumBatches;
	NumericallyOptimisedMLAlgorithmCpp *MLalgorithm;
	
	//Constructor
	OptimiserCpp(const int size, const int rank) {
		
		this->size = size;
		this->rank = rank;
		
		}

	//Virtual destructor		
	virtual ~OptimiserCpp() {}
	
	//MLalgorithm: pointer to machine learning algorithm to be optimised
	//I: number of instances in dataset
	//lengthW: number of weights
	//GlobalBatchSize: size of minibatches for updating
	//tol: the error tolerance
	//MaxNumIterations: the maximum number of iterations tolerated until optimiser stops
	//size: the number of threads used
	//rank: Each thread is assigned an individual number ranging from 0 to the number of threads minus one. This is used for identification. 	
	//maximise() and minimise() both run until either the sum of gradients is smaller than tol or the number of iterations reaches MaxNumIterations
	void maximise (MPI_Comm comm, NumericallyOptimisedMLAlgorithmCpp *MLalgorithm, int I, int lengthW, int GlobalBatchSize, const double tol, const int MaxNumIterations);
	void minimise (MPI_Comm comm, NumericallyOptimisedMLAlgorithmCpp *MLalgorithm, int I, int lengthW, int GlobalBatchSize, const double tol, const int MaxNumIterations);
		
	//This is were you come in. What happens within these two functions is entirely your responsibility.
	virtual void max(MPI_Comm comm, const double tol, const int MaxNumIterations) {std::cout << "This shouldn't happen!\n You need to use an optimising algorithm, not the base class!\n";}
	virtual void min(MPI_Comm comm, const double tol, const int MaxNumIterations) {std::cout << "This shouldn't happen!\n You need to use an optimising algorithm, not the base class!\n";}
	
	//CalcNumBatches calculates the number of batches needed
	void CalcNumBatches (MPI_Comm comm);
	void CalcNumBatches (MPI_Comm comm, int I, int GlobalBatchSize, int &NumBatches);
		
	//BatchBegin: Integer signifying the beginning of the batch 
	//BatchEnd: Integer signifying the end of the batch
	//BatchSize: Size of the Batch
	//BatchSize = BatchEnd - BatchBegin
	//BatchNum: Integer iterating throught the batches
	void CalcBatchBeginEnd (int &BatchBegin, int &BatchEnd, int &BatchSize, const int BatchNum);
	void CalcBatchBeginEnd (int &BatchBegin, int &BatchEnd, int &BatchSize, const int BatchNum, const int I, const int NumBatches);
		
};

//Functions are in alphabetical order

//CalcNumBatches calculates the number of batches needed
void OptimiserCpp::CalcNumBatches (MPI_Comm comm) {
	
	int GlobalI;
	
	//Add all local I and store the result in GlobalI
	MPI_Allreduce(&(this->I), &GlobalI, 1, MPI_INT, MPI_SUM, comm);
	MPI_Barrier(comm);
	
	if (this->GlobalBatchSize < 1 || this->GlobalBatchSize > GlobalI) this->GlobalBatchSize = GlobalI;		
			
	//Calculate the number of batches needed to divide GlobalI such that the sum of all local batches approximately equals GlobalBatchSize
	if (GlobalI % GlobalBatchSize == 0) this->NumBatches = GlobalI/this->GlobalBatchSize; else this->NumBatches = GlobalI/this->GlobalBatchSize + 1;
	MPI_Barrier(comm);
		
}

//CalcNumBatches calculates the number of batches needed
//Some machine learning algorithms need the function CalcBatchBeginEnd for initialisation. We therefore create a version that does not depend on calling variables contained in the optimiser class (no this->).
void OptimiserCpp::CalcNumBatches (MPI_Comm comm, int I, int GlobalBatchSize, int &NumBatches) {
	
	int GlobalI;
	
	//Add all local I and store the result in GlobalI
	MPI_Allreduce(&I, &GlobalI, 1, MPI_INT, MPI_SUM, comm);
	MPI_Barrier(comm);
	
	if (GlobalBatchSize < 1 || GlobalBatchSize > GlobalI) GlobalBatchSize = GlobalI;		
			
	//Calculate the number of batches needed to divide GlobalI such that the sum of all local batches approximately equals GlobalBatchSize
	if (GlobalI % GlobalBatchSize == 0) NumBatches = GlobalI/GlobalBatchSize; else NumBatches = GlobalI/GlobalBatchSize + 1;
	MPI_Barrier(comm);
				
}	

//BatchBegin and BatchEnd are used to share the burden evenly among the processes
void OptimiserCpp::CalcBatchBeginEnd (int &BatchBegin, int &BatchEnd, int &BatchSize, const int BatchNum) {
									
		//Calculate BatchBegin
		BatchBegin = BatchNum*(this->I/this->NumBatches);
		
		//Calculate WBatchSize
		if (BatchNum < this->NumBatches-1) BatchSize = this->I/this->NumBatches;
		else BatchSize = this->I - BatchBegin;
		
		//Calculate WBatchEnd
		BatchEnd = BatchBegin + BatchSize;
	
}

//BatchBegin and BatchEnd are used to share the burden evenly among the processes
//Some machine learning algorithms need the function CalcBatchBeginEnd for initialisation. We therefore create a version that does not depend on calling variables contained in the optimiser class (no this->).
void OptimiserCpp::CalcBatchBeginEnd (int &BatchBegin, int &BatchEnd, int &BatchSize, const int BatchNum, const int I, const int NumBatches) {
									
		//Calculate BatchBegin
		BatchBegin = BatchNum*(I/NumBatches);
		
		//Calculate WBatchSize
		if (BatchNum < NumBatches-1) BatchSize = I/NumBatches;
		else BatchSize = I - BatchBegin;
		
		//Calculate WBatchEnd
		BatchEnd = BatchBegin + BatchSize;
	
}


void OptimiserCpp::maximise (MPI_Comm comm, NumericallyOptimisedMLAlgorithmCpp *MLalgorithm, int I, int lengthW, int GlobalBatchSize, const double tol, const int MaxNumIterations) {
	
	//Store all of the input values
	this->MLalgorithm = MLalgorithm;
	this->I = I; 
	this->lengthW = lengthW; 
	this->GlobalBatchSize = GlobalBatchSize;
				
	//W: weights, stored by the MLalgorithm
	//dZdW: gradient of the value to be optimised
	//SumdZdW: the sum over all batches in one iteration
	//dZdW is always set to 0 before passing in to g
	//SumGradients is used to document the gradients after every iteration
	this->dZdW = (double*)calloc(lengthW, sizeof(double));
	this->localdZdW = (double*)calloc(lengthW, sizeof(double));	
	this->SumdZdW = (double*)calloc(lengthW, sizeof(double));	
	
	//W and SumGradients are free'd when the MLalgorithm is destroyed
	this->W = this->MLalgorithm->W;	//Note that we are simply pointing to the W's in MLalgorithm
	if (this->MLalgorithm->SumGradients != NULL) free(this->MLalgorithm->SumGradients);
	this->MLalgorithm->SumGradients = (double*)calloc(MaxNumIterations, sizeof(double));
	
	//Calculate the number of batches
	this->CalcNumBatches (comm);
					
	//Create the threads and pass the values they need
	max(comm, tol, MaxNumIterations);
	
	free(this->dZdW);
	free(this->localdZdW);	
	free(this->SumdZdW);
	
}


void OptimiserCpp::minimise (MPI_Comm comm, NumericallyOptimisedMLAlgorithmCpp *MLalgorithm, int I, int lengthW, int GlobalBatchSize, const double tol, const int MaxNumIterations) {
	
	//Store all of the input values
	this->MLalgorithm = MLalgorithm;
	this->I = I; 
	this->lengthW = lengthW; 
	this->GlobalBatchSize = GlobalBatchSize;
				
	//W: weights, stored by the MLalgorithm
	//dZdW: gradient of the value to be optimised
	//SumdZdW: the sum over all batches in one iteration
	//dZdW is always set to 0 before passing in to g
	//SumGradients is used to document the gradients after every iteration
	this->dZdW = (double*)calloc(lengthW, sizeof(double));
	this->localdZdW = (double*)calloc(lengthW, sizeof(double));		
	this->SumdZdW = (double*)calloc(lengthW, sizeof(double));	
	
	//W and SumGradients are free'd when the MLalgorithm is destroyed
	this->W = this->MLalgorithm->W;	//Note that we are simply pointing to the W's in MLalgorithm
	if (this->MLalgorithm->SumGradients != NULL) free(this->MLalgorithm->SumGradients);
	this->MLalgorithm->SumGradients = (double*)calloc(MaxNumIterations, sizeof(double));
	
	//Calculate the number of batches
	this->CalcNumBatches (comm);
					
	//Create the threads and pass the values they need
	min(comm, tol, MaxNumIterations);
	
	free(this->dZdW);
	free(this->localdZdW);		
	free(this->SumdZdW);
	
}


#include "GradientDescent.hpp"
#include "GradientDescentWithMomentum.hpp"
#include "BacktrackingLineSearch.hpp"

