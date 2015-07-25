class RegulariserCpp {
	
	public:
	double alpha;
	int lengthW;
	
	//Constructor
	RegulariserCpp(double alpha) {
		
		this->alpha = alpha;
				
		}

	//Virtual destructor		
	virtual ~RegulariserCpp() {}

	//Z: Value to be optimised	
	//dZdW: derivative of value to be optimised	
	//WBatchBegin: Integer signifying the beginning of the batch 
	//WBatchEnd: Integer signifying the end of the batch
	//WBatchSize: Size of the Batch
	//WBatchSize = WBatchEnd - WBatchBegin	
	virtual void f (double &Z, const double *W, const int WBatchBegin, const int WBatchEnd, const int WBatchSize, const double BatchSizeDouble) {}
	virtual void g (double *dZdW, const double *W, const int WBatchBegin, const int WBatchEnd, const int WBatchSize, const double BatchSizeDouble) {}
		
};

#include "L1Regulariser.hpp"
#include "L2Regulariser.hpp"

