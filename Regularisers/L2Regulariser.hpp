class L2RegulariserCpp: public RegulariserCpp {
	
	public:
	
	//Constructor
	L2RegulariserCpp(double alpha): RegulariserCpp(alpha) {}

	//Virtual destructor		
	~L2RegulariserCpp() {}
	
	//Z: Value to be optimised
	//WBatchBegin: Integer signifying the beginning of WBatch 
	//WBatchEnd: Integer signifying the end of WBatch
	//WBatchSize: Size of the WBatch
	//WBatchSize = WBatchEnd - WBatchBegin
	//BatchSize = size of batch of data (not W)
	void f (double &Z, const double *W, const int WBatchBegin, const int WBatchEnd, const int WBatchSize, const double BatchSizeDouble) {
		
		int i;
				
		for (i=WBatchBegin; i<WBatchEnd; ++i) Z += this->alpha*W[i]*W[i]*BatchSizeDouble;
				
	}
	
	//dZdW: derivative of value to be optimised	
	//WBatchBegin: Integer signifying the beginning of the batch 
	//WBatchEnd: Integer signifying the end of the batch
	//WBatchSize: Size of the Batch
	//WBatchSize = WBatchEnd - WBatchBegin	
	//BatchSize = size of batch of data (not W)	
	void g (double *dZdW, const double *W, const int WBatchBegin, const int WBatchEnd, const int WBatchSize, const double BatchSizeDouble) {
		
		int i;

		for (i=WBatchBegin; i<WBatchEnd; ++i) dZdW[i] += this->alpha*W[i]*BatchSizeDouble;

	}
		
};
