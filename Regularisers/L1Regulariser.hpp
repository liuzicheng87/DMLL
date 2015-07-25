class L1RegulariserCpp: public RegulariserCpp {
	
	public:
	
	//Constructor
	L1RegulariserCpp(double alpha): RegulariserCpp(alpha) {}

	//Virtual destructor		
	~L1RegulariserCpp() {}
	
	//Z: Value to be optimised
	//WBatchBegin: Integer signifying the beginning of the batch 
	//WBatchEnd: Integer signifying the end of the batch
	//WBatchSize: Size of the Batch
	//WBatchSize = WBatchEnd - WBatchBegin
	void f (double &Z, const double *W, const int WBatchBegin, const int WBatchEnd, const int WBatchSize, const double BatchSizeDouble) {
		
		int i;
				
		for (i=WBatchBegin; i<WBatchEnd; ++i) Z += abs(W[i])*this->alpha*BatchSizeDouble;
				
	}
	
	//dZdW: derivative of value to be optimised	
	//WBatchBegin: Integer signifying the beginning of the batch 
	//WBatchEnd: Integer signifying the end of the batch
	//WBatchSize: Size of the Batch
	//WBatchSize = WBatchEnd - WBatchBegin	
	void g (double *dZdW, const double *W, const int WBatchBegin, const int WBatchEnd, const int WBatchSize, const double BatchSizeDouble) {

		int i;
		
		for (i=WBatchBegin; i<WBatchEnd; ++i) {
			if (W[i] > 0.0) dZdW[i] += this->alpha*BatchSizeDouble; 
			else if (W[i] < 0.0)  dZdW[i] -= this->alpha*BatchSizeDouble;
		}
				
	}
		
};
