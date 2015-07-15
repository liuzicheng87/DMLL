	
	//Calculate Xext
	void RBFMahaFeatExtSparseCpp::calcXext(double *Xext, const double *W, const int BatchBegin, const int BatchEnd) {
						
		int i,j1,j2,k;
				
		for (i=BatchBegin; i<BatchEnd; ++i) {//layer 1
			
		//Calculate distance		
			j1 = this->XIndptr[i];
			j2 = 0;
			while (true) {//layer 2
				
				if (this->XIndices[j1] > this->cIndices[j2]) {//layer 3
										
					//If cIndices[j2] is smaller than XIndices[j1], calculate the weighted Euclidian distance between c[j2] and 0.					
					Xext[i-BatchBegin] -= this->cData[j2]*this->cData[j2]*W[this->cIndices[j2]]*W[this->cIndices[j2]];
					++j2;
					
				} else if (this->XIndices[j1] < this->cIndices[j2]) {//layer 3
										
					//If cIndices[j2] is greter than XIndices[j1], calculate the weighted Euclidian distance between c[j2] and 0.					
					Xext[i-BatchBegin] -= this->XData[j1]*this->XData[j1]*W[this->XIndices[j1]]*W[this->XIndices[j1]];	
					++j1;
					
				} else {//layer 3
															
					//If the indices are equal, calculate the weighted Euclidian distance between X[j1] and c[j2]
					Xext[i-BatchBegin] -= (this->XData[j1] - this->cData[j2])*(this->XData[j1] - this->cData[j2])*W[this->XIndices[j1]]*W[this->XIndices[j1]];					
					++j1;
					++j2;			
						
				}//layer 3
								
				//Once either j1 or j2 is greater than the maximum possible value, we just increment the other and calculate its distance to 0
				if (j1 == this->XIndptr[i+1]) {//layer 3
					
					 for (; j2 < this->cIndptr[1]; ++j2) Xext[i-BatchBegin] -= this->cData[j2]*this->cData[j2]*W[this->cIndices[j2]]*W[this->cIndices[j2]];
					 break;
					 
				 }//layer 3
				 
				if (j2 == cIndptr[1]) {//layer 3
					
					 for (; j1 < this->XIndptr[i+1]; ++j1) Xext[i-BatchBegin] -= this->XData[j1]*this->XData[j1]*W[this->XIndices[j1]]*W[this->XIndices[j1]];
					 break;
					 
				 }//layer 3				 
			}//layer 2

			Xext[i-BatchBegin] = exp(Xext[i-BatchBegin]);
			
		}//layer 1 close
	
	}	
	
				
	//Calculate XMinusCSquaredData
	void RBFMahaFeatExtSparseCpp::calcXMinusCSquared(const int BatchBegin, const int BatchSize, const int BatchEnd, const int BatchNum) {
		
		int *calcXMinusCSquaredLength = (int*)calloc(this->J,sizeof(int));
		
		int i,j,k;
										
		//Determine length of each column
		for (i=this->XIndptr[BatchBegin]; i<this->XIndptr[BatchEnd]; ++i) 
			calcXMinusCSquaredLength[this->XIndices[i]] += 1;
		
		//Set XMinusCSquaredIndptr
		for (i=1; i<=this->J; ++i) this->XMinusCSquaredIndptr[BatchNum][i] = this->XMinusCSquaredIndptr[BatchNum][i-1] + calcXMinusCSquaredLength[i-1];
						
		free(calcXMinusCSquaredLength);
		
		calcXMinusCSquaredLength = (int*)calloc(this->J,sizeof(int));
			
		//Transpose submatrices of X
		for (i=0; i<BatchSize; ++i) for (j=this->XIndptr[BatchBegin + i]; j<this->XIndptr[BatchBegin + i + 1]; ++j) {//layer 1
				
			this->XMinusCSquaredIndices[BatchNum][this->XMinusCSquaredIndptr[BatchNum][this->XIndices[j]] + calcXMinusCSquaredLength[this->XIndices[j]]] = i;
			this->XMinusCSquaredData[BatchNum][this->XMinusCSquaredIndptr[BatchNum][this->XIndices[j]] + calcXMinusCSquaredLength[this->XIndices[j]]] = this->XData[j];	
			++calcXMinusCSquaredLength[this->XIndices[j]];		
					
		}//layer 1 close
		
		free(calcXMinusCSquaredLength);
		
		//The resulting matrix is simply a transpose of X. But we want (x-c)*(x-c), where x <> 0. Thus, we do the following:
		//Find all c that are unequal to zero. Set the corresponding x to (x-c)*(x-c). Set all others to x*x.
		k = 0;
		for (j=0; j<this->J; ++j) {//layer 1
			if (j == this->cIndices[k]) {//layer 2
				
				//Case 1: Non-zero element in the centre
				for (i=this->XMinusCSquaredIndptr[BatchNum][j]; i<this->XMinusCSquaredIndptr[BatchNum][j+1]; ++i) 
						this->XMinusCSquaredData[BatchNum][i] =  (this->XMinusCSquaredData[BatchNum][i] - cData[k])*(this->XMinusCSquaredData[BatchNum][i] - cData[k]);
						
				//Increment k, if there still is another non-zero element in the centre
				if (k < cIndptr[1]) ++k;
				
			} else {//layer 2
				
				//Case 2: Corresponding element in the centre is zero		
				for (i=this->XMinusCSquaredIndptr[BatchNum][j]; i<this->XMinusCSquaredIndptr[BatchNum][j+1]; ++i) this->XMinusCSquaredData[BatchNum][i] *=  this->XMinusCSquaredData[BatchNum][i];
				
			}//layer 2 close
		}//layer 1 close
								
	}
	
	//Calculate sumY 
	void RBFMahaFeatExtSparseCpp::reduce1(const int BatchBegin, const int BatchEnd, const int BatchNum) {
		
		int i;
		
		this->LocalsumY[BatchNum] = 0.0;
		
		for (i=BatchBegin; i<BatchEnd; ++i) {this->LocalsumY[BatchNum] += this->Y[i];}
								
	}	
	
	//Calculate sumYY
	void RBFMahaFeatExtSparseCpp::reduce2(const int BatchBegin, const int BatchEnd, const int BatchNum) {
		
		int i;
		
		this->LocalsumYY[BatchNum] = 0.0;
		
		for (i=BatchBegin; i<BatchEnd; ++i) {this->LocalsumYY[BatchNum] += this->Y[i]*this->Y[i];}
	
	}												
		
	
	//Calculate sumXext
	void RBFMahaFeatExtSparseCpp::reduce3(const double *Xext, const int BatchSize) {
				
		int i;
				
		this->LocalsumXext[0] = 0.0;
			
		for (i=0; i<BatchSize; ++i) this->LocalsumXext[0] += Xext[i];
															
	}		

	//Calculate sumXextY
	void RBFMahaFeatExtSparseCpp::reduce4(const double *Xext, const double *y, const int BatchSize) {
				
		int i;
				
		this->LocalsumXextY[0] = 0.0;
			
		for (i=0; i<BatchSize; ++i) this->LocalsumXextY[0] += Xext[i]*y[i];
										
	}		
		
	//Calculate sumXextXext
	void RBFMahaFeatExtSparseCpp::reduce5(const double *Xext, const int BatchSize) {
				
		int i;
				
		this->LocalsumXextXext[0] = 0.0;
		
		for (i=0; i<BatchSize; ++i) this->LocalsumXextXext[0] += Xext[i]*Xext[i];
			
	}			
	
	//Calculate LocalsumdXextdW, LocalsumdXextdWY and LocalsumdXextdWXext
	void RBFMahaFeatExtSparseCpp::reduce6to8(const double *Xext, const double *y, const double *W, const int BatchBegin, const int BatchNum) {
			
		int i, j1, j2=0;
		double TemporaryValue;
		
		//Since (x-c)*(x-c) is inverted, we iterate over the instances, not the features! This is more efficient.
		for (j1=0; j1<this->J; ++j1) {//layer 1	
								
				//There are two possible cases: Either c[j1] is zero or it isn't.
				//First case: c[j1] is zero
				if (j1 != cIndices[j2]) {//layer 2
					
					this->LocalsumdXextdW[j1] = 0.0;
					this->LocalsumdXextdWY[j1] = 0.0;
					this->LocalsumdXextdWXext[0][j1] = 0.0;					
					
					for (i=this->XMinusCSquaredIndptr[BatchNum][j1]; i< this->XMinusCSquaredIndptr[BatchNum][j1+1]; ++i) {//layer 3
					
						this->LocalsumdXextdW[j1] += TemporaryValue = (-2.0)*this->XMinusCSquaredData[BatchNum][i]*Xext[this->XMinusCSquaredIndices[BatchNum][i]]*W[j1];
						this->LocalsumdXextdWXext[0][j1] += TemporaryValue*Xext[this->XMinusCSquaredIndices[BatchNum][i]];						
						this->LocalsumdXextdWY[j1] += TemporaryValue*y[i];
						
				} //layer 3 close

			//Second case: c[j1] is not zero
			} else {//layer 2
				
				//Whereever x[i] is zero, (x-c)*(x-c) is equal to c*c. Since we assume this to be the case most of the time, we simply initialise to the sum assuming all c is 0 to all cases and the substract the difference where this assumption fails:
				
				this->LocalsumdXextdW[j1] = (-2.0)*this->LocalsumXext[0]*W[j1]*this->cData[j2]*this->cData[j2];
				this->LocalsumdXextdWY[j1] = (-2.0)*this->LocalsumXextY[0]*W[j1]*this->cData[j2]*this->cData[j2];
				this->LocalsumdXextdWXext[0][j1] = (-2.0)*this->LocalsumXextXext[0]*W[j1]*this->cData[j2]*this->cData[j2];							
				
				for (i=this->XMinusCSquaredIndptr[BatchNum][j1]; i< this->XMinusCSquaredIndptr[BatchNum][j1+1]; ++i) {//layer 3 
				
						this->LocalsumdXextdW[j1] += TemporaryValue = (-2.0)*(this->XMinusCSquaredData[BatchNum][i] - this->cData[j2]*this->cData[j2])*Xext[this->XMinusCSquaredIndices[BatchNum][i]]*W[j1];
						this->LocalsumdXextdWXext[0][j1] += TemporaryValue*Xext[this->XMinusCSquaredIndices[BatchNum][i]];						
						this->LocalsumdXextdWY[j1] += TemporaryValue*y[i];				
					
				} //layer 3 close
				
				if (j2 < cIndptr[1] - 1) ++j2;
				
			}//layer 2	close
		}	//layer 1	 close
						
			
						
	}	
	
