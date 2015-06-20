	
	//Calculate Xext
	void LinearMahaFeatExtSparseCpp::calcXext(double *Xext, const double *W, const int BatchBegin, const int BatchSize) {
						
		int i,j,k;
		
		for (j=0; j<this->Jext; ++j) for (i=0; i<BatchSize; ++i) 
			for (k=this->XIndptr[BatchBegin + i]; k<this->XIndptr[BatchBegin + i + 1]; ++k)
				Xext[j*BatchSize + i] += this->XData[k]*W[j*J + this->XIndices[k]];
						
	}	
	
				
	//Calculate dXextdWData
	void LinearMahaFeatExtSparseCpp::calcdXextdW(const int BatchBegin, const int BatchSize, const int BatchEnd, const int BatchNum) {
		
		int *dXextdWLength = (int*)calloc(this->J,sizeof(int));
		
		int i,j;
										
		//Determine length of each column
		for (i=this->XIndptr[BatchBegin]; i<this->XIndptr[BatchEnd]; ++i) 
			dXextdWLength[this->XIndices[i]] += 1;
		
		//Set dXextdWDataIndptr
		for (i=1; i<=this->J; ++i) this->dXextdWIndptr[BatchNum][i] = this->dXextdWIndptr[BatchNum][i-1] + dXextdWLength[i-1];
						
		free(dXextdWLength);
		
		dXextdWLength = (int*)calloc(this->J,sizeof(int));
			
		//Transpose submatrices of X
		for (i=0; i<BatchSize; ++i) for (j=this->XIndptr[BatchBegin + i]; j<this->XIndptr[BatchBegin + i + 1]; ++j) {//layer 1
				
			this->dXextdWIndices[BatchNum][this->dXextdWIndptr[BatchNum][this->XIndices[j]] + dXextdWLength[this->XIndices[j]]] = i;
			this->dXextdWData[BatchNum][this->dXextdWIndptr[BatchNum][this->XIndices[j]] + dXextdWLength[this->XIndices[j]]] = this->XData[j];	
			++dXextdWLength[this->XIndices[j]];		
					
		}//layer 1 close
		
		free(dXextdWLength);
		
								
	}
	
	//Calculate sumY and sumYY
	void LinearMahaFeatExtSparseCpp::reduce1(const double *y, int BatchSize, double &LocalsumY, double &LocalsumYY) {
		
		int i;
		
		LocalsumY=0.0;
		LocalsumYY=0.0;
		
		for (i=0; i<BatchSize; ++i) {LocalsumY += y[i]; LocalsumYY += y[i]*y[i];}
								
	}					
		
	//Calculate sumdXextdW
	void LinearMahaFeatExtSparseCpp::reduce2(const int BatchNum) {
						
		int k,c;
		for (c=0; c<this->J;++c) {//layer 1
			
			this->LocalsumdXextdW[BatchNum][c] = 0.0;
						
			for (k=this->dXextdWIndptr[BatchNum][c]; k<this->dXextdWIndptr[BatchNum][c + 1]; ++k) this->LocalsumdXextdW[BatchNum][c] += this->dXextdWData[BatchNum][k];
													
		}//layer 1	
						
	}	
	
	//Calculate sumdXextdWY
	void LinearMahaFeatExtSparseCpp::reduce3(const int BatchBegin, const int BatchNum) {
						
		int k,c;
		for (c=0; c<this->J;++c) {//layer 1
			
			this->LocalsumdXextdWY[BatchNum][c] = 0.0;
			
			for (k=this->dXextdWIndptr[BatchNum][c]; k<this->dXextdWIndptr[BatchNum][c + 1]; ++k) 
				this->LocalsumdXextdWY[BatchNum][c] += this->dXextdWData[BatchNum][k]*this->Y[BatchBegin + this->dXextdWIndices[BatchNum][k]];
							
		}//layer 1	
						
	}						
	
	//Calculate sumXext
	void LinearMahaFeatExtSparseCpp::reduce4(const double *Xext, const int BatchSize) {
				
		int i, a;
				
		for (a=0; a<this->Jext; ++a) {//layer 1
			
			this->LocalsumXext[a] = 0.0;
			
			for (i=0; i<BatchSize; ++i) this->LocalsumXext[a] += Xext[a*BatchSize + i];
													
		}//layer 1 close
			
	}		

	//Calculate sumXextY
	void LinearMahaFeatExtSparseCpp::reduce5(const double *Xext, const double *y, const int BatchSize) {
				
		int i, a;
				
		for (a=0; a<this->Jext; ++a) {//layer 1
			
			this->LocalsumXextY[a] = 0.0;
			
			for (i=0; i<BatchSize; ++i) this->LocalsumXextY[a] += Xext[a*BatchSize + i]*y[i];
							
		}//layer 1 close
			
	}		
		
	//Calculate sumXextXext
	void LinearMahaFeatExtSparseCpp::reduce6(const double *Xext, const int BatchSize) {
				
		int i, a1, a2;
				
		for (a1=0; a1<this->Jext; ++a1) for (a2=0; a2<=a1; ++a2) {//layer 1
			
			this->LocalsumXextXext[(a1*(a1+1))/2 + a2] = 0.0;
			for (i=0; i<BatchSize; ++i) this->LocalsumXextXext[(a1*(a1+1))/2 + a2] += Xext[a1*BatchSize + i]*Xext[a2*BatchSize + i];
			
				
		}//layer 1 close
			
	}			
			
	//Calculate sumXextdWXext
	void LinearMahaFeatExtSparseCpp::reduce7(const double *Xext, const int BatchSize,  const int BatchNum) {
						
		int k,c,d2;
		for (d2=0; d2<this->Jext; ++d2) for (c=0; c<this->J;++c) {//layer 1
						
			this->LocalsumdXextdWXext[d2][c] = 0.0;
			
			for (k=this->dXextdWIndptr[BatchNum][c]; k<this->dXextdWIndptr[BatchNum][c + 1]; ++k) 
				this->LocalsumdXextdWXext[d2][c] += this->dXextdWData[BatchNum][k]*Xext[d2*BatchSize + this->dXextdWIndices[BatchNum][k]];
							
		}//layer 1	
						
	}					
