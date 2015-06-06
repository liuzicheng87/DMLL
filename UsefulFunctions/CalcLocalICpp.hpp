//LocalI can be used to share any burden evenly among the processes
void CalcLocalICpp (int *LocalI, int SizeLocalI, int *CumulativeLocalI, int SizeCumulativeLocalI, const int GlobalI) {
	
		int i;
		
		CumulativeLocalI[0] = 0;
		
		//We divide LengthLocal as evenly as possible. The remainder is given to the process with the highest rank. 
		for (i=0; i<SizeLocalI-1; ++i) {LocalI[i] = GlobalI/SizeLocalI; CumulativeLocalI[i+1] = CumulativeLocalI[i] + LocalI[i];}
		LocalI[SizeLocalI-1] = GlobalI - CumulativeLocalI[SizeLocalI-1];
		
}
