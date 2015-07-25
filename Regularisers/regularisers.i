class RegulariserCpp {

	public:

	RegulariserCpp(double alpha);
	
	virtual ~RegulariserCpp() {}
	
};

class L1RegulariserCpp: public RegulariserCpp {
	
	public:
		
	L1RegulariserCpp (double alpha):RegulariserCpp(alpha);

	~L1RegulariserCpp();
				
};

class L2RegulariserCpp: public RegulariserCpp {
	
	public:
		
	L2RegulariserCpp (double alpha):RegulariserCpp(alpha);

	~L2RegulariserCpp();
				
};
