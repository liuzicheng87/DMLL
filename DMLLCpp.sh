swig -c++ -python DMLLCpp.i
mpic++ -O2 -fPIC -std=c++11 -c DMLLCpp_wrap.cxx -I/usr/include/python2.7
mpic++ -shared DMLLCpp_wrap.o -o _DMLLCpp.so
