%module DMLLCpp
%{
#define SWIG_FILE_WITH_INIT
#include <mpi.h>
#include "DMLLCpp.cpp"
%}

%include mpi4py/mpi4py.i
%mpi4py_typemap(Comm, MPI_Comm);

%include "numpy.i"

%init %{
import_array();
%}

%apply (double* IN_ARRAY1, int DIM1) {(double* X, int I)};
%apply (double* IN_ARRAY1, int DIM1) {(double* X, int I)};
%apply (double* IN_ARRAY1, int DIM1) {(double *XData, int XDataLength)};
%apply (int* IN_ARRAY1, int DIM1) {(int *XIndices, int XIndicesLength)};
%apply (int* IN_ARRAY1, int DIM1) {(int *XIndptr, int XIndptrLength)};
%apply (double* IN_ARRAY1, int DIM1) {(double *cData, int cDataLength)};
%apply (int* IN_ARRAY1, int DIM1) {(int *cIndices, int cIndicesLength)};
%apply (int* IN_ARRAY1, int DIM1) {(int *cIndptr, int cIndptrLength)};
%apply (int* IN_ARRAY1, int DIM1) {(int* X, int I)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2 ) {(double* X, int I, int J)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2 ) {(double *Xext, int Jext, int I)};
%apply (double* IN_ARRAY1, int DIM1) {(double* Y, int IY)};
%apply (double* IN_ARRAY1, int DIM1) {(double* Yhat, int IY)};
%apply (double* IN_ARRAY1, int DIM1) {(double *SumGradients, int IterationsNeeded)};
%apply (double* IN_ARRAY1, int DIM1) {(double* W, int lengthW)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2 ) {(double* GlobalX, int GlobalI, int GlobalJ)};
%apply (double* IN_ARRAY1, int DIM1) {(double* GlobalX, int GlobalI)};
%apply (int* IN_ARRAY1, int DIM1) {(int* LocalI, int SizeLocalI)};
%apply (int* IN_ARRAY1, int DIM1) {(int* CumulativeLocalI, int SizeCumulativeLocalI)};

%include UsefulFunctions/UsefulFunctions.i
%include optimisers/optimisers.i
%include Regularisers/regularisers.i

%include linear/linear.i
%include DimensionalityReduction/DimensionalityReduction.i
