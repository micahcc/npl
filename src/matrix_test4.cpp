#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstddef>

#include "matrix.h"

using namespace npl;
using namespace std;

int main()
{
	double corvals[3] = {149.0,161.9,174.8};
	Matrix<3,3> mat1;
	Matrix<4,4> mat2;
	Matrix<3,1> vec;
	for(size_t rr=0; rr<3; rr++) {
		for(size_t cc=0; cc<3; cc++) {
			mat1(rr,cc) = rr+cc*10;
		}
		vec[rr] = 3.3+rr;
	}
	
	for(size_t rr=0; rr<4; rr++) {
		for(size_t cc=0; cc<4; cc++) {
			mat2(rr,cc) = rand()/(double)RAND_MAX;
		}
	}

	cerr << mat1 << "\n*\n" << vec << "\n=\n";
	MatrixP* mat1p = &mat1;
	MatrixP* mat2p = &mat2;
	MatrixP* vecp = &vec;
	
	// vecp should be converted by the matrix
	mat1p->mvproduct(vecp);
	cerr << vec << endl;

	for(size_t ii=0; ii<3; ii++) {
		// need to add check of correct values
		if(fabs(corvals[ii] - vec[ii]) > 0.00001) {
			cerr << "FAIL" << endl;
			return -1;
		}
	}
	
	// should runtime fail
	try {
	mat2p->mvproduct(vecp);
	} catch(...) {
		cerr << "PASS!" << endl;
	}
	
}


