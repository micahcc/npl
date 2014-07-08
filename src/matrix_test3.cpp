#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstddef>

#include "matrix.h"

using namespace npl;
using namespace std;

// determinant test
int main()
{
	Matrix<3,3> mat1;
	Matrix<4,4> mat2;
	
	auto t = clock();
	for(size_t rr=0; rr<3; rr++) {
		for(size_t cc=0; cc<3; cc++) {
			mat1(rr,cc) = rand()/(double)RAND_MAX;
		}
	}
	for(size_t rr=0; rr<4; rr++) {
		for(size_t cc=0; cc<4; cc++) {
			mat1(rr,cc) = rand()/(double)RAND_MAX;
		}
	}

	cerr << "det(" << mat1 << ") = " << determinant(mat1) << endl;
	cerr << "det(" << mat2 << ") = " << determinant(mat2) << endl;
}

