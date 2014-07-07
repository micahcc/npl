#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstddef>

#include "matrix.h"

using namespace npl;

// speed test
int main()
{
	const int iters = 100000000;
	Matrix<3,3> mat1;
	Matrix<3,1> vec;
	Matrix<3,1> acc(0);
	
	auto t = clock();
	for(size_t rr=0; rr<3; rr++) {
		for(size_t cc=0; cc<3; cc++) {
			mat1(rr,cc) = rand()/(double)RAND_MAX;
		}
		vec[rr] = rand()/(double)RAND_MAX;
	}

	t = clock();
	for(size_t ii=0; ii<iters; ii++) {

		// new vector value
		for(size_t rr=0; rr<3; rr++) {
			vec[rr] = rand()/(double)RAND_MAX;
		}
		acc += mat1*vec;
	}
	t = clock() - t;
	printf("%li clicks (%f seconds)\n",t,((float)t)/CLOCKS_PER_SEC);

}
