/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * NPL is free software: you can redistribute it and/or modify it under the
 * terms of the BSD 2-Clause License available in LICENSE or at
 * http://opensource.org/licenses/BSD-2-Clause
 *
 * @file matrix_test2.cpp
 *
 *****************************************************************************/

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
	Matrix<3,1> acc(0.0);

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
