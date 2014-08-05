/******************************************************************************
 * Copyright 2014 Micah C Chambers (micahc.vt@gmail.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file matrix_test4.cpp
 *
 *****************************************************************************/

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
	Matrix<3,1> ovec;
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
	MatrixP* ovecp = &ovec;
	
	// vecp should be converted by the matrix
	mat1p->mvproduct(vecp, &ovec);
	cerr << ovec << endl;

	for(size_t ii=0; ii<3; ii++) {
		// need to add check of correct values
		if(fabs(corvals[ii] - ovec[ii]) > 0.00001) {
			cerr << "FAIL" << endl;
			return -1;
		}
	}
	
	// should runtime fail
	try {
	mat2p->mvproduct(vecp, ovecp);
	} catch(...) {
		cerr << "Should have received a dynamic_cast error, you did, so"
			<< endl << "PASS!" << endl;
	}
	
}


