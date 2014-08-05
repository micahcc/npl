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
 * @file matrix_test3.cpp
 *
 *****************************************************************************/

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

