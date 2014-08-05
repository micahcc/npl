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
 * @file matrix_test1.cpp
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
	Matrix<3,3> mat1;
	Matrix<3,3> mat2;
	Matrix<3,1> vec;
	for(size_t rr=0; rr<3; rr++) {
		for(size_t cc=0; cc<3; cc++) {
			mat1(rr,cc) = rand()/(double)RAND_MAX;
			mat2(rr,cc) = rand()/(double)RAND_MAX;
		}
		vec[rr] = rand()/(double)RAND_MAX;
	}

	auto mat3 = mat1*mat2;
	cerr << mat1 << " * " << mat2 << " = " << mat3 << endl;

	auto mat4 = mat3*vec;
	cerr << mat3 << " * " << vec << " = " << mat4 << endl;

//	auto mat5 = vec*mat3; //error conflicting types
	
	Matrix<2,2> mat5;
	for(size_t rr=0; rr<2; rr++) {
		for(size_t cc=0; cc<2; cc++) {
			mat5(rr,cc) = rand()/(double)RAND_MAX;
		}
	}

	auto mat6 = inverse(mat5);
	cerr << mat5 << "^-1 = " << endl << mat6 << endl;
	
	auto mat7 = mat1;
	auto mat1I = inverse(mat1);
	cerr << mat1 << "^-1 = " << endl << mat1I << endl;

	auto mat8 = mat7*mat1I;
	cerr << "Identity? " << endl << mat8 << endl;
	auto mat9 = mat1I*mat7;
	cerr << "Identity? " << endl << mat9 << endl;
}

