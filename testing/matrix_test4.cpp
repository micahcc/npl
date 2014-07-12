/*******************************************************************************
This file is part of Neural Program Library (NPL), 

Written and Copyrighted by by Micah C. Chambers (micahc.vt@gmail.com)

The Neural Program Library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The Neural Programs and Libraries are distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along with
the Neural Programs Library.  If not, see <http://www.gnu.org/licenses/>.
*******************************************************************************/

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


