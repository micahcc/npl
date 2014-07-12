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
