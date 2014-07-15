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

#include "ndarray.h"
#include <iostream>
#include <ctime>

using namespace std;
using namespace npl;

int main()
{
	NDArrayStore<3, float> test1({100,1000,1000});
	cerr << "Bytes: " << test1.bytes() << endl;

	for(size_t ii = 0; ii < test1.bytes()/sizeof(float); ii++)
		test1[ii] = ii;
	
	NDArray* testp = &test1;
	clock_t t;
	
	cerr << "Dimensions:" << testp->ndim() << endl;

	double total = 0;
	t = clock();
	for(int64_t zz=0; zz < testp->dim(2); zz++) {
		for(int64_t yy=0; yy < testp->dim(1); yy++) {
			for(int64_t xx=0; xx < testp->dim(0); xx++) {
				total += testp->get_dbl({xx,yy,zz});
//				cerr << testp->getD(xx,yy,zz) << endl;
//				cerr << (*testp)(xx,yy,zz);
			}
		}
	}
	t = clock()-t;
	std::cout << "xyz: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";
    
	t = clock();
	for(int64_t xx=0; xx < testp->dim(0); xx++) {
		for(int64_t yy=0; yy < testp->dim(1); yy++) {
			for(int64_t zz=0; zz < testp->dim(2); zz++) {
				total += testp->get_dbl({xx,yy,zz});
//				cerr << testp->getD(xx,yy,zz) << endl;
//				cerr << (*testp)(xx,yy,zz);
			}
		}
	}
	t = clock()-t;
	std::cout << "zyx: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";

	return 0;
}
