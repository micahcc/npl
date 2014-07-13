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

int64_t clamp(int64_t low, int64_t hi, int64_t v)
{
	return std::max(low, std::min(hi, v));
}
int main()
{
	NDArrayStore<3, float> test1({100,100,100});
	cerr << "Bytes: " << test1.bytes() << endl;

	for(size_t ii = 0; ii < test1.elements(); ii++)
		test1[ii] = rand();
	
	NDArray* testp = &test1;
	clock_t t;
	
	cerr << "Comparing Acessors" << endl;
	double total = 0;
	t = clock();
	for(size_t zz=0; zz < testp->dim(2); zz++) {
		for(size_t yy=0; yy < testp->dim(1); yy++) {
			for(size_t xx=0; xx < testp->dim(0); xx++) {
				size_t ind[3] = {xx,yy,zz};
				if(testp->get_dbl({xx,yy,zz}) != testp->get_dbl(3, ind)) {
					cerr << "Error, difference between accessors!" << endl;
					return -1;
				}
			}
		}
	}
	t = clock()-t;
	cerr << "xyz: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";
    
	cerr << "Iterator Offset Acessors with Direct Accessor" << endl;
	total = 0;
	t = clock();
	NDArray::iterator it(testp);
	for(size_t xx=0; xx < testp->dim(0); xx++) {
		for(size_t yy=0; yy < testp->dim(1); yy++) {
			for(size_t zz=0; zz < testp->dim(1); zz++, ++it) {
				for(int64_t xi = -1; xi < 1; xi++) {
					for(int64_t yi = -1; yi < 1; yi++) {
						for(int64_t zi = -1; zi < 1; zi++) {
							size_t realx = clamp(0, testp->dim(0), xx+xi);
							size_t realy = clamp(0, testp->dim(0), yy+yi);
							size_t realz = clamp(0, testp->dim(0), zz+zi);
							if(testp->get_dbl({realx,realy,realz}) != it.get_dbl({xi,yi,zi})) {
								cerr << "Error, difference between accessors!" << endl;
								return -1;
							}

						}
					}
				}
			}
		}
	}
	t = clock()-t;
	cerr << "xyz: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";
	
	cerr << "Speed with with iterator/offset" << endl;
	double val = 9;
	t = clock();
	for(it.goBegin(); !it.isEnd(); ++it) {
		for(int64_t xi = -1; xi < 1; xi++) {
			for(int64_t yi = -1; yi < 1; yi++) {
				for(int64_t zi = -1; zi < 1; zi++) {
					val += it.get_dbl({xi,yi,zi});
				}
			}
		}
	}
	t = clock()-t;
	cerr << "xyz: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";
	
	cerr << "Speed with with accessors" << endl;
	val = 9;
	t = clock();
	for(size_t xx=0; xx < testp->dim(0); xx++) {
		for(size_t yy=0; yy < testp->dim(1); yy++) {
			for(size_t zz=0; zz < testp->dim(1); zz++, ++it) {
				for(int64_t xi = -1; xi < 1; xi++) {
					for(int64_t yi = -1; yi < 1; yi++) {
						for(int64_t zi = -1; zi < 1; zi++) {
							size_t realx = clamp(0, testp->dim(0), xx+xi);
							size_t realy = clamp(0, testp->dim(0), yy+yi);
							size_t realz = clamp(0, testp->dim(0), zz+zi);
							val += testp->get_dbl({realx,realy,realz});

						}
					}
				}
			}
		}
	}
	t = clock()-t;
	cerr << "xyz: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";
	
	return 0;
}
