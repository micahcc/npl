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
#include "slicer.h"
#include <iostream>
#include <list>
#include <ctime>

using namespace std;
using namespace npl;

int main()
{
	NDArrayStore<3, float> test1({100,100,100});
	cerr << "Bytes: " << test1.bytes() << endl;

	for(size_t ii = 0; ii < test1.elements(); ii++)
		test1[ii] = ii;
	
	NDArray* testp = &test1;
	clock_t t;
	
	cerr << "Dimensions:" << testp->ndim() << endl;

	cerr << "Using Acessors" << endl;
	double total = 0;
	t = clock();
	for(int64_t zz=0; zz < testp->dim(2); zz++) {
		for(int64_t yy=0; yy < testp->dim(1); yy++) {
			for(int64_t xx=0; xx < testp->dim(0); xx++) {
				total += testp->get_dbl({xx,yy,zz});
			}
		}
	}
	t = clock()-t;
	cerr << "xyz: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";
    
	t = clock();
	for(int64_t xx=0; xx < testp->dim(0); xx++) {
		for(int64_t yy=0; yy < testp->dim(1); yy++) {
			for(int64_t zz=0; zz < testp->dim(2); zz++) {
				total += testp->get_dbl({xx,yy,zz});
			}
		}
	}
	t = clock()-t;
	cerr << "zyx: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";
	
	cerr << "Using Iterator" << endl;
	// forward order
	std::list<size_t> order({2,1,0});
	total = 0;
	t = clock();
	for(Slicer it(testp->ndim(), testp->dim(), order); !it.isEnd(); ++it) {
		total += testp->get_dbl(*it);
	}
	t = clock()-t;
	std::cout << "zyx: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";
    
	// reverse order of dimensions (x,y,z), not reverse iterator
	t = clock();
	for(Slicer it(testp->ndim(), testp->dim(), order, true); !it.isEnd(); ++it) {
		total += testp->get_dbl(*it);
	}
	t = clock()-t;
	std::cout << "xyz: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";
	
	Slicer it(testp->ndim(), testp->dim());
	
	it.setOrder(order);
	it.goBegin();
	cerr << "Comparing Outputs, in zyx ordering" << endl;
	// forward order
	for(int64_t xx=0; xx < testp->dim(0); xx++) {
		for(int64_t yy=0; yy < testp->dim(1); yy++) {
			for(int64_t zz=0; zz < testp->dim(2); zz++) {
				if(it.isEnd()) {
					cerr << "Error, iterator ended early" << endl;
					return -1;
				}

				double dirv = testp->get_dbl({xx,yy,zz});
				double itev = testp->get_dbl(*it);
				
				if(dirv != itev) {
					cerr << "Methods disagree, index: " << endl;
					std::vector<int64_t> index;
					index = it.index();
					for(auto val : index) {
						cerr << val << ",";
					}

					cerr << "\n" << xx << "," << yy << "," << zz << endl;
					cerr << dirv << " vs " << itev << " = " << itev-dirv << endl;
					return -2;
				}
				++it;
			}
		}
	}
    
	// reverse order
	cerr << "Comparing Outputs, in xyz ordering" << endl;
	it.setOrder(order, true);
	it.goBegin();
	for(int64_t zz=0; zz < testp->dim(2); zz++) {
		for(int64_t yy=0; yy < testp->dim(1); yy++) {
			for(int64_t xx=0; xx < testp->dim(0); xx++) {
				if(it.isEnd()) {
					cerr << "Error, iterator ended early" << endl;
					return -1;
				}

				double dirv = testp->get_dbl({xx,yy,zz});
				double itev = testp->get_dbl(*it);
				
				if(dirv != itev) {
					cerr << "Methods disagree, index: " << endl;
					std::vector<int64_t> index;
					index = it.index();
					for(auto val : index) {
						cerr << val << ",";
					}

					cerr << "\n" << xx << "," << yy << "," << zz << endl;
					cerr << dirv << " vs " << itev << " = " << itev-dirv << endl;
					return -2;
				}
				++it;
			}
		}
	}
	if(!it.isEnd()) {
		cerr << "Error, iterator ended late" << endl;
		return -1;
	}
	
	return 0;
}

