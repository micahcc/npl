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
 * @file test1.cpp
 *
 *****************************************************************************/

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
		test1._m_data[ii] = ii;
	
	NDArray* testp = &test1;
	clock_t t;
	
	cerr << "Dimensions:" << testp->ndim() << endl;

	double total = 0;
	t = clock();
	for(size_t zz=0; zz < testp->dim(2); zz++) {
		for(size_t yy=0; yy < testp->dim(1); yy++) {
			for(size_t xx=0; xx < testp->dim(0); xx++) {
				total += testp->dbl({xx,yy,zz});
//				cerr << testp->getD(xx,yy,zz) << endl;
//				cerr << (*testp)(xx,yy,zz);
			}
		}
	}
	t = clock()-t;
	std::cout << "zyx: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";

	t = clock();
	for(size_t xx=0; xx < testp->dim(0); xx++) {
		for(size_t yy=0; yy < testp->dim(1); yy++) {
			for(size_t zz=0; zz < testp->dim(2); zz++) {
				total += testp->dbl({xx,yy,zz});
//				cerr << testp->getD(xx,yy,zz) << endl;
//				cerr << (*testp)(xx,yy,zz);
			}
		}
	}
	t = clock()-t;
	std::cout << "xyz: " << ((double)t)/CLOCKS_PER_SEC << " s.\n";
}
