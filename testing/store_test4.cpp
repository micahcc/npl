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
 * @file store_test4.cpp
 *
 *****************************************************************************/

#include "ndarray.h"
#include <iostream>
#include <set>
#include <map>

using namespace std;
using namespace npl;

inline
int64_t clamp(int64_t ii, int64_t low, int64_t high)
{
	return std::min<int64_t>(high, std::max<int64_t>(ii,low));
}


int main(int argc, char** argv)
{
	if(argc != 2) {
		cerr << "Need one argument (the size of clusters)" << endl;
		return -1;
	}

	//size_t csize = atoi(argv[1]);
	// need to produce a cache based on this

	////////////////////////////////////////////////////////////////////////////////////
	cerr << "4D Test" << endl;
	NDArrayStore<4, float> array1({500, 500, 1, 1});
	NDArrayStore<4, float> array2({500, 500, 1, 1});

	cerr << "Filling..." << endl;
	auto t = clock();
	for(size_t ii = 0; ii < array1._m_dim[0]; ii++) {
		for(size_t jj = 0; jj < array1._m_dim[1]; jj++) {
			for(size_t kk = 0; kk < array1._m_dim[2]; kk++) {
				for(size_t tt = 0; tt < array1._m_dim[3]; tt++) {
					double val = rand()/(double)RAND_MAX;
					array1.dbl({ii, jj, kk, tt}, val);
				}
			}
		}
	}
	t = clock() - t;
	printf("Fill: %li clicks (%f seconds)\n",t,((float)t)/CLOCKS_PER_SEC);
	
	int64_t radius = 4;
	t = clock();
	for(size_t tt = 0; tt < array1._m_dim[3]; tt++) {
		for(size_t kk = 0; kk < array1._m_dim[2]; kk++) {
			for(size_t jj = 0; jj < array1._m_dim[1]; jj++) {
				for(size_t ii = 0; ii < array1._m_dim[0]; ii++) {
					double sum = 0;
					double n = 0;
					for(int64_t xx=-radius; xx<=radius ; xx++) {
						for(int64_t yy=-radius; yy<=radius ; yy++) {
							for(int64_t zz=-radius; zz<=radius ; zz++) {
								for(int64_t rr=-radius; rr<=radius ; rr++) {
									size_t ix = clamp(ii+xx, 0, array1._m_dim[0]-1);
									size_t jy = clamp(jj+yy, 0, array1._m_dim[1]-1);
									size_t kz = clamp(kk+zz, 0, array1._m_dim[2]-1);
									size_t tr = clamp(tt+rr, 0, array1._m_dim[2]-1);
									sum += array1.dbl({ix, jy, kz, tr});
									n++;
								}
							}
						}
					}
					array2.dbl({ii, jj, kk, tt}, sum/n);
				}
			}
		}
	}
	t = clock() - t;
	printf("Radius %li Kernel: %li clicks (%f seconds)\n",radius, t,((float)t)/CLOCKS_PER_SEC);
}



