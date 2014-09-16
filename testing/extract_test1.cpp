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
 * @file nifti_test1.cpp
 *
 *****************************************************************************/

#include <iostream>
#include <memory>

#include "mrimage.h"
#include "iterators.h"

using namespace std;
using namespace npl;

int main()
{
	vector<size_t> imagesz({10, 12, 14, 16});
	auto img1 = createMRImage(imagesz.size(), imagesz.data(), FLOAT32);
	vector<int64_t> index(imagesz.size());
	
	for(NDIter<double> it(img1); !it.eof(); ++it) {
		it.index(index);
		it.set(index[3]);
	}

	vector<size_t> roi(img1->dim(), img1->dim()+img1->ndim());
	roi[3] = 0;

	std::fill(index.begin(), index.end(), 0);
	for(size_t ii=0; ii<imagesz[3]; ii++) {
		index[3] = ii;
		auto img2 = img1->extractCast(index.size(), index.data(), roi.data());

		size_t count=0;
		for(NDIter<double> it(img2); !it.eof(); ++it) {
			if(ii != *it) {
				cerr << "Error, extracted a wrong value" << endl;
				return -1;
			}
			count++;
		}
		if(count != imagesz[0]*imagesz[1]*imagesz[2]) {
			cerr << "Error, count different!" << endl;
			return -1;
		}
	}

	return 0;
}
