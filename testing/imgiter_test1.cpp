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
 * @file imgiter_test1.cpp
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"
#include "mrimage_utils.h"
#include "iterators.h"

using namespace std;
using namespace npl;

int main()
{
	auto img = readMRImage("../../data/grad_imag.nii.gz");
	OrderIter<int> it(img);
	it.setOrder({0,1,2});
	it.goBegin();
	for(size_t ii=1; ii<=125; ii++, ++it) {
		if(*it != ii) {
			cerr << "Difference between read image and theoretical image" << endl;
			return -1;
		}
	}
	
	it.setOrder({}, true);
	it.goBegin();
	for(size_t ii=1; ii<=125; ii++, ++it) {
		if(*it != ii) {
			cerr << "Difference between read image and theoretical image"
				" when using reversed default order" << endl;
			return -1;
		}
	}
	
	return 0;
}

