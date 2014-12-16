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
 * @file txt_write1.cpp Creates a new image then reads it back
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"
#include "nplio.h"
#include "iterators.h"

using namespace std;
using namespace npl;

int main()
{
	// create an image
	size_t sz[] = {3, 1, 1, 10};

	{
		auto in = createMRImage(sizeof(sz)/sizeof(size_t), sz, FLOAT64);

		NDIter<double> sit(in);
		for(int ii=0; !sit.eof(); ++sit, ++ii) {
			sit.set(ii);
		}
		in->write("test_txt1.csv");
	}

	{
		auto reread = dPtrCast<MRImage>(readMRImage("test_txt1.csv"));

		NDIter<double> sit(reread);
		for(int ii=0; !sit.eof(); ++sit, ++ii) {
			if(*sit != ii) {
				cerr << "Value Mismatch! " << ii << " vs " << *sit << endl; 
				return -1;
			}
		}

	}
	return 0;
}



