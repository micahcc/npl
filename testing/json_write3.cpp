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
 * @file nifti_orient_test1.cpp
 *
 *****************************************************************************/

#include <iostream>

#include "mrimage.h"
#include "nplio.h"
#include "mrimage_utils.h"
#include "iterators.h"

using namespace std;
using namespace npl;

int main()
{
	// create an image
	size_t sz[] = {2, 1, 4, 3};
    {
        auto in = createNDArray(sizeof(sz)/sizeof(size_t), sz, FLOAT64);

        NDIter<double> sit(in);
        for(int ii=0; !sit.eof(); ++sit, ++ii) {
            sit.set(ii);
        }
        in->write("test.json.gz");
    }

    {
        auto reread = readNDArray("test.json.gz");
        
        NDIter<double> sit(reread);
        for(int ii=0; !sit.eof(); ++sit, ++ii) {
            if(*sit != ii) {
                cerr << "Value Mismatch!" << endl; 
                return -1;
            }
        }

        if(reread->type() != FLOAT64) {
            cerr << "Type Mismatch!" << endl;
            return -1;
        }

    }
	return 0;
}



